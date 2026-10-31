#include "gdn-wmma.cuh"

#include <cstdlib>

// The gated delta rule of a prompt batch in chunks of 16 tokens, on the RDNA3 matrix cores: Strata's gfx11 kernels
// (src/prefill/kernels.cu, gdn_wmma_prep_kernel and gdn_wmma_scan_kernel, MIT, Niko1221 and the Strata
// contributors), read by strides from our q, k, v, gate and beta, with our transposed state, and without the output
// norm (our gated norm follows). Per value head, gamma_t the gate's sum from the chunk's start to t (inclusive) and
// S0 the state before the chunk:
//   A[t][i] = beta_t exp(gamma_t - gamma_i) k_t.k_i (i < t),  T = (I + A)^-1,  P[t][i] = exp(gamma_t - gamma_i) q_t.k_i (i <= t),
//   Y = beta (V - exp(gamma) K S0),  D = T Y,  o = exp(gamma) Q S0 + P D,
//   S = exp(gamma_last) S0 + K^T (exp(gamma_last - gamma) D).
// The prep kernel computes T and P of every chunk (they do not depend on the state); the scan walks the chunks, a
// block per value head and a wave per 16 value columns, the state in the waves' WMMA accumulators (FP32). k, q, T, P,
// D and the state as an operand are rounded to FP16.

#if defined(GGML_USE_HIP) && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
                              defined(__gfx1150__) || defined(__gfx1151__))
#define GDN_WMMA_DEVICE 1
#endif

namespace {

constexpr int GW_S   = 128;            // head size
constexpr int GW_HK  = 16;             // key heads
constexpr int GW_HV  = 48;             // value heads
constexpr int GW_CK  = 16;             // tokens per chunk (a WMMA tile)
constexpr int GW_NW  = GW_S / 16;      // the scan's waves per block: one per 16 value columns
constexpr int GW_VPK = GW_HV / GW_HK;  // value heads per key head: the prep's waves per block
constexpr int GW_KP  = GW_S + 8;       // the scan's padded FP16 k / q rows
constexpr int GW_AHEAD = 3;            // the scan warms the cache this many chunks ahead

#ifdef GGML_USE_HIP
typedef _Float16 gw_h16 __attribute__((ext_vector_type(16)));
typedef _Float16 gw_h8  __attribute__((ext_vector_type(8)));
typedef _Float16 gw_h2  __attribute__((ext_vector_type(2)));
typedef float    gw_f8  __attribute__((ext_vector_type(8)));
typedef float    gw_f4  __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ gw_f8 gw_mma(gw_h16 a, gw_h16 b, gw_f8 c) {
#ifdef GDN_WMMA_DEVICE
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    (void) a; (void) b;
    __builtin_trap();   // not an RDNA3 device pass: the host gate never launches it
    return c;
#endif
}

// An accumulator tile (lane l: rows 2v + (l >> 4) for v = 0..7, column l & 15) as a B operand (lane l: column l & 15,
// rows 0..15): pairs of rows rounded to FP16 and packed, the other half's pairs fetched across (permlanex16), then
// interleaved by a byte permute whose selector depends on the lane's half.
struct gw_sel_t { unsigned even, odd; };
static __device__ __forceinline__ gw_sel_t gw_sel(int hi) {
    return hi ? gw_sel_t{0x01000504u, 0x03020706u} : gw_sel_t{0x05040100u, 0x07060302u};
}
static __device__ __forceinline__ gw_h16 gw_acc_to_b(gw_f8 x, gw_sel_t ps) {
    unsigned wd[8];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const gw_h2 p = {(_Float16) x[2 * m], (_Float16) x[2 * m + 1]};
        const unsigned own = __builtin_bit_cast(unsigned, p);
#ifdef GDN_WMMA_DEVICE
        const unsigned other = __builtin_amdgcn_permlanex16(own, own, 0x76543210u, 0xfedcba98u, false, false);
#else
        const unsigned other = own;
#endif
        wd[2 * m]     = __builtin_amdgcn_perm(other, own, ps.even);
        wd[2 * m + 1] = __builtin_amdgcn_perm(other, own, ps.odd);
    }
    return __builtin_bit_cast(gw_h16, wd);
}
static __device__ __forceinline__ gw_h16 gw_ld(const _Float16 * p) {   // 32 bytes of LDS, 16-byte aligned
    const uint4 a = *reinterpret_cast<const uint4 *>(p), b = *reinterpret_cast<const uint4 *>(p + 8);
    const uint4 ab[2] = {a, b};
    return __builtin_bit_cast(gw_h16, ab);
}
// the wave's own LDS writes complete (read back across its lanes)
static __device__ __forceinline__ void gw_lds_wait() {
#ifdef GDN_WMMA_DEVICE
    __asm__ volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#endif
}
// a barrier over LDS only: __syncthreads() would also wait for the next chunk's global loads in flight
static __device__ __forceinline__ void gw_lds_barrier() {
#ifdef GDN_WMMA_DEVICE
    __asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
#else
    __syncthreads();
#endif
}

// A block per (key head, chunk), a wave per value head of the key head: T then P, FP16 row-major (1 KB), at
// tp[(chunk * HV + value head) * 512].
__global__ void __launch_bounds__(32 * GW_VPK) gdn_wmma_prep(const ggml_cuda_gdn_wmma_args a, _Float16 * __restrict__ tp) {
    __shared__ float sa[GW_VPK][GW_CK * GW_CK], sg[GW_VPK][GW_CK], sb[GW_VPK][GW_CK];
    const int qh = blockIdx.x, ch = blockIdx.y, w = threadIdx.x >> 5, l = threadIdx.x & 31, c = l & 15, hi = l >> 4;
    const int vh = qh + w * GW_HK;
    const int64_t t0 = (int64_t) ch * GW_CK, T = a.T;
    gw_f8 kk = {}, qk = {};   // k_t.k_i and q_t.k_i (lane: column i = c, rows t = 2v + hi)
    {
        const bool ok = t0 + c < T;   // past the end: zero rows
        const int64_t tt = ok ? t0 + c : 0;
        const float * kt = a.k + tt * a.sq2 + qh * a.sq1;
        const float * qt = a.q + tt * a.sq2 + qh * a.sq1;
#pragma unroll
        for (int j = 0; j < GW_S / 16; ++j) {
            gw_h16 ka, qa;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const gw_f4 k4 = *reinterpret_cast<const gw_f4 *>(kt + 16 * j + 4 * i);
                const gw_f4 q4 = *reinterpret_cast<const gw_f4 *>(qt + 16 * j + 4 * i);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    ka[4 * i + e] = (_Float16) (ok ? k4[e] : 0.0f);
                    qa[4 * i + e] = (_Float16) (ok ? q4[e] : 0.0f);
                }
            }
            kk = gw_mma(ka, ka, kk);   // B = K^T: lane i holds column i of K^T, row i of K - the A operand itself
            qk = gw_mma(qa, ka, qk);
        }
    }
    if (l < GW_CK) {   // past the end: gate 0, beta 0
        const bool ok = t0 + l < T;
        float g = ok ? a.g[(t0 + l) * a.sb2 + vh * a.sb1] : 0.0f;
#pragma unroll
        for (int o = 1; o < GW_CK; o <<= 1) {
            const float u = __shfl_up(g, o, GW_CK);
            if (l >= o) {
                g += u;
            }
        }
        sg[w][l] = g;
        sb[w][l] = ok ? a.beta[(t0 + l) * a.sb2 + vh * a.sb1] : 0.0f;
    }
    gw_lds_wait();
    _Float16 * out = tp + ((int64_t) ch * GW_HV + vh) * (2 * GW_CK * GW_CK);
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const int t = 2 * v + hi;
        sa[w][t * GW_CK + c] = c < t ? sb[w][t] * __expf(sg[w][t] - sg[w][c]) * kk[v] : 0.0f;
        out[GW_CK * GW_CK + t * GW_CK + c] = (_Float16) (c <= t ? __expf(sg[w][t] - sg[w][c]) * qk[v] : 0.0f);
    }
    gw_lds_wait();
    if (l < GW_CK) {   // lane c: column c of T = (I + A)^-1 by forward substitution
        float tc[GW_CK];
#pragma unroll
        for (int t = 0; t < GW_CK; ++t) {
            float v = t == c ? 1.0f : 0.0f;
#pragma unroll
            for (int k = 0; k < t; ++k) {
                v = fmaf(-sa[w][t * GW_CK + k], tc[k], v);
            }
            tc[t] = v;
            out[t * GW_CK + c] = (_Float16) v;
        }
    }
}

// A thread's slice of a chunk in the scan (token threadIdx / 16, columns (threadIdx % 16) * 8 .. + 7): k, q and v,
// loaded a chunk ahead, and one word of the chunk's T and P; lanes 0..15 of wave 0 also gate and beta of token l.
// Rows past the end read the last row (finite: their gate and beta become 0 at staging).
struct gw_chunk {
    gw_f4 k0, k1, q0, q1, v0, v1;
    float g, b;
    unsigned tp;
};
static __device__ __forceinline__ void gw_load(gw_chunk & in, const ggml_cuda_gdn_wmma_args & a, const unsigned * __restrict__ tp,
                                               int64_t t0, int vh, int qh) {
    const int t = threadIdx.x >> 4, r = (threadIdx.x & 15) * 8;
    const int64_t tt = min(t0 + t, a.T - 1);
    in.tp = tp[((t0 / GW_CK) * GW_HV + vh) * (GW_CK * GW_CK) + threadIdx.x];
    const float * kt = a.k + tt * a.sq2 + qh * a.sq1 + r;
    const float * qt = a.q + tt * a.sq2 + qh * a.sq1 + r;
    const float * vt = a.v + tt * a.sv2 + vh * a.sv1 + r;
    in.k0 = *reinterpret_cast<const gw_f4 *>(kt);
    in.k1 = *reinterpret_cast<const gw_f4 *>(kt + 4);
    in.q0 = *reinterpret_cast<const gw_f4 *>(qt);
    in.q1 = *reinterpret_cast<const gw_f4 *>(qt + 4);
    in.v0 = *reinterpret_cast<const gw_f4 *>(vt);
    in.v1 = *reinterpret_cast<const gw_f4 *>(vt + 4);
    if (threadIdx.x < GW_CK) {
        const int64_t tl = min(t0 + (int64_t) threadIdx.x, a.T - 1);
        in.g = a.g   [tl * a.sb2 + vh * a.sb1];
        in.b = a.beta[tl * a.sb2 + vh * a.sb1];
    }
}
// brings a later chunk's k, q and v lines into the cache with a load whose result is consumed (by an empty asm) one
// chunk later, so the real loads of that chunk hit the cache
static __device__ __forceinline__ unsigned gw_touch(const ggml_cuda_gdn_wmma_args & a, int64_t t0, int vh, int qh) {
    const int t = threadIdx.x >> 4, part = threadIdx.x & 15, kind = part >> 2;
    if (t0 + t >= a.T || kind == 3) {
        return 0u;
    }
    const float * p = kind == 0 ? a.k + (t0 + t) * a.sq2 + qh * a.sq1
                    : kind == 1 ? a.q + (t0 + t) * a.sq2 + qh * a.sq1
                                : a.v + (t0 + t) * a.sv2 + vh * a.sv1;
    return __float_as_uint(p[32 * (part & 3)]);
}

// Per chunk: stage it (k / q FP16, k also transposed, v, the gate's running sum, beta, T and P); barrier (1);
// X = K S0, QS = Q S0; Y, D, o (written out); the state update; barrier (2), before the next chunk's staging
// overwrites what the slower waves still read.
__global__ void __launch_bounds__(GW_NW * 32) gdn_wmma_scan(const ggml_cuda_gdn_wmma_args a, const unsigned * __restrict__ tp) {
    __shared__ __attribute__((aligned(16))) _Float16 sk[GW_CK * GW_KP], sq[GW_CK * GW_KP], skt[GW_S * GW_CK],
        stp[2 * GW_CK * GW_CK];
    __shared__ __attribute__((aligned(16))) float sv[GW_CK * GW_S];
    __shared__ float sg[GW_CK], sb[GW_CK];
    const _Float16 * st_ = stp, * sp_ = stp + GW_CK * GW_CK;   // T, P
    const int vh = blockIdx.x, qh = vh % GW_HK, w = threadIdx.x >> 5, l = threadIdx.x & 31, c = l & 15, hi = l >> 4;
    const int col = 16 * w + c;                                       // this lane's column in the products
    const int st = threadIdx.x >> 4, sr = (threadIdx.x & 15) * 8;    // this thread's staging slice
    const int64_t T = a.T;
    const gw_sel_t ps = gw_sel(hi);
    // our state is transposed: M[col][i] = S[i][col], the column contiguous
    const float * s_in = a.state + (a.state_ids ? (int64_t) a.state_ids[0] * a.state_row_stride : 0) + (int64_t) vh * GW_S * GW_S + col * GW_S;
    gw_f8 s[GW_S / 16];   // the state: rows 16j + 2v + hi, column col
#pragma unroll
    for (int j = 0; j < GW_S / 16; ++j) {
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            s[j][v] = s_in[16 * j + 2 * v + hi];
        }
    }
    const float out_scale = rsqrtf((float) GW_S);
    gw_chunk cur;
    gw_load(cur, a, tp, 0, vh, qh);
    unsigned touched = 0;
    for (int64_t t0 = 0; t0 < T; t0 += GW_CK) {
        {   // stage this chunk
            gw_h8 kh, qq;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                kh[i] = (_Float16) (i < 4 ? cur.k0[i] : cur.k1[i - 4]);
                qq[i] = (_Float16) (i < 4 ? cur.q0[i] : cur.q1[i - 4]);
                skt[(sr + i) * GW_CK + st] = kh[i];
            }
            *reinterpret_cast<uint4 *>(&sk[st * GW_KP + sr]) = __builtin_bit_cast(uint4, kh);
            *reinterpret_cast<uint4 *>(&sq[st * GW_KP + sr]) = __builtin_bit_cast(uint4, qq);
            *reinterpret_cast<gw_f4 *>(&sv[st * GW_S + sr])     = cur.v0;
            *reinterpret_cast<gw_f4 *>(&sv[st * GW_S + sr + 4]) = cur.v1;
            reinterpret_cast<unsigned *>(stp)[threadIdx.x] = cur.tp;
            if (threadIdx.x < GW_CK) {   // past the end: gate 0, beta 0
                const bool ok = t0 + l < T;
                float g = ok ? cur.g : 0.0f;
#pragma unroll
                for (int o = 1; o < GW_CK; o <<= 1) {
                    const float u = __shfl_up(g, o, GW_CK);
                    if (l >= o) {
                        g += u;
                    }
                }
                sg[l] = g;
                sb[l] = ok ? cur.b : 0.0f;
            }
        }
        if (t0 + GW_CK < T) {
            gw_load(cur, a, tp, t0 + GW_CK, vh, qh);   // lands during this chunk
        }
        const unsigned touch_now = gw_touch(a, t0 + GW_AHEAD * GW_CK, vh, qh);
        gw_lds_barrier();   // (1) this chunk staged
        __asm__ volatile("" ::"v"(touched));   // the previous touch, consumed here
        touched = touch_now;
        gw_f8 x = {}, qs = {};   // K S0 and Q S0, this wave's columns
#pragma unroll
        for (int j = 0; j < GW_S / 16; ++j) {
            const gw_h16 b = gw_acc_to_b(s[j], ps);
            x  = gw_mma(gw_ld(sk + c * GW_KP + 16 * j), b, x);
            qs = gw_mma(gw_ld(sq + c * GW_KP + 16 * j), b, qs);
        }
        gw_f8 y;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int t = 2 * v + hi;
            y[v] = sb[t] * (sv[t * GW_S + col] - __expf(sg[t]) * x[v]);
        }
        const gw_f8 d = gw_mma(gw_ld(st_ + c * GW_CK), gw_acc_to_b(y, ps), gw_f8{});
        gw_f8 o;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            o[v] = __expf(sg[2 * v + hi]) * qs[v];
        }
        o = gw_mma(gw_ld(sp_ + c * GW_CK), gw_acc_to_b(d, ps), o);
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int64_t t = t0 + 2 * v + hi;
            if (t < T) {
                a.dst[(t * GW_HV + vh) * GW_S + col] = o[v] * out_scale;
            }
        }
        const float gl = sg[GW_CK - 1];
        gw_f8 dd;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            dd[v] = __expf(gl - sg[2 * v + hi]) * d[v];
        }
        const gw_h16 db = gw_acc_to_b(dd, ps);
        const float eg = __expf(gl);
#pragma unroll
        for (int j = 0; j < GW_S / 16; ++j) {
            s[j] = gw_mma(gw_ld(skt + (16 * j + c) * GW_CK), db, s[j] * eg);
        }
        gw_lds_barrier();   // (2)
    }
    __asm__ volatile("" ::"v"(touched));
    float * s_out = a.state_out + (int64_t) vh * GW_S * GW_S + col * GW_S;
#pragma unroll
    for (int j = 0; j < GW_S / 16; ++j) {
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            s_out[16 * j + 2 * v + hi] = s[j][v];
        }
    }
}
#endif // GGML_USE_HIP

} // namespace

size_t ggml_cuda_gdn_wmma_scratch_size(int64_t T) {
    return (size_t) ((T + GW_CK - 1) / GW_CK) * GW_HV * (2 * GW_CK * GW_CK) * sizeof(uint16_t);
}

bool ggml_cuda_gdn_wmma_usable(const ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    // GGML_CUDA_GDN_WMMA_MIN=N: from N tokens (0: never)
    static const int64_t min_tokens = getenv("GGML_CUDA_GDN_WMMA_MIN") ? atoll(getenv("GGML_CUDA_GDN_WMMA_MIN")) : 128;
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * g = dst->src[3], * beta = dst->src[4];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool aligned16 = [&] {
        for (const ggml_tensor * t : { q, k, v }) {
            if ((uintptr_t) t->data % 16 != 0 || t->nb[0] != sizeof(float) || t->nb[1] % 16 != 0 || t->nb[2] % 16 != 0) {
                return false;
            }
        }
        return true;
    }();
    return GGML_CUDA_CC_IS_RDNA3(cc) && min_tokens > 0 && v->ne[2] >= min_tokens &&
           v->ne[0] == GW_S && q->ne[0] == GW_S && v->ne[1] == GW_HV && q->ne[1] == GW_HK && k->ne[1] == GW_HK &&
           v->ne[3] == 1 && q->ne[3] == 1 && g->ne[0] == 1 &&               // one sequence, a scalar gate (not KDA)
           ggml_get_op_params_i32(dst, 0) <= 1 && dst->src[6] == nullptr &&  // no snapshots, no replay
           ggml_are_same_stride(q, k) && ggml_is_contiguous(g) && ggml_is_contiguous(beta) && aligned16;
#else
    GGML_UNUSED(dst);
    return false;
#endif // GGML_USE_HIP
}

void ggml_cuda_gdn_wmma_launch(const ggml_cuda_gdn_wmma_args & a, cudaStream_t stream) {
#ifdef GGML_USE_HIP
    const int64_t nch = (a.T + GW_CK - 1) / GW_CK;
    gdn_wmma_prep<<<dim3(GW_HK, (unsigned) nch), 32 * GW_VPK, 0, stream>>>(a, (_Float16 *) a.scratch);
    gdn_wmma_scan<<<GW_HV, GW_NW * 32, 0, stream>>>(a, (const unsigned *) a.scratch);
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(a, stream);
    GGML_ABORT("the chunked GDN runs on RDNA3 only");
#endif // GGML_USE_HIP
}
