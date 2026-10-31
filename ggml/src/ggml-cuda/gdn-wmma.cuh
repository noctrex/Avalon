#pragma once

#include "common.cuh"

// The prompt's gated delta rule in chunks of 16 tokens on the RDNA3 matrix cores (the WY form; Strata's gfx11
// kernels, src/prefill/kernels.cu, MIT, Niko1221 and the Strata contributors). FP16 operands, FP32 accumulation.
struct ggml_cuda_gdn_wmma_args {
    const float * q;      // [S, H_k] rows of token t at q + t*sq2 + head*sq1 (floats)
    const float * k;
    const float * v;      // [S, H] rows at v + t*sv2 + head*sv1
    const float * g;      // log decay, g[t*sb2 + head*sb1]
    const float * beta;
    const float * state;  // [H][col][i] (the transposed state), the sequence's row at state + state_off
    float *       dst;    // [S, H, T] output, scaled by 1/sqrt(S)
    float *       state_out;
    void *        scratch;   // ggml_cuda_gdn_wmma_scratch_size(T) bytes
    int64_t T;
    int64_t sq1, sq2, sv1, sv2, sb1, sb2;
    const int32_t * state_ids;   // the state read in the cache: row state_ids[0] of state_row_stride floats
    int64_t         state_row_stride;
};

// the scratch the chunked kernels need for a prompt batch of T tokens (T and P of every chunk and value head)
size_t ggml_cuda_gdn_wmma_scratch_size(int64_t T);
// whether the chunked kernels take this node (RDNA3, the Qwen3.8 shape, a long batch of one sequence)
bool   ggml_cuda_gdn_wmma_usable(const ggml_tensor * dst);
void   ggml_cuda_gdn_wmma_launch(const ggml_cuda_gdn_wmma_args & a, cudaStream_t stream);
