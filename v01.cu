#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

namespace py = pybind11;

template <const int BLOCKSIZE>
__global__ void neogemm_shared_mem_block(int M, int N, int K, float alpha, const __nv_bfloat16 *A, const __nv_bfloat16 *B, float beta, float *C)
{
    // output block
    const uint outRow = blockIdx.x;
    const uint outCol = blockIdx.y;

    // smem buffers
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // row and col processed
    const uint thCol = threadIdx.x % BLOCKSIZE;
    const uint thRow = threadIdx.x / BLOCKSIZE;

    // advance matrices pointers to correct data block
    A += outRow * BLOCKSIZE * K;
    B += outCol * BLOCKSIZE;
    C += outRow * BLOCKSIZE * N + outCol * BLOCKSIZE;

    float tmp = 0.0; // acc for results

    // iterate over common dim
    for ( int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // load tile to smem
        As[thRow * BLOCKSIZE + thCol] = __bfloat162float(A [thRow * K + thCol]);
        Bs[thRow * BLOCKSIZE + thCol] = __bfloat162float(B [thRow * N + thCol]);




        __syncthreads();
        // advance pointers
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
        tmp += As[thRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + thCol];
        }
        __syncthreads();
    }

    C[thRow * N + thCol] = alpha * tmp + beta * C[thRow * N + thCol];
    
}

void neoGemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    
    auto A_ptr = reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>());
    auto B_ptr = reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>());
    float *C_ptr = C.data_ptr<float>();
    
    dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 block(32 * 32);
    
    neogemm_shared_mem_block<32><<<grid, block>>>(M, N, K, 1.0f, A_ptr, B_ptr, 0.0f, C_ptr);
}

PYBIND11_MODULE(neogemmBeta, m) {
    m.def("neoGemm", &neoGemm, "GEMM",
          py::arg("A"), py::arg("B"), py::arg("C"));
}
