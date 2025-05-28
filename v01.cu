#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

namespace py = pybind11;

template <unsigned int BLOCKSIZE>
__global__ void gemm_bf16(int M, int N, int K, float alpha,
                          const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                          float beta, float *C) {
    const int Row = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    const int Col = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    
    if (Row < M && Col < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            float a = __bfloat162float(A[Row * K + i]);
            float b = __bfloat162float(B[i * N + Col]);
            tmp += a * b;
        }
        C[Row * N + Col] = alpha * tmp + beta * C[Row * N + Col];
    }
}

void neoGemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    
    auto A_ptr = reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>());
    auto B_ptr = reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>());
    float *C_ptr = C.data_ptr<float>();
    
    dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 block(32 * 32);
    
    gemm_bf16<32><<<grid, block>>>(M, N, K, 1.0f, A_ptr, B_ptr, 0.0f, C_ptr);
}

PYBIND11_MODULE(neogemm, m) {
    m.def("neoGemm", &neoGemm, "GEMM",
          py::arg("A"), py::arg("B"), py::arg("C"));
}
