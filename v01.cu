#include <cuda_runtime.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

namespace py = pybind11;

template <const uint BLOCKSIZE>
__global__ void gemm(int M, int N, int K, float alpha,
                     const float *A, const float *B,
                     float beta, float *C) {
  const int Row = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
  const int Col = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);


  if (Row < M && Col < K) {
    float tmp = 0.0;
    for (int i = 0; i < N; ++i) {
      tmp += A[Row * N + i] * B[i * K + Col];
    }
    C[Row * K + Col] = alpha * tmp + beta * C[Row * K + Col];
  }
}

void neoGemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    const float *A_ptr = A.data_ptr<float>();
    const float *B_ptr = B.data_ptr<float>();
    float *C_ptr = C.data_ptr<float>();

    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32 * 32);

    float alpha = 1;
    float beta = 0;

    gemm<32><<<gridDim, blockDim>>>(M, N, K, alpha, A_ptr, B_ptr, beta, C_ptr);
}
PYBIND11_MODULE(neogemm, m) {
    m.doc() = "Matrix multiplication with CUDA";
    m.def("neoGemm", &neoGemm, "GEMM function", py::arg("A"), py::arg("B"), py::arg("C"));
}
