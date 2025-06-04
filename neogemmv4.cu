#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <ctime>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_bf16.h>
#include <cassert>

// binding libraries
#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace cde = cuda::device::experimental;
typedef __nv_bfloat16 bf16;
using barrier = cuda::barrier<cuda::thread_scope_block>;

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

// Global variables for TMA maps to avoid reallocation
static CUtensorMap *d_tma_map_A = nullptr;
static CUtensorMap *d_tma_map_B = nullptr;
static int prev_m = 0, prev_n = 0, prev_k = 0;

template <int BlockMajorSize, int BlockMinorSize>
void create_tensor_map(CUtensorMap *tma_map, bf16* data_ptr, int blocks_height, int blocks_width) {
    void* gmem_address = static_cast<void*>(data_ptr);
    uint64_t gmem_prob_shape[5] = {
        static_cast<uint64_t>(BlockMinorSize * blocks_width), 
        static_cast<uint64_t>(BlockMajorSize * blocks_height), 
        1, 1, 1
    };
    uint64_t gmem_prob_stride[5] = {
        sizeof(bf16), 
        sizeof(bf16) * BlockMinorSize * blocks_width, 
        0, 0, 0
    };
    uint32_t smem_box_shape[5] = {
        static_cast<uint32_t>(BlockMinorSize), 
        static_cast<uint32_t>(BlockMajorSize), 
        1, 1, 1
    };
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gmem_address, gmem_prob_shape,
        gmem_prob_stride + 1, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);
}

template <int BlockMajorSize, int BlockMinorSize>
__host__ static inline CUtensorMap* allocate_and_create_tensor_map(bf16* data_ptr, int blocks_height, int blocks_width) {
    CUtensorMap *tma_map_d_tmp;
    cudaMalloc(&tma_map_d_tmp, sizeof(CUtensorMap));
    CUtensorMap tma_map_host;
    create_tensor_map<BlockMajorSize, BlockMinorSize>(&tma_map_host, data_ptr, blocks_height, blocks_width);
    cudaMemcpy(tma_map_d_tmp, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d_tmp;
}

__device__ static inline void tmem_alloc_maxColumns(uint32_t* tmem_base_addr) {
    asm volatile (
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
        :
        : "l"(tmem_base_addr), "n"(512)
    );
}

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { 
    return (((x) & 0x3FFFF) >> 0x4); 
}



__device__ void cta_commit(uint32_t* mma_barrier_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n" 
        :: "l"(mma_barrier_addr) : "memory"
    );
}

__device__ static inline void barrier_init(uint32_t* mma_barrier_addr) {
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :
        : "l"(mma_barrier_addr), "n"(128)
    );
}

__device__ static inline void barrier_arrive(uint32_t *mma_barrier_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mma_barrier_ptr));
    asm volatile(
        "mbarrier.arrive.shared::cta.b64 _,[%0];\n"
        :
        : "r"(addr)
    );
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int AccD>
__device__ void mma128x256x16(float* d, bf16* sA, bf16* sB, uint32_t const &base_tmem_ptr) {
    uint64_t desc_a = 0x4000004000000000 | 
        (matrix_descriptor_encode(static_cast<uint64_t>(__cvta_generic_to_shared(sA))));
    uint64_t desc_b = 0x4000004000000000 | 
        (matrix_descriptor_encode(static_cast<uint64_t>(__cvta_generic_to_shared(sB))));
    
    constexpr uint32_t instruction_desc = 0x084004A0;
    constexpr uint32_t mask[4] = {0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p; \n\t"
        "}\n"
        :
        : "r"(base_tmem_ptr), "l"(desc_a), "l"(desc_b), "r"(instruction_desc), "r"(AccD),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
}

// Load accumulator from TMEM to registers using tcgen05.ld
__device__ void load_tmem_to_registers(float* d, uint32_t const &tmem_base_addr, int offset) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x1.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "[%8+%9];\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]),
          "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
        : "r"(tmem_base_addr), "r"(offset)
    );
}

template<int BM, int BN, int BK, int MMA_M, int MMA_N, int MMA_K, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) gemm_kernel(
    int M, int N, int K, 
    bf16* C, bf16* D,
    const CUtensorMap* tensorMapA, 
    const CUtensorMap* tensorMapB,
    float alpha, float beta
) {
    int tid = threadIdx.x;
    int warp = tid / 32;

    // Shared memory buffers - 128-byte aligned
    __shared__ alignas(128) bf16 sA[BM * BK];
    __shared__ alignas(128) bf16 sB[BK * BN];
    __shared__ alignas(16) uint32_t tmem_base_addr;
    __shared__ alignas(16) uint32_t mma_barrier_addr;


    // Initialize barriers
    if (tid == 0) {
        barrier_init(&mma_barrier_addr);
    }

    // Allocate tensor memory
    if (warp == 0 && tid == 0) {
        tmem_alloc_maxColumns(&tmem_base_addr);
    }
    __syncthreads();

    // Output accumulator
    float d[MMA_N/16][8] = {};

    // Block indices
    const int num_blocks_k = K / BK;
    int num_blocks_n = blockIdx.x % (N / BN);
    int num_blocks_m = blockIdx.x / (N / BN);

    // TMA barriers
    __shared__ barrier barA;
    __shared__ barrier barB;

    if (threadIdx.x == 0) {
        init(&barA, blockDim.x);
        init(&barB, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    // Main computation loop
    barrier::arrival_token tokenA, tokenB;
    for (int bkIdx = 0; bkIdx < num_blocks_k; ++bkIdx) {
        // TMA loads
        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &sA[0], tensorMapA, bkIdx * BK, num_blocks_m * BM, barA);
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sA));
            
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &sB[0], tensorMapB, bkIdx * BK, num_blocks_n * BN, barB);
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sB));
        } else {
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }

        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        __syncthreads();

        
        
        if (tid == 0) {
            // Perform MMA operations for different K iterations
            mma128x256x16<1, 1, 1, 0, 0, 0>(d[0], &sA[0], &sB[0], tmem_base_addr);
            mma128x256x16<1, 1, 1, 0, 0, 1>(d[0], &sA[MMA_K], &sB[MMA_K * BN], tmem_base_addr);
            mma128x256x16<1, 1, 1, 0, 0, 1>(d[0], &sA[2*MMA_K], &sB[2*MMA_K * BN], tmem_base_addr);
            mma128x256x16<1, 1, 1, 0, 0, 1>(d[0], &sA[3*MMA_K], &sB[3*MMA_K * BN], tmem_base_addr);
        }
        
        barrier_arrive(&mma_barrier_addr);
        cta_commit(&mma_barrier_addr);
    }
    
    int lane = tid % 32;
    uint32_t row = warp * 16 + lane / 4;
    bf16 *block_C = C + num_blocks_n * BN * M + num_blocks_m * BM;
    bf16 *block_D = D + num_blocks_n * BN * M + num_blocks_m * BM;

    // Load accumulator from TMEM
    load_tmem_to_registers(d[0], tmem_base_addr, 0);

    for (int m_it = 0; m_it < BM/MMA_M; ++m_it) {
        for (int n_it = 0; n_it < BN/MMA_N; ++n_it) {
            for (int w = 0; w < MMA_N/16; ++w) {
                int col = 16*w + 2*(tid % 4);
                
                #define IDX(i, j) ((j + n_it*MMA_N)*M + ((i) + m_it*MMA_M))
                
                // Read C matrix values
                float c_vals[8];
                c_vals[0] = __bfloat162float(block_C[IDX(row, col)]);
                c_vals[1] = __bfloat162float(block_C[IDX(row, col+1)]);
                c_vals[2] = __bfloat162float(block_C[IDX(row+8, col)]);
                c_vals[3] = __bfloat162float(block_C[IDX(row+8, col+1)]);
                c_vals[4] = __bfloat162float(block_C[IDX(row, col+8)]);
                c_vals[5] = __bfloat162float(block_C[IDX(row, col+9)]);
                c_vals[6] = __bfloat162float(block_C[IDX(row+8, col+8)]);
                c_vals[7] = __bfloat162float(block_C[IDX(row+8, col+9)]);
                
                // Apply alpha and beta scaling: D = alpha * A * B + beta * C
                block_D[IDX(row, col)] = __float2bfloat16(alpha * d[w][0] + beta * c_vals[0]);
                block_D[IDX(row, col+1)] = __float2bfloat16(alpha * d[w][1] + beta * c_vals[1]);
                block_D[IDX(row+8, col)] = __float2bfloat16(alpha * d[w][2] + beta * c_vals[2]);
                block_D[IDX(row+8, col+1)] = __float2bfloat16(alpha * d[w][3] + beta * c_vals[3]);
                block_D[IDX(row, col+8)] = __float2bfloat16(alpha * d[w][4] + beta * c_vals[4]);
                block_D[IDX(row, col+9)] = __float2bfloat16(alpha * d[w][5] + beta * c_vals[5]);
                block_D[IDX(row+8, col+8)] = __float2bfloat16(alpha * d[w][6] + beta * c_vals[6]);
                block_D[IDX(row+8, col+9)] = __float2bfloat16(alpha * d[w][7] + beta * c_vals[7]);
                
                #undef IDX
            }
        }
    }
}

void neoGemmV1(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor D, 
               float alpha = 1.0f, float beta = 0.0f) {
    int M = A.size(0); // A -> MxK
    int K = A.size(1);
    int N = B.size(0); // B -> NxK

    // Get data pointers
    bf16* bf16_data_ptr_A = reinterpret_cast<bf16*>(A.data_ptr<at::BFloat16>());
    bf16* bf16_data_ptr_B = reinterpret_cast<bf16*>(B.data_ptr<at::BFloat16>());
    bf16* bf16_data_ptr_C = reinterpret_cast<bf16*>(C.data_ptr<at::BFloat16>());
    bf16* bf16_data_ptr_D = reinterpret_cast<bf16*>(D.data_ptr<at::BFloat16>());

    // Tile sizes
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 64;

    // Check if we need to reallocate TMA maps
    if (!d_tma_map_A || M != prev_m || N != prev_n || K != prev_k) {
        // Free previous maps if they exist
        if (d_tma_map_A) cudaFree(d_tma_map_A);
        if (d_tma_map_B) cudaFree(d_tma_map_B);
        
        // Allocate new TMA maps
        d_tma_map_A = allocate_and_create_tensor_map<BM, BK>(bf16_data_ptr_A, M / BM, K / BK);
        d_tma_map_B = allocate_and_create_tensor_map<BN, BK>(bf16_data_ptr_B, N / BN, K / BK);
        
        prev_m = M;
        prev_n = N;
        prev_k = K;
    }

    // Assert dimensions are correct
    assert(M == prev_m && N == prev_n && K == prev_k);
    assert(M % BM == 0 && N % BN == 0 && K % BK == 0);

    // Launch configuration
    constexpr int NUM_THREADS = 128;
    dim3 grid((M/BM) * (N/BN));
    dim3 block(NUM_THREADS);

    // Launch kernel
    gemm_kernel<BM, BN, BK, 128, 256, 16, NUM_THREADS><<<grid, block>>>(
        M, N, K, bf16_data_ptr_C, bf16_data_ptr_D, d_tma_map_A, d_tma_map_B, alpha, beta
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
}

namespace py = pybind11;

// Python binding
PYBIND11_MODULE(neoGEMM, m) {
    m.def("neoGemmV1", &neoGemmV1, "Optimized GEMM with TMA and tcgen05",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("D"),
          py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f);
}
