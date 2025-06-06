#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
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
static constexpr int maxCollums = 512;
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
    uint32_t tmem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(tmem_base_addr));
    asm volatile (
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
        :
        : "r"(tmem_ptr), "r"(maxCollums));
}

__device__ static inline void dealloc_tmem(uint32_t tmem_base_addr) {
    asm volatile(
        "{\n\t"
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32  %0, %1; \n\t"
      "}"
      :
      : "r"(tmem_base_addr), "r"(maxCollums));
}

__device__ static inline void release_lock() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::);
}


__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { 
    return (((x) & 0x3FFFF) >> 0x4); 
}



__device__ void cta_commit(uint64_t &mma_barrier_addr) {
    uint32_t mma_barrier_ptr =  static_cast<uint32_t>(__cvta_generic_to_shared(&mma_barrier_addr));
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n" 
        :: "r"(mma_barrier_ptr) : "memory"
    );
}

__device__ static inline void barrier_init(uint64_t &mma_barrier_addr) {
    uint32_t mma_barrier_ptr =  static_cast<uint32_t>(__cvta_generic_to_shared(&mma_barrier_addr));
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :
        : "r"(mma_barrier_ptr), "n"(128)
    );
}

__device__ static inline void barrier_arrive(uint64_t &mma_barrier_addr) {
    uint32_t mma_barrier_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&mma_barrier_addr));
    asm volatile(
        "mbarrier.arrive.shared::cta.b64 _,[%0];\n"
        :
        : "r"(mma_barrier_ptr)
    );
}

__device__ static inline void load_wait() {
    asm volatile (
        "tcgen05.wait::ld.sync.aligned;\n"
    );
}

// Load accumulator from TMEM to registers using tcgen05.ld
__device__ void load_tmem_to_registers(float d[32][64], uint32_t const &tmem_base_addr) {
    asm volatile (" {\n"
                  " tcgen05.ld.sync.aligned.32x32b.x64.b32 "
                  "{%0, %1, %2, %3, "
                  " %4, %5, %6, %7, "
                  " %8, %9, %10, %11, "
                  " %12, %13, %14, %15, "
                  " %16, %17, %18, %19, "
                  " %20, %21, %22, %23, "
                  " %24, %25, %26, %27, "
                  " %28, %29, %30, %31, "
                  " %32, %33, %34, %35, "
                  " %36, %37, %38, %39, "
                  " %40, %41, %42, %43, "
                  " %44, %45, %46, %47, "
                  " %48, %49, %50, %51, "
                  " %52, %53, %54, %55, "
                  " %56, %57, %58, %59, "
                  " %60, %61, %62, %63}, "              
                  "[%64];\n"
                  "}\n"
                : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]),
                  "=f"(d[0][4]), "=f"(d[0][5]), "=f"(d[0][6]), "=f"(d[0][7]),
                  "=f"(d[0][8]), "=f"(d[0][9]), "=f"(d[0][10]), "=f"(d[0][11]),
                  "=f"(d[0][12]), "=f"(d[0][13]), "=f"(d[0][14]), "=f"(d[0][15]),
                  "=f"(d[0][16]), "=f"(d[0][17]), "=f"(d[0][18]), "=f"(d[0][19]),
                  "=f"(d[0][20]), "=f"(d[0][21]), "=f"(d[0][22]), "=f"(d[0][23]),
                  "=f"(d[0][24]), "=f"(d[0][25]), "=f"(d[0][26]), "=f"(d[0][27]),
                  "=f"(d[0][28]), "=f"(d[0][29]), "=f"(d[0][30]), "=f"(d[0][31]),
                  "=f"(d[0][32]), "=f"(d[0][33]), "=f"(d[0][34]), "=f"(d[0][35]),
                  "=f"(d[0][36]), "=f"(d[0][37]), "=f"(d[0][38]), "=f"(d[0][39]),
                  "=f"(d[0][40]), "=f"(d[0][41]), "=f"(d[0][42]), "=f"(d[0][43]),
                  "=f"(d[0][44]), "=f"(d[0][45]), "=f"(d[0][46]), "=f"(d[0][47]),
                  "=f"(d[0][48]), "=f"(d[0][49]), "=f"(d[0][50]), "=f"(d[0][51]),
                  "=f"(d[0][52]), "=f"(d[0][53]), "=f"(d[0][54]), "=f"(d[0][55]),
                  "=f"(d[0][56]), "=f"(d[0][57]), "=f"(d[0][58]), "=f"(d[0][59]),
                  "=f"(d[0][60]), "=f"(d[0][61]), "=f"(d[0][62]), "=f"(d[0][63])
                :  "r"(tmem_base_addr));
}


template<uint8_t Acc>
__device__ void mma64x64x16(bf16* sA, bf16* sB, uint32_t const &base_tmem_ptr) {
    uint64_t desc_a = 0x4000004000000000 | 
        (matrix_descriptor_encode(static_cast<uint64_t>(__cvta_generic_to_shared(sA))));
    uint64_t desc_b = 0x4000004000000000 | 
        (matrix_descriptor_encode(static_cast<uint64_t>(__cvta_generic_to_shared(sB))));
    
    constexpr uint32_t instruction_desc = 0x04100490;
    constexpr uint32_t mask[4] = {0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p; \n\t"
        "}\n"
        :
        : "r"(base_tmem_ptr), "l"(desc_a), "l"(desc_b), "r"(instruction_desc), "r"(Acc),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
}

template<int BM, int BN, int BK, int MMA_M, int MMA_N, int MMA_K, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) gemm_kernel(
    int M, int N, int K, 
    bf16* C,
    const CUtensorMap* tensorMapA, 
    const CUtensorMap* tensorMapB) {
    int tid = threadIdx.x;
    int warp = tid / 32;

    // Shared memory buffers - 128-byte aligned
    __shared__ alignas(128) bf16 sA[BM * BK];
    __shared__ alignas(128) bf16 sB[BK * BN];
    __shared__ alignas(16) uint32_t tmem_base_addr;
    __shared__ alignas(16) uint64_t mma_barrier_addr;


    // Allocate tensor memory
    if (warp == 0) {
        tmem_alloc_maxColumns(&tmem_base_addr);
    }
    __syncthreads();

    // Output accumulator
    float d[64][64] = {};

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

    }
    if (tid == 0) {
        barrier_init(mma_barrier_addr); // Initialize barriers

        // Perform MMA operations for different K iterations
        mma64x64x16<0>(&sA[0], &sB[0], tmem_base_addr);
        mma64x64x16<1>(&sA[MMA_K], &sB[MMA_K * BN], tmem_base_addr);
        mma64x64x16<1>(&sA[2 * MMA_K], &sB[2 * MMA_K * BN], tmem_base_addr);
        mma64x64x16<1>(&sA[3 * MMA_K], &sB[2 * MMA_K * BN], tmem_base_addr);
    }

    barrier_arrive(mma_barrier_addr);
    cta_commit(mma_barrier_addr);

    // Load accumulator from TMEM
    if(warp == 0) {
        load_tmem_to_registers(d, tmem_base_addr);
        load_tmem_to_registers(d+32, (tmem_base_addr + 0x00200000));
    }
    load_wait();
        
    bf16 *block_C = C + num_blocks_n * BN * M + num_blocks_m * BM;             
    int idx = tid % 64;
    for(int cols = 0; cols < 32; ++cols) {
        if(warp == 0 || warp == 1) {
            block_C[idx] = __float2bfloat16(d[idx][cols]);
        } else {
            block_C[idx] = __float2bfloat16(d[idx][cols + 32]);
        }
        __syncthreads();
    }
    if(warp == 0) {
        release_lock();
        dealloc_tmem(tmem_base_addr);
    }
    
}

void neoGemmV1(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0); // A -> MxK
    int K = A.size(1);
    int N = B.size(0); // B -> NxK

    // Get data pointers
    bf16* bf16_data_ptr_A = reinterpret_cast<bf16*>(A.data_ptr<at::BFloat16>());
    bf16* bf16_data_ptr_B = reinterpret_cast<bf16*>(B.data_ptr<at::BFloat16>());
    bf16* bf16_data_ptr_C = reinterpret_cast<bf16*>(C.data_ptr<at::BFloat16>());

    // Tile sizes
    constexpr int BM = 64;
    constexpr int BN = 64;
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
    gemm_kernel<BM, BN, BK, 64, 64, 16, NUM_THREADS><<<grid, block>>>(
        M, N, K, bf16_data_ptr_C, d_tma_map_A, d_tma_map_B);

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
          py::arg("A"), py::arg("B"), py::arg("C"));
}
