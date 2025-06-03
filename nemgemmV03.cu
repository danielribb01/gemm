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
#include <unistd.h>

// binding libraries
#include <torch/extension.h>
#include <pybind11/pybind11.h>


namespace cde = cuda::device::experimental;  // rename library
typedef __nv_bfloat16 bf16; // rename type
using barrier = cuda::barrier<cuda::thread_scope_block>; // set barrier as barrier for all threads in the same block


#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y)) // ceil div func

template <int BlockMajorSize, int BlockMinorSize>
void create_tensor_map(CUtensormap *tma_map, bf16* data_ptr, int blocks_height, int blocks_width) {
    void* gmem_adress = static_cast<void*>(data_ptr); // cast to void 
    uint64_t gmem_prob_shape[5] = {(uint64_t) BlockMinorSize * blocks_width, (uint64_t) BlockMajorSize * blocks_height, 1, 1, 1}; // {numCols, numRows, 1, 1, 1} -> (Major, Minor)
    uint64_t gmem_prob_stride[5] = {sizeof(bf16), sizeof(bf16) * BlockMinorSize *  blocks_width, 0, 0, 0};                        // {2B, K * 2B, 0, 0, 0} -> row major 
    uint32_t smem_box_shape[5] = {uint32_t(BlockMinorSize), uint32_t(BLockMajorSize), 1, 1, 1};                                   // {numCols, numRows, 1, 1, 1} -> (Major, Minor)      * tile for smem (bMajor, bMinor)
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};                                                                                // linear storage on shared

    CUresult result = cuTensorMapEncodeTiled(                                                                                     // creates tma map pointer 
        tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gmem_address, gmem_prob_shape,
        gmem_prob_stride + 1, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

        assert(result == CUDA_SUCCESS);     // assert success
}


template <int BlockMajorSize, int BlockMinorSize>
__host__ static inline CUtensorMap* allocate_and_create_tensor_map(bf16* data_ptr, int blocks_height, int blocks_width) {
    CUtensorMap *tma_map_d_tmp; // create device variable
    cudaMalloc(&tma_map_d_tmp, sizeof(CUtensorMap)); // alloc
    CUtensorMap tma_map_host; // host variable
    create_tensor_map<BlockMajorSize, BlockMinorSize>(&tma_map_host, data_ptr, blocks_height, blocks_width); // map pointer
    cudaMemcpy(tma_map_d_tmp, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice); // copy to device
    return tma_map_d_tmp; // return device
}


__device__ static inline void tmem_alloc_maxCollumns(uint32_t* tmem_base_addr) {
    asm volatile (
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
        :
        : "r"(tmem_base_addr), "n"(constexpr int 512))
}

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); } // get last 18 bit and then divide by 16 (take out last 4 bits)



__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); // sync warpgroup ptx instruction
}

__device__ void cta_commit() {
    asm volatile("tcgen05.commit.cta_group::1.completion_mechanism.mbarrier::arrive::one.b64;\n" ::: "memory"); // commit mma operations ptx intruction
}

template <int N>
__device__ void warpgroup_wait() {
    static_assert(N>= 0 && N <= 7, "Invalid number"); // 7 = max operations that a warp in the warp group does 
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); // wgmma wait ptx intruction
}


__device__ static inline void barrier_alloc(uint32_t* mma_barrier_addr) {
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :
        : "r"(mma_barrier_addr), "n"(constexpr uint32_t 128);
    )
}

__device__ static inline void att_barrier(uint32_t* mma_barrier_ptr) {
    asm volatile(
        "mbarrier.arrive.shared::cta.b64 [%0]"
        :
        : "r"(mma_barrier_ptr);
    )
}



template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int AccD>
__device__ void mma128x256x16(bf16* sA, bf16* sB, uint_32t* base_tmem_ptr,) {
    // descriptor will always be 0x4000004000000000 | addr 
    //                             ^ represents the 128B swizzle
    //                                   ^ represents the stride dimension
    //                                         ^^^^ will be replaced by addr 
    uint64_t desc_a = 0x4000004000000000 | (matrix_descriptor_encode(static_cast<uint32_t>(__cvta_generic_to_shared(&sA[0]))));
    uint64_t desc_b = 0x4000004000000000 | (matrix_descriptor_encode(static_cast<uint32_t>(__cvta_generic_to_shared(&sB[0]))));
    constexpr uint32_t instruction_desc = 0x084004A0; // instruction desc 
    constexpr uint32_t mask[4] = {0, 0, 0, 0}; // 0-ed to enable all input d
    asm volatile( 
        "{\n" // ptx instruction call
            "tcgen05.mma.cta_group::1.kind::f16"
            "[%0]," // d-tmem
            " %1, " // a-desc
            " %2, " // b-dec
            " %3, " // idesc
            " {%4, %5, %6, %7} " // disable-output-lane
            " %8, "  // enable-input-d
            "}\n" 
            :
            : "r"(base_tmem_ptr), "l"(desc_a), "l"(desc_b), "n"(instruction_desc),
              "n"(mask[0]), "n"(mask[1]), "n"(mask[2]), "n"(mask[3]), "n"(AccD));   // inputs -> l = long | n = constant values

}


template<int BM, int BN, int BK, int MMA_M, int MMA_N, int MMA_K, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) gemm_kernel(int M, int N, int K, bf16* C, const CUtensorMap* tensorMapA, const CUtensorMap* tensorMapB) {

    int tid = threadIdx.x; // [0,127] -> thread ID on block
    int warp = tid / 32;   // [0,3] -> warp ID on warp group

    // create smem tile buffers *128bytes aligned 
    __shared__ alignas(128) bf16 sA[BM*BK];
    __shared__ alignas(128) bf16 sB[BK*BN];
    __shared__ alignas(16) uint32_t tmem_base_addr; //  addr to store tmem pos
    __shared__ alignas(16) uint32_t mma_barrier_addr; // addr to mma barrier
    
    if(tid == 0 && warp == 0) {
        barrier_alloc(&mma_barrier_addr);
    }

    // output matrix -> size = 32 elem -> each thread will process 32 elems 
    float d[MMA_N/16][8]; 
    
    if(warp == 0) {
        tmem_alloc_maxCollumns(&tmem_base_addr);
    }
    __syncthreads();

    

    static_assert(sizeof(d) * 128 == BM * BN * sizeof(float)); // assert sizes are equal
    //memset(d, 0, sizeof(d)); // set d to 0 

    // num blocks
    const int num_blocks_k = K / BK;
    int num_blocks_n = blockIdx.x % (N / BN);
    int num_blocks_m = blockIdx.x / (N / BN);


    #pragma nv_diag_supress static_var_with_dynamic_init // remove warning for dynamic init
    // create smem barrier buffers
    __shared__ barrier barA;
    __shared__ barrier barB;

    // only one thread per block iniciate the barrier 
    if(threadIdx.x == 0) {
        init(&barA, blockDim.x); // init barrier A
        init(&barB, blockDim.x); // init barrier B
        cde::fence_proxy_async_shared_cta(); // sync smem threads on the same block  
    }

    __syncthreads();

    barrier::arrival_token tokenA, tokenB; // get sync tokens for A and B
    for(int bkIdx = 0; bkIdx < num_blocks_k; ++bkIdx) {

        if (threadIdx.x == 0) { // only one thread per block sets the instruction
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[0], tensorMapA, bkIdx * BK, num_blocks_m * BM, barA); //  (M,K) tile A tma copy from gmem to smem
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sA)); // att the barriers 
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[0], tensorMapB, bkIdx * BK, num_blocks_n * BN, barB); //  (N,K) tile B tma copy from gmem to smem
            tokenA = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sB)); // att the barriers 
        } else {
            tokenA = barA.arrive(); // make all threads arrive at the barriers
            tokenB = barB.arrive(); // make all threads arrive at the barriers
        }

        barA.wait(std::move(tokenA)); // wait for all threads to complete the copy and arrive at the barrier
        barB.wait(std::move(tokenB)); // wait for all threads to complete the copy and arrive at the barrier
        __syncthreads();


        warpgroup_arrive(); // asserts all warps on the warp group arrives
        // mma calls that iterate over all MMA_M | MMA_N tiles 
        if(warp == 0 && tid == 0) {
            mma128x256x16<1, 1, 1, 0, 0, 0>(d, &sA[0], &sB[0], tmem_base_addr); 
            mma128x256x16<1, 1, 1, 0, 0, 1>(d, &sA[MMA_K], &sB[MMA_K], tmem_base_addr);
            mma128x256x16<1, 1, 1, 0, 0, 1>(d, &sA[2*MMA_K], &sB[2*MMA_K], tmem_base_addr);
            mma128x256x16<1, 1, 1, 0, 0, 1>(d, &sA[3*MMA_K], &sB[3*MMA_K], tmem_base_addr);
            
        }
        att_barrier(mma_barrier_addr);

        cta_commit(mma_barrier_addr); // make sure the cta finishes the mma operations
        
    }

    {
       // tcgen05.ld (tmem -> rmem)

        warpgroup_wait<0>();     // wait untils all operations on the warpgroup are done 
        int lane = tid % 32;   // [0,31] -> thread ID on warp 
        uint32_t row = warp*16 + lane / 4; //   gets rows [0,7]
        bf16 *block_C = C + num_block_n*BN*M + num_block_m*BM;

        for (int m_it = 0; m_it < BM/MMA_M; ++m_it) {                // 64 / 64 = 1 it
            for (int n_it = 0; n_it < BN/MMA_N; ++n_it) {            // 64 /64 = 1 it
                for (int w = 0; w < MMA_N/16; ++w) {                 // 64 / 16 = 4 it
                    int col = 16*w + 2*(tid % 4);                      //  cols [0, 7] % 2 = 0
                    #define IDX(i, j) ((j + n_it*MMA_N)*M + ((i) + m_it*MMA_M)) // gets true id

                    block_C[IDX(row, col)] = d[w][0];        // rows [0,7] | cols [0,7] % 2 = 0 
                    block_C[IDX(row, col+1)] = d[w][1];      // rows [0,7] | cols [0,7] % 2 = 1 
                    block_C[IDX(row+8, col)] = d[w][2];      // rows [8,15] | cols [0,7] % 2 = 0
                    block_C[IDX(row+8, col+1)] = d[w][3];    // rows [8,15] | cols [0,7] % 2 = 1
    
                    block_C[IDX(row, col+8)] = d[w][4];      // rows [0,7] | cols [8,15] % 2 = 0
                    block_C[IDX(row, col+9)] = d[w][5];      // rows [0,7] | cols [8,15] % 2 = 1
                    block_C[IDX(row+8, col+8)] = d[w][6];    // rows [8,15] | cols [8,15] % 2 = 0
                    block_C[IDX(row+8, col+9)]s = d[w][7];    // rows [8,15] | cols [8,15] % 2 = 1

                    #undef IDX
                }
            }
        }
    }
}

void neoGemmV1(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0) // A -> MxK
    int K = A.size(1) 
    int N = B.size(0); // B -> NxK

    // get data pointers
    bf16* bf16_data_ptr_A = reinterpret_cast<bf16*>(A.data_ptr<at::BFloat16>());
    bf16* bf16_data_ptr_B = reinterpret_cast<bf16*>(B.data_ptr<at::BFloat16>());
    bf16* bf16_data_ptr_C = reinterpret_cast<bf16*>(C.data_ptr<at::BFloat16>());

    // tile
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 64;

    // create tma atom variables
    CUtensorMap *d_tma_map_A = 0;
    CUtensorMap *d_tma_map_B = 0;

    int prev_m, prev_n, _prev_k;
    if (!d_tma_map_A) {
        // allocate tma atoms
        d_tma_map_A = allocate_and_create_tensor_map<BM, BK>(bf16_data_ptr_A, M / BM, K / BK);
        d_tma_map_B = allocate_and_create_tensor_map<BN, BK>(bf16_data_ptr_A, N / BN, K / BK);

        prev_m = M;
        prev_n = N;
        prev_k = K;
    }

    // assert that the tma dimensions are corrert
    assert(M == prev_m && N = prev_n && K == prev_K);

    // blocksize  
    constexpr int NUM_THREADS = 128;

    // kernel launch
    gemm< /* block tiles */ 
          BM, BN, BK, 
          /* warp tiles*/
          128, 256, 16,
          /*threads*/ 
          NUM_THREADS><<<(M/BM) * (N/BN), NUM_THREADS>>>(M, N, K, bf16_data_ptr_C, d_tma_map_A, d_tma_map_B);
}

namespace py = pybind11; // rename pybind

// pybind
PYBIND11_MODULE(neoGEMM, m) {
    m.def("neoGemmV1", &neoGemmV1, "GEMM",
          py::arg("A"), py::arg("B"), py::arg("C"));
}
