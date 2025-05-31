/*
 * Nome do Arquivo: neogemm.cu
 *
 * Autores: Gabriel Ambrosio Valentin E Daniel Augusto Ribeiro
 *          
 * 
 *
 * Versão: 1.0
 * 
 * Descrição: Esta lib implementa a NeoGEMM (Neospace General Matrix Multiply), em CUDA.
 * 
 * Copyright (c) NEOSPACE A.I. TECHNOLOGIES LTDA.
 */



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
#include <torch/extension.h>
#include <pybind11/pybind11.h>


#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

typedef __nv_bfloat16 bf16;

namespace py = pybind11;

using barrier = cuda::barrier<cuda::thread_scope_block>;
// cuda::barrier - primitiva de sincronização 
// <cuda::thread_scope_block> - escopo de onde ocorre a sincronização(no bloco de threads)
// Pode ser feita em toda gpu, warp(32 threads), e no sistema 


namespace cde = cuda::device::experimental;

// 0x3FFFF extrai os 18 bits menos significativos 
// >>0x4 = divide por 16
// isso permite codificar os valores em granulidade de 16 bytes(WGMMA trabalha com granulidade de 16bytes)

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {return (((x) & 0x3FFFF) >> 0x4);}


// descritor da shared memory é um valor de 64 bits(contido em um registrador)
// 13-0  indica o endereço do ponteiro - tem tamanho de 14 bits
// 29-16 indica o leading dimension(indica se é row-wise ou col-wise) - tem tamanho de 14 bits
// 45-32 indica o stride - tem tamanho de 14 bits 
// 51-49 indica o offset base da matriz - tem tamanho 3 de bits
// 63-62 indica o modo swizzle usado - tem tamanho de 2 bits


// Modos do Swizzle
// 0 - não usa nenhum
// 1 - swizzle de 128 - Bytes
// 2 - swizzle de 64 - Bytes
// 3 - swizlle de 32 - Bytes



// wgmma.fence - ele força uma ordenação de acesso nos registrados entre as operações wgmma.mma_async
// Apenas os registradores do acumulador e o registrador contendo os fragmentos da matriz A realiza a operação

__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
// o warpgroup_commit e o warpgroup_wait são usados para esperar a operação WGMMA Async ficar completa

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}


// cada warpgroup realiza 8 operações para a dimensão do bm64nNb16

template <int N>
__device__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}


// Para usarmos o TMA o hardware requer um tensor map
// o tensor map ele descreve o layout multi-dimensional do array na GMEM e na SMEM
// o tensor map é criado usando o cuTensorMapEncode API
// ele é transferido do host para o device como um parâmetro const kernel(__grid_constant__)


// Criar um tensor map requer muitos parâmetros
// Ponteiro base para um array na global memory, stride de uma linha para outra em bytes
// o tamanho do buffer na SMEM em número de elementos e etc...


// Estrutura recomendada no CUDA GUIDE
/*
struct tensormap_params {
  void* global_address;
  int rank;
  uint32_t box_dim[5];
  uint64_t global_dim[5];
  size_t global_stride[4];
  uint32_t element_stride[5];
};
*/

template <int BlockMajorSize, int BlockMinorSize>
void create_tensor_map(CUtensorMap *tma_map, bf16* gmem_ptr, int blocks_height, int blocks_width)
{
    void* gmem_address = (void *)gmem_ptr;
    uint64_t gmem_prob_shape[5] = {(uint64_t)BlockMinorSize * blocks_width, (uint64_t)BlockMajorSize*blocks_height,1,1,1}; // define as dimensões na memmória global
    // 1,1,1(como o vetor precisa de 5 posições, preenchemos o restante com 1, pois não afeta)
    uint64_t gmem_prob_stride[5] = {sizeof(bf16), sizeof(bf16) * BlockMinorSize*blocks_width, 0,0,0}; // define os strides na GMEM
    //(0,0,0), pois não há stride para calcular
    uint32_t smem_box_shape[5] = {uint32_t(BlockMinorSize), uint32_t(BlockMajorSize),1,1,1}; // define os tiles que vão para a SMEM
    uint32_t smem_box_stride[5] = {1,1,1,1,1}; // define os strides na SMEM, strides unitários, pois os dados na SMEM ficam contíguos

    // 2 - número de dimensões ativas
    // gmem_prob_stride + 1 - avança o ponteiro
    CUresult result = cuTensorMapEncodeTiled(
        tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gmem_address, gmem_prob_shape,
        gmem_prob_stride + 1, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);

}


template <int BlockMajorSize, int BlockMinorSize>
__host__ static inline CUtensorMap* allocate_and_create_tensor_map(bf16* src, int blocks_height, int blocks_width) {
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host;
    create_tensor_map<BlockMajorSize, BlockMinorSize>(&tma_map_host, src, blocks_height, blocks_width);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64(float d[4][8], bf16* sA, bf16* sB)
{
    uint64_t desc_a = 0x4000004000000000 | (matrix_descriptor_encode(static_cast<uint32_t>(__cvta_generic_to_shared(&sA[0]))));
    uint64_t desc_b = 0x4000004000000000 | (matrix_descriptor_encode(static_cast<uint32_t>(__cvta_generic_to_shared(&sB[0]))));
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
          "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
          "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
          "+f"(d[3][6]), "+f"(d[3][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}


template<int BM, int BN, int BK, int WGMMA_M, int WGMMA_N, int WGMMA_K, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) gemmkernel(int M, int N, int K, bf16* C, const CUtensorMap* tensorMapA, const CUtensorMap* tensorMapB)
{
    __shared__ alignas(128) bf16 sA[BM * BK]; // tma requer alinhamento de 128 bytes
    __shared__ alignas(128) bf16 sB[BK * BN];// tma requer alinhamento de 128 bytes

    float d[WGMMA_N/16][8] = {};// [64/16][8] - [4][8], registrador dos acumuladores

    static_assert(sizeof(d) * 128 == BM * BN * sizeof(float));

    const int num_blocks_k = K / BK;
    int num_block_n = blockIdx.x % (N / BN);
    int num_block_m = blockIdx.x / (N / BN);
    #pragma nv_diag_suppress static_var_with_dynamic_init // usamos esse pragma para  desativar os avisos sobre a inicialização dinâmica de variáveis estáticas, que vai ocorrer com o barrier A e B
    __shared__ barrier barA;
    __shared__ barrier barB;

    // Apenas a thread 0 inicializa as barreiras
    if(threadIdx.x == 0){
        init(&barA, blockDim.x); // barreira de cópia de A
        init(&barB, blockDim.x);    // barreira de cópia de B
        cde::fence_proxy_async_shared_cta(); // garante que a inicialização das barreiras seja enxergadas por todas as threads
    }
    __syncthreads();

    barrier::arrival_token tokenA, tokenB; // garanta que as cópias de A e B realmente tenham sido feitas
    for(int block_k_iter = 0; block_k_iter < num_blocks_k; ++ block_k_iter)
    {
        if(threadIdx.x == 0){
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[0], tensorMapA, block_k_iter*BK, num_block_m*BM, barA);
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sA)); //barrier_arrive_tx(barreira, 1(indica que estamos fazendo uma atualização), deve preencher toda a SMEM)
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[0], tensorMapB, block_k_iter*BK, num_block_n*BN, barB);
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sB)); 
        }else{
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }

        barA.wait(std::move(tokenA)); // garante que todas as threads tenha chegado
        barB.wait(std::move(tokenB));
        __syncthreads();

        warpgroup_arrive();
        wgmma64<1,1,1,0,0>(d, &sA[0], &sB[0]);
        wgmma64<1,1,1,0,0>(d, &sA[WGMMA_K], &sB[WGMMA_K]);
        wgmma64<1,1,1,0,0>(d, &sA[WGMMA_K * 2], &sB[2 * WGMMA_K]);
        wgmma64<1,1,1,0,0>(d, &sA[WGMMA_K * 3], &sB[WGMMA_K * 3]);
        warpgroup_commit_batch();
        warpgroup_wait<0>();

    }

    {
        int tid = threadIdx.x;
        int lane = tid % 32;
        int warp = tid / 32;
        uint32_t row = warp*16 + lane / 4;
        bf16 *block_C = C + num_block_n*BN*M + num_block_m*BM;

        for(int m_it = 0; m_it<BM/WGMMA_M; ++m_it){
            for(int n_it = 0; n_it<BN/WGMMA_N; ++n_it){
                for(int w = 0; w < WGMMA_N/16; ++w){
                    int col = 16*w + 2*(tid % 4);
                    #define IDX(i, j) ((j + n_it*WGMMA_N) * M + ((i) + m_it * WGMMA_M))
                   
                    block_C[IDX(row, col)] = __float2bfloat16(d[w][0]);
                    block_C[IDX(row, col+1)] = __float2bfloat16(d[w][1]);
                    block_C[IDX(row+8, col)] = __float2bfloat16(d[w][2]);
                    block_C[IDX(row+8, col+1)] = __float2bfloat16(d[w][3]);
    
                    block_C[IDX(row, col+8)] = __float2bfloat16(d[w][4]);
                    block_C[IDX(row, col+9)] = __float2bfloat16(d[w][5]);
                    block_C[IDX(row+8, col+8)] = __float2bfloat16(d[w][6]);
                    block_C[IDX(row+8, col+9)] = __float2bfloat16(d[w][7]);

                    #undef IDX                  
                }
            }
        }
    }

}

void neogemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
        
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int NUM_THREADS = 128;

    at::BFloat16* data_ptr_A = A.data_ptr<at::BFloat16>();
    at::BFloat16* data_ptr_B = B.data_ptr<at::BFloat16>();
    at::BFloat16* data_ptr_C = C.data_ptr<at::BFloat16>();

    bf16* bf16_data_ptr_A = reinterpret_cast<bf16*>(data_ptr_A);
    bf16* bf16_data_ptr_B = reinterpret_cast<bf16*>(data_ptr_B);    
    bf16* bf16_data_ptr_C = reinterpret_cast<bf16*>(data_ptr_C);    
    
    CUtensorMap *d_tma_map_A = 0;
    CUtensorMap *d_tma_map_B = 0;
    
    int prev_m, prev_n, prev_k;
    if (!d_tma_map_A) {
        // Passar ponteiros corretos ao invés de tensors
        d_tma_map_A = allocate_and_create_tensor_map<BM, BK>(bf16_data_ptr_A, M / BM, K / BK);
        d_tma_map_B = allocate_and_create_tensor_map<BN, BK>(bf16_data_ptr_B, N / BN, K / BK);
 
        prev_n = N;
        prev_k = K;
    }
    
    // Assert cached values are of same size
    assert(M == prev_m && N == prev_n && K == prev_k);
    
    gemmkernel<
        /* BM */ BM,
        /* BN */ BN,
        /* BK */ BK,
        /* WGMMA_M */ 64,
        /* WGMMA_N */ 64,
        /* WGMMA_K */ 16,
        /* NUM_THREADS */ NUM_THREADS>
        <<<(M/BM) * (N/BN), NUM_THREADS>>>(M, N, K, bf16_data_ptr_C, d_tma_map_A, d_tma_map_B);
}
PYBIND11_MODULE(neogemm, m) {
    m.def("neogemm", &neogemm, "GEMM",
          py::arg("A"), py::arg("B"), py::arg("C"));
}
