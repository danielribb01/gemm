#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time>
#include <unist.h>
#include <ctime>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_bf16.h>
#include <cassert>
#include <torch/extension.h>
#include <pybind11/pybind11.h>


#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y))


typedef __nv_bfloat16 bf16;

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



__device__ uint64_t make_smem_desc(bf16* ptr)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); //converte um ponteiro da matriz para um ponteiro que aponta para o endereço da SMEM
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16; // Leading Dimension = 16 elementos por linha
    desc |= matrix_descriptor_encode((uint64_t)1024) << 32; // stride 1024
    desc |= 1llu << 62;
    return desc;
}

// wgmma.fence - ele força uma ordenação de acesso nos registrados entre as operações wgmma.mma_async
// Apenas os registradores do acumulador e o registrador contendo os fragmentos da matriz A realiza a operação

__device__ void warpgroup_arrive()
{
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

// o warpgroup_commit e o warpgroup_wait são usados para esperar a operação WGMMA Async ficar completa

__device__ void warpgroup_commit_batch()
{
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}



// cada warpgroup realiza 8 operações para a dimensão do bm64nNb16
template<int N>
__device__ void warpgroup_wait()
{
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0,7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}


