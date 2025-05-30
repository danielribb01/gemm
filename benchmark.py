import torch
import neoGEMM
import time

# Dimensões
M = 1024
N = 1024
K = 1024
# Matrizes em bfloat16
tensorA = torch.randn(M, K, device='cuda').to(torch.bfloat16)
tensorB = torch.randn(N, K, device='cuda').to(torch.bfloat16)
tensorC = torch.zeros(M, N, device='cuda').to(torch.bfloat16)  # FP32

print("=== COMPARAÇÃO SIMPLIFICADA ===")

# PyTorch: BF16→FP32 antes da multiplicação (referência)
print("\n1. PyTorch (BF16→FP32):")
torch_result = tensorA @ tensorB.t()
print(f"Shape: {torch_result.shape}, dtype: {torch_result.dtype}")
print(f"Amostra [0:3, 0:3]:\n{torch_result[:5, :5]}")

# Seu kernel customizado
print("\n2. neoGemm kernel:")
neoGEMM.neoGemmV1(tensorA, tensorB, tensorC)
tensorC = tensorC.t()
print(f"Shape: {tensorC.shape}, dtype: {tensorC.dtype}")
print(f"Amostra [0:3, 0:3]:\n{tensorC[:5, :5]}")



# Análise das diferenças
print("\n=== ANÁLISE DE DIFERENÇAS ===")
diff = torch.abs(tensorC - torch_result)
print(f"Erro máximo: {diff.max().item():.8f}")
print(f"Erro médio: {diff.mean().item():.8f}")
print(f"Erro RMS: {torch.sqrt(torch.mean(diff**2)).item():.8f}")

# Estatísticas dos valores para contexto
print(f"\nMagnitude dos valores:")
print(f"PyTorch - min: {torch_result.min().item():.4f}, max: {torch_result.max().item():.4f}")
print(f"neoGemm - min: {tensorC.min().item():.4f}, max: {tensorC.max().item():.4f}")

# Verificação de proximidade com diferentes tolerâncias
tolerances = [1e-2, 1e-3, 1e-4, 1e-5]
print(f"\n=== VERIFICAÇÃO DE PROXIMIDADE ===")
for tol in tolerances:
    is_close = torch.allclose(tensorC, torch_result, atol=tol, rtol=tol)
    print(f"Tolerância {tol:.0e}: {'✓ PASSOU' if is_close else '✗ FALHOU'}")

# Percentual de elementos próximos
close_elements = torch.abs(tensorC - torch_result) < 1e-3
percentage_close = (close_elements.sum().float() / close_elements.numel() * 100).item()
print(f"\nElementos dentro de 1e-3: {percentage_close:.2f}%")



flops = 2 * M * N * K 

print("\n1. PyTorch (BF16→FP32):")
torch.cuda.synchronize()

# Warmup
for _ in range(10):
    _ = tensorA @ tensorB
torch.cuda.synchronize()

tensorAA = tensorA
tensorBB = tensorB.t()

# Medição real
start_time = time.time()
for _ in range(100):
    torch_result = tensorAA @ tensorBB
torch.cuda.synchronize()
pytorch_time = (time.time() - start_time) / 100

print(f"Tempo médio: {pytorch_time * 1000:.3f} ms")
pytorch_tflops = flops / pytorch_time / 1e12
print(f"**PERFORMANCE: {pytorch_tflops:.2f} TFLOPS**")

# Seu kernel customizado
print("\n2. neoGemm kernel:")
torch.cuda.synchronize()

# Warmup
for _ in range(100):
    tensorC.zero_()
    neoGEMM.neoGemmV1(tensorA, tensorB, tensorC)
torch.cuda.synchronize()

# Medição real
start_time_neogemm = time.time()
for _ in range(100):
    neoGEMM.neoGemmV1(tensorA, tensorB, tensorC)
torch.cuda.synchronize()
neogemm_time = (time.time() - start_time_neogemm) / 100

print(f"Tempo médio: {neogemm_time * 1000:.3f} ms")
neogemm_tflops = flops / neogemm_time / 1e12
print(f"**PERFORMANCE: {neogemm_tflops:.2f} TFLOPS**")
