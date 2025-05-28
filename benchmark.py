import torch
import neogemm

# Dimensões
M = 1024
N = 1024
K = 1024

# Matrizes em bfloat16
tensorA = torch.randn(M, K, device='cuda').to(torch.bfloat16)
tensorB = torch.randn(K, N, device='cuda').to(torch.bfloat16)
tensorC = torch.zeros(M, N, device='cuda')  # FP32

print("=== COMPARAÇÃO SIMPLIFICADA ===")

# PyTorch: BF16→FP32 antes da multiplicação (referência)
print("\n1. PyTorch (BF16→FP32):")
torch_result = tensorA.float() @ tensorB.float()
print(f"Shape: {torch_result.shape}, dtype: {torch_result.dtype}")
print(f"Amostra [0:3, 0:3]:\n{torch_result[:3, :3]}")

# Seu kernel customizado
print("\n2. neoGemm kernel:")
neogemm.neoGemm(tensorA, tensorB, tensorC)
print(f"Shape: {tensorC.shape}, dtype: {tensorC.dtype}")
print(f"Amostra [0:3, 0:3]:\n{tensorC[:3, :3]}")

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
