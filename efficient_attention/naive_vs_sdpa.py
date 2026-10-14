import torch
import torch.nn as nn
import math
import torch.nn.functional as F


torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

B=1
H=12
T=256
D=768
D_head=D//H

Q = torch.randn(B, H, T, D_head, device=device)
K = torch.randn(B, H, T, D_head, device=device)
V = torch.randn(B, H, T, D_head, device=device)

def naive():
    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(D_head)
    mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool),diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    output = weights @ V
    return output.view(-1)

def sdpa():
    output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    return output.view(-1)

if __name__=="__main__":
    output_naive= naive()
    output_sdpa = sdpa()
    diff= output_naive - output_sdpa

    max_error = diff.abs().max().item()
    mae = diff.abs().mean().item()
    rmse = diff.square().mean().sqrt().item()

    print(f"Maximum absolute error: {max_error:.8e}")
    print(f"Mean absolute error:    {mae:.8e}")
    print(f"RMSE:                   {rmse:.8e}")


