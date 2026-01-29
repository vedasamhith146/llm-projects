import torch
from moe import MOElayer,din,noe,topk
import matplotlib.pyplot as plt

def compare_flops(din,noe,topk,seq_len=1024,batch_size=1):
    params_per_expert=din*din
    dense_ops_per_token=noe*params_per_expert
    total_dense_flops=batch_size*seq_len*dense_ops_per_token
    sparse_ops_per_token=topk*params_per_expert
    router_ops_per_token=din*noe
    total_sparse_flops=batch_size*seq_len*(router_ops_per_token+sparse_ops_per_token)
    savings=total_dense_flops/total_sparse_flops
    print(f"--- Efficiency Report ---")
    print(f"Model Configuration: {noe} Experts, Routing Top-{topk}")
    print(f"Dense Compute (Baseline): {total_dense_flops:.2e} FLOPs")
    print(f"Sparse Compute (MoE):     {total_sparse_flops:.2e} FLOPs")
    print(f"Speedup / Savings:        {savings:.2f}x cheaper")
    
    return total_dense_flops, total_sparse_flops

d_flops,s_flops=compare_flops(din,noe,topk)
plt.figure(figsize=(6, 6))
plt.bar(['Dense (Baseline)', 'MoE (Sparse)'], [d_flops, s_flops], color=['gray', 'green'])
plt.ylabel("FLOPs (Lower is Better)")
plt.title("Computational Cost Comparison")
plt.show()
