import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from benchmark import CausalSelfAttention,run_benchmark

def perform_ablation():
    n_head=32
    kv_head_options=[1,2,4,8,16,32]
    groups_list=[]
    latencies=[]
    print(f"Starting Ablation study(Total Heads={n_head})")
    print('-'*40)

    for kv_head in kv_head_options:
        groups=n_head//kv_head
        latency=run_benchmark(n_head=n_head,n_kv_head=kv_head)
        latencies.append(latency)
        groups_list.append(groups)
        print(f"Groups:{groups} (KV Heads:{kv_head}) | Latency:{latency:.4f}ms")

    plt.figure(figsize=(10,6))
    plt.plot(groups_list,latencies,marker='o',linestyle='-',color='b')
    
    plt.title(f"Impact of GQA Groups on Latency (Total Heads={n_head})")
    plt.xlabel("Number of groups (n_head/n_kv_head)")
    plt.ylabel("Average Latency(ms)")
    plt.grid(True,linestyle='--',alpha=0.7)
    plt.show()

perform_ablation()

