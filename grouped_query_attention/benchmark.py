import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

class CausalSelfAttention(nn.Module):
    def __init__(self,n_head,n_kv_head,n_embd=768):
        super().__init__()
        self.n_head=n_head
        self.n_kv_head=n_kv_head
        self.n_embd=n_embd

        self.head_dim=n_embd//n_head
        self.kv_dim=n_kv_head*self.head_dim
        self.groups=n_head//n_kv_head
        self.c_attn=nn.Linear(n_embd,n_embd+2*self.kv_dim,bias=False)
        self.c_proj=nn.Linear(n_embd,n_embd,bias=False)

    def forward(self,x,k_cache=None,v_cache=None):
        B,T,C=x.size()
        q,k,v=self.c_attn(x).split((self.n_embd,self.kv_dim,self.kv_dim),dim=2)
        q=q.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        k=k.view(B,T,self.n_kv_head,self.head_dim).transpose(1,2)
        v=v.view(B,T,self.n_kv_head,self.head_dim).transpose(1,2)

        if k_cache is not None:
            k=k_cache
            v=v_cache
        if self.groups>1:
           q=q.contiguous().view(B,self.n_kv_head,self.groups,T,self.head_dim)
           k=k.unsqueeze(2)
           v=v.unsqueeze(2)

        y=F.scaled_dot_product_attention(q,k,v,is_causal=False)
        y=y.transpose(1,2).contiguous().view(B,T,C)
        return self.c_proj(y)


def run_benchmark(n_head=8,n_kv_head=8,batch_size=64,seq_len=1024):

    if torch.backends.mps.is_available():
        device=torch.device('mps')
    else:
        print("MPS not found ! Using CPU")
        device=torch.device('cpu')

    model=CausalSelfAttention(n_head,n_kv_head).to(device)

    x=torch.randn(batch_size,1,768,device=device)
    head_dim=768//n_head
    k_cache=torch.randn(batch_size,n_kv_head,seq_len,head_dim,device=device)
    v_cache=torch.randn(batch_size,n_kv_head,seq_len,head_dim,device=device)

    print(f"> Warming up",end="")

    for _ in range(10):
        with torch.no_grad():
            _=model(x,k_cache,v_cache)
    
    torch.mps.synchronize()
    print("Done .")

    iterations=100
    start_time=time.perf_counter()

    for _ in range(iterations):
        with torch.no_grad():
            _=model(x,k_cache,v_cache)

    torch.mps.synchronize()

    end_time=time.perf_counter()

    avg_latency=((end_time-start_time)/iterations)*1000

    return avg_latency

if __name__=="__main__":

    print("\n--- Project Step 2: Speed Measurement ---")
    print("Conditions: Batch=64, History=1024 tokens\n")

    latency_mha = run_benchmark(n_head=8, n_kv_head=8)
    print(f"Vanilla MHA Latency: {latency_mha:.2f} ms")

    latency_gqa = run_benchmark(n_head=8, n_kv_head=2)
    print(f"GQA (Group=4) Latency: {latency_gqa:.2f} ms")


    speedup = latency_mha / latency_gqa
    print(f"\nResult: GQA is {speedup:.2f}x faster on this Mac.")



            
    