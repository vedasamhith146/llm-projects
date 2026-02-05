import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision('high')
batch_size=64
block_size=256
max_iter=500
eval_interval=100
learning_rate=3e-4
device='cuda' if torch.cuda.is_available() else 'cpu'
n_embd=192
n_head=6
n_layer=4



print(f"Running on the device:{device}")

os.system("wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
with open('input.txt','r',encoding='utf-8') as f:
    text=f.read()

chars=sorted(list(set(text)))
stoi={ch:i for i,ch in enumerate(chars)}
encode= lambda s:[stoi[c] for c in s]

data=torch.tensor(encode(text),dtype=torch.long)
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

def get_batch(split):
    data=train_data if split=='train' else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1]for i in ix])
    return x.to(device),y.to(device)


class SwiGLU(nn.Module):
    def __init__(self,n_embd,hid_dim):
        super().__init__()
        self.hid_dim=hid_dim
        self.w_gate_val=nn.Linear(n_embd,2*hid_dim,bias=False)
        self.w_out=nn.Linear(hid_dim,n_embd,bias=False)
    def forward(self,x):
        fused=self.w_gate_val(x)
        gate,val=fused.split(self.hid_dim,dim=-1)
        return self.w_out(F.silu(gate)*val)
    
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        if config['activation']=='swiglu':
            hidden_dim=int(4*n_embd*(2/3))
            self.net=SwiGLU(n_embd,hidden_dim)
        else:
            self.net=nn.Sequential(
                nn.Linear(n_embd,4*n_embd),
                nn.GELU(),
                nn.Linear(4*n_embd,n_embd)
            )

    def forward(self,x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        if config['norm']=='rmsnorm':
            self.ln1=nn.RMSNorm(n_embd)
            self.ln2=nn.RMSNorm(n_embd)
        else:
            self.ln1=nn.LayerNorm(n_embd)
            self.ln2=nn.LayerNorm(n_embd)

        self.attn=nn.MultiheadAttention(n_embd,n_head,batch_first=True)
        self.mlp=MLP(config)

    def forward(self,x):
        mask=torch.nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(device)
        attn_out,_=self.attn(self.ln1(x),self.ln1(x),self.ln1(x),attn_mask=mask,is_causal=True)
        x=x+attn_out
        x=x+self.mlp(self.ln2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.token_embedding=nn.Embedding(len(chars),n_embd)
        self.position_embedding=nn.Embedding(block_size,n_embd)
        self.blocks=nn.Sequential(*[Block(config) for _ in range(n_layer)])

        if config['norm']=='rmsnorm':
            self.ln_f=nn.RMSNorm(n_embd)
        else:
            self.ln_f=nn.LayerNorm(n_embd)

        self.lm_head=nn.Linear(n_embd,len(chars),bias=False)

    def forward(self,idx,targets=None):
        B,T=idx.shape
        x=self.token_embedding(idx)+self.position_embedding(torch.arange(T,device=device))
        x=self.blocks(x)
        x=self.ln_f(x)
        logits=self.lm_head(x)

        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits,loss
    

def run_expirement(config_name,config_dict):
    print(f"\n----Starting {config_name}----")

    torch.manual_seed(1337)
    if torch.cuda.is_available():torch.cuda.manual_seed(1337)

    model=GPT(config_dict).to(device)
    try:
        model=torch.compile(model)
        print("Model compiled with torch.compile() for extra speed")
    except:
        pass

    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
    losses=[]
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time=time.perf_counter()

    for iter in range(max_iter):
        xb,yb=get_batch('train')
        with torch.cuda.amp.autocast(enabled=(device=='cuda')):
            logits,loss=model(xb,yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if iter%100==0:
            print(f"Step {iter}: Loss{loss.item():.4f}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time=time.perf_counter()
    net=end_time-start_time
    print(f"Finished in {net:.2f}s ({max_iter/net:.1f} it/s) | Final loss:{losses[-1]:.4f}")
    return losses

configs={
    "1.LayerNorm + GELU":{'norm':'layernorm','activation':'gelu'},
    "2.RMSNorm + GELU":{'norm':'rmsnorm','activation':'gelu'},
    "3.RMSNorm +SwiGLU":{'norm':'rmsnorm','activation':'swiglu'},
}

results={}

for name,conf in configs.items():
    results[name]=run_expirement(name,conf)

plt.figure(figsize=(10,6))
for name, loss_curve in results.items():
    smooth_curve = torch.tensor(loss_curve).view(-1, 10).mean(dim=1).numpy()
    plt.plot(smooth_curve, label=name)

plt.title("Fast Ablation: Norm & Activation Variants (PyTorch Optimized)")
plt.xlabel("Steps (x10)")
plt.ylabel("Training Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("fast_ablation_results.png")
print("\nPlot saved to 'fast_ablation_results.png'")



