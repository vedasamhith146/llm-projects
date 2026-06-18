import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
import sys


device='cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

batch_size=4
learning_rate=3e-4
max_steps=200


@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class RotationalPositionalEmbedding(nn.Module):
    def __init__(self,d_model,max_len=1024):
        super().__init__()
        position=torch.arange(max_len).unsqueeze(1)
        thetha_term=torch.exp(torch.repeat_interleave(torch.arange(0,d_model,2),repeats=2)*(-torch.log(torch.tensor(10000.0))/d_model))
        cos_terms=torch.cos(position*thetha_term)
        sin_terms=torch.sin(position*thetha_term)

        self.register_buffer("cos_term",cos_terms)
        self.register_buffer("sin_term",sin_terms)

    def reverse_tensor(self,input_tensor):
        B,N,T,D=input_tensor.size()
        modified_tensor=input_tensor.view(B,N,T,D//2,2)
        flipped_tensor=torch.flip(modified_tensor,dims=[-1])
        reversed_tensor= (flipped_tensor*torch.tensor((-1,1),device=input_tensor.device)).view(B,N,T,D)
        return reversed_tensor
    
    def forward(self,input_tensor):
        T=input_tensor.size(-2)
        cos_term=self.cos_term[:T]
        sin_term=self.sin_term[:T]
        reversed_tensor=self.reverse_tensor(input_tensor)
        final_tensor=(input_tensor*cos_term)+(reversed_tensor*sin_term)
        return final_tensor

class mini_former(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.miniformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            #rmsnorm_f=RMSNorm(config)
        ))
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.lm_head.weight=self.miniformer.wte.weight
        self.per_layer_activations={}
        self.per_layer_activations_mean={}
        self.per_layer_activations_std={}
        self.per_layer_max={}
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self,idx,targets=None):
        self.per_layer_activations={}
        self.per_layer_activations_mean={}
        self.per_layer_activations_std={}
        i=0
        x=self.miniformer.wte(idx)
        print(f"\n{'='*70}")
        print(f"EMBEDDING (h.0): norm={torch.sqrt(torch.sum(x**2)).item():.2e}, mean={x.mean().item():.2e}, max_abs={x.abs().max().item():.2e}")
        print(f"{'='*70}")
        
        for block in self.miniformer.h:
            print(f"\n--- BLOCK {i} ---")
            print(f"  INPUT:  norm={torch.sqrt(torch.sum(x**2)).item():.2e}, mean={x.mean().item():.2e}, max_abs={x.abs().max().item():.2e}")
            print(f"  INPUT is all zeros? {(x == 0).all().item()}")
            print(f"  INPUT has non-zero? {(x != 0).any().item()}")
            if (x != 0).any():
                print(f"  INPUT non-zero count: {(x != 0).sum().item()}")
                print(f"  INPUT smallest non-zero: {x[x != 0].abs().min().item():.2e}")
            
            # Track inside the block
            x = block(x, verbose=True, block_idx=i)
            
            self.per_layer_activations[f"h.{i}"]=torch.sqrt(torch.sum(x**2)).item()
            self.per_layer_activations_mean[f"h.{i}"]=torch.mean(x).item()
            self.per_layer_activations_std[f"h.{i}"]=math.sqrt(torch.var(x).item())
            self.per_layer_max[f"h.{i}"]=torch.max((torch.abs(x))).item()
            
            print(f"  OUTPUT: norm={self.per_layer_activations[f'h.{i}']:.2e}, mean={self.per_layer_activations_mean[f'h.{i}']:.2e}, max_abs={self.per_layer_max[f'h.{i}']:.2e}")
            print(f"  OUTPUT is all zeros? {(x == 0).all().item()}")
            print(f"  OUTPUT has non-zero? {(x != 0).any().item()}")
            if (x != 0).any():
                print(f"  OUTPUT non-zero count: {(x != 0).sum().item()}")
                print(f"  OUTPUT smallest non-zero: {x[x != 0].abs().min().item():.2e}")
            
            i+=1
        
        print(f"\n{'='*70}")
        print(f"FINAL h.11: norm={torch.sqrt(torch.sum(x**2)).item():.2e}")
        print(f"{'='*70}")
        
        #x=self.miniformer.rmsnorm_f(x)
        logits=self.lm_head(x)
        loss=None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        #self.rmsnorm_1=RMSNorm(config)
        self.attn=CausalSelfAttention(config)
        #self.rmsnorm_2=RMSNorm(config)
        self.mlp=SwiGLU_MLP(config)

    def forward(self,x, verbose=False, block_idx=0):
        if verbose:
            print(f"  [Block {block_idx}] Before attn: norm={torch.sqrt(torch.sum(x**2)).item():.2e}")
        x=self.attn(x, verbose=verbose, block_idx=block_idx)
        if verbose:
            print(f"  [Block {block_idx}] After attn:  norm={torch.sqrt(torch.sum(x**2)).item():.2e}, all_zero? {(x == 0).all().item()}")
        x=self.mlp(x, verbose=verbose, block_idx=block_idx)
        if verbose:
            print(f"  [Block {block_idx}] After mlp:   norm={torch.sqrt(torch.sum(x**2)).item():.2e}, all_zero? {(x == 0).all().item()}")
        #x=self.attn(self.rmsnorm_1(x))
        #x=self.mlp(self.rmsnorm_2(x))
        return x
    
class RMSNorm(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.alpha=nn.Parameter(torch.ones((config.n_embd)))
        self.eps=1e-5

    def forward(self,x):
        rms= ((torch.sum((x**2),dim=-1))/x.size(-1)).unsqueeze(-1)
        output = self.alpha*(x/torch.sqrt(rms+self.eps)) 
        return output
    
class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj=nn.Linear(config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1.0
        self.n_embd=config.n_embd
        self.n_head=config.n_head
        self.rope=RotationalPositionalEmbedding(config.n_embd//config.n_head)
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)))

    def forward(self,x, verbose=False, block_idx=0):
        B,T,C=x.size()
        qkv = self.c_attn(x)
        if verbose and block_idx >= 10:  # Only print details for last few blocks
            print(f"    [Attn {block_idx}] c_attn output: norm={torch.sqrt(torch.sum(qkv**2)).item():.2e}, all_zero? {(qkv == 0).all().item()}")
            print(f"    [Attn {block_idx}] c_attn bias: max_abs={self.c_attn.bias.abs().max().item():.2e}")
        
        q,k,v=qkv.split(self.n_embd,dim=2)
        q=q.view(B,T,self.n_head,self.n_embd//self.n_head).transpose(1,2)
        k=k.view(B,T,self.n_head,self.n_embd//self.n_head).transpose(1,2)
        v=v.view(B,T,self.n_head,self.n_embd//self.n_head).transpose(1,2)
        
        if verbose and block_idx >= 10:
            print(f"    [Attn {block_idx}] q norm: {torch.sqrt(torch.sum(q**2)).item():.2e}, all_zero? {(q == 0).all().item()}")
            print(f"    [Attn {block_idx}] k norm: {torch.sqrt(torch.sum(k**2)).item():.2e}, all_zero? {(k == 0).all().item()}")
            print(f"    [Attn {block_idx}] v norm: {torch.sqrt(torch.sum(v**2)).item():.2e}, all_zero? {(v == 0).all().item()}")
        
        q=self.rope(q)
        k=self.rope(k)
        
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.n_embd//self.n_head)
        
        if verbose and block_idx >= 10:
            print(f"    [Attn {block_idx}] att scores: max={att.max().item():.2e}, min={att.min().item():.2e}")
            print(f"    [Attn {block_idx}] att has -inf? {(att == float('-inf')).any().item()}")
            print(f"    [Attn {block_idx}] att has finite? {(att != float('-inf')).any().item()}")
            if (att != float('-inf')).any():
                finite_att = att[att != float('-inf')]
                print(f"    [Attn {block_idx}] finite att: max={finite_att.max().item():.2e}, min={finite_att.min().item():.2e}")
        
        att = att.masked_fill(self.bias[:T,:T]==0,float('-inf'))
        att=F.softmax(att,dim=-1)
        
        if verbose and block_idx >= 10:
            print(f"    [Attn {block_idx}] softmax att: max={att.max().item():.2e}, min={att.min().item():.2e}, sum={att[0,0].sum().item():.4f}")
        
        y= att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        if verbose and block_idx >= 10:
            print(f"    [Attn {block_idx}] att@v output: norm={torch.sqrt(torch.sum(y**2)).item():.2e}, all_zero? {(y == 0).all().item()}")
        
        out = self.c_proj(y)
        
        if verbose and block_idx >= 10:
            print(f"    [Attn {block_idx}] c_proj output: norm={torch.sqrt(torch.sum(out**2)).item():.2e}, all_zero? {(out == 0).all().item()}")
            print(f"    [Attn {block_idx}] c_proj bias: max_abs={self.c_proj.bias.abs().max().item():.2e}")
        
        return out
    
class SwiGLU_MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.W=nn.Linear(config.n_embd,4*config.n_embd)
        self.V=nn.Linear(config.n_embd,4*config.n_embd)
        self.P=nn.Linear(4*config.n_embd,config.n_embd)
        self.P.NANOGPT_SCALE_INIT=1.0
    def forward(self,x, verbose=False, block_idx=0):
        if verbose and block_idx >= 10:
            print(f"    [MLP {block_idx}] input: norm={torch.sqrt(torch.sum(x**2)).item():.2e}, all_zero? {(x == 0).all().item()}")
        
        input_1=self.W(x)
        input_2=self.V(x)
        
        if verbose and block_idx >= 10:
            print(f"    [MLP {block_idx}] W output: norm={torch.sqrt(torch.sum(input_1**2)).item():.2e}, all_zero? {(input_1 == 0).all().item()}")
            print(f"    [MLP {block_idx}] V output: norm={torch.sqrt(torch.sum(input_2**2)).item():.2e}, all_zero? {(input_2 == 0).all().item()}")
            print(f"    [MLP {block_idx}] W bias: max_abs={self.W.bias.abs().max().item():.2e}")
            print(f"    [MLP {block_idx}] V bias: max_abs={self.V.bias.abs().max().item():.2e}")
        
        swish=F.silu(input_1)
        
        if verbose and block_idx >= 10:
            print(f"    [MLP {block_idx}] silu(Wx): norm={torch.sqrt(torch.sum(swish**2)).item():.2e}, all_zero? {(swish == 0).all().item()}")
        
        gated = swish * input_2
        
        if verbose and block_idx >= 10:
            print(f"    [MLP {block_idx}] silu*V: norm={torch.sqrt(torch.sum(gated**2)).item():.2e}, all_zero? {(gated == 0).all().item()}")
        
        output=self.P(gated)
        
        if verbose and block_idx >= 10:
            print(f"    [MLP {block_idx}] P output: norm={torch.sqrt(torch.sum(output**2)).item():.2e}, all_zero? {(output == 0).all().item()}")
            print(f"    [MLP {block_idx}] P bias: max_abs={self.P.bias.abs().max().item():.2e}")
        
        return output
    
if __name__=="__main__":
    config=Config()
    model=mini_former(config)
    model.to(device)
    
    # DON'T compile for first run — we want clean prints
    # model=torch.compile(model)
    
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,betas=(0.9,0.95),weight_decay=0.1)
    
    # Create dummy data if train_tokens.pt doesn't exist
    try:
        train_tokens=torch.load("train_tokens.pt")
        val_tokens=torch.load("val_tokens.pt")
    except:
        print("Creating dummy data...")
        train_tokens = torch.randint(0, config.vocab_size, (100000,))
        val_tokens = torch.randint(0, config.vocab_size, (10000,))

    def get_batch(split):
        data= train_tokens if split =="train" else val_tokens
        ix=torch.randint(len(data)-config.block_size-1,(batch_size,))
        x=torch.stack([data[i:i+config.block_size] for i in ix])
        y=torch.stack([data[i+1:i+config.block_size+1] for i in ix])
        return x,y   
    
    def estimate_loss(split, eval_iters=50):
        model.eval()
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)
    
    # Run just ONE step with full diagnostics
    print(f"\n{'#'*70}")
    print(f"# FIRST TRAINING STEP (step 0) - FULL DIAGNOSTICS")
    print(f"{'#'*70}")
    
    x, y = get_batch("train")
    x, y = x.to(device), y.to(device)
    
    print(f"\nBatch shape: x={x.shape}, y={y.shape}")
    
    logits, loss = model(x, y)
    
    print(f"\nLoss: {loss.item():.4f}")
    
    print(f"\n{'#'*70}")
    print(f"# BACKWARD PASS")
    print(f"{'#'*70}")
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Check gradients to h.11 biases
    print(f"\nGradients to h.11 parameters:")
    for name, param in model.named_parameters():
        if 'miniformer.h.11' in name:
            grad_norm = param.grad.norm().item() if param.grad is not None else 0
            print(f"  {name}: grad_norm={grad_norm:.4e}, param_norm={param.data.norm().item():.4e}")
    
    print(f"\n{'#'*70}")
    print(f"# OPTIMIZER STEP")
    print(f"{'#'*70}")
    
    optimizer.step()
    
    # Check updated biases
    print(f"\nUpdated h.11 biases:")
    for name, param in model.named_parameters():
        if 'miniformer.h.11' in name and 'bias' in name:
            print(f"  {name}: max_abs={param.data.abs().max().item():.2e}")
    
    print(f"\n{'#'*70}")
    print(f"# SECOND FORWARD PASS (after optimizer step)")
    print(f"{'#'*70}")
    
    # Run again to see if h.11 changed
    logits2, loss2 = model(x, y)
    print(f"\nLoss after step: {loss2.item():.4f}")
    
    # Save results
    print(f"\n{'#'*70}")
    print(f"# SUMMARY")
    print(f"{'#'*70}")
    print(f"Per-layer activations from FIRST forward (init weights):")
    for k, v in sorted(model.per_layer_activations.items()):
        print(f"  {k}: {v:.2e}")