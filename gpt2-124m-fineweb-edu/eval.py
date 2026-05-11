import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass

checkpoint="gpt2_step_11500.pt"
val_shard="edufineweb_train_000099.bin"
device = "mps" if torch.backends.mps.is_available() else "cpu"
B,T=8,1024
eval_steps=50 


@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos    = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x      = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x      = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss
    

print("Using device {device}")
state_dict=torch.load(checkpoint,map_location=device)

for key in list(state_dict.keys()):
    if key.startswith('_orig_mod.'):
        state_dict[key[len('_orig_mod.'):]]=state_dict.pop(key)

model=GPT2(GPT2Config())
model.load_state_dict(state_dict)
model.to(device)

print("Model loaded from checkpoint")

tokens=np.fromfile(val_shard,dtype=np.uint16)
tokens=torch.tensor(tokens,dtype=torch.long)

print(f"tokens in the shard : {len(tokens)}")

losses=[]
with torch.no_grad():
    for i in range(eval_steps):
        start= i*B*T
        buf=tokens[start:start+B*T+1].to(device)
        x=buf[:-1].view(B,T)
        y=buf[1:].view(B,T)
        logits,loss=model(x,y)
        losses.append(loss)
        print(f"step :{i+1}/{eval_steps} | loss : {loss.item():.4f} ")

avg_loss=sum(losses)/len(losses)
perplexity=math.exp(avg_loss)

print(f"Val loss : {avg_loss:4f}")
print(f"Tokens evaluated on {eval_steps*B*T}")
print(f"perplexity:{perplexity:.2f}")

print("Baselines:")
print("GPT2 124M (openai) : loss  ~3.11 : perplexity ~22.4")







