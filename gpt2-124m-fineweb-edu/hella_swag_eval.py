import os
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass
from datasets import load_dataset
import tiktoken
import sys

device='mps' if torch.backends.mps.is_available() else 'cpu'
checkpoint="gpt2_step_11500.pt"
print(f"Using device {device}")


@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50304 
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx,idx_len,inp_len, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h: x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        logits=logits[:,idx_len-1:inp_len-1,:]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

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
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))
    

    
model=GPT2(GPT2Config())
model.to(device)

state_dict=torch.load(checkpoint,map_location=device)

for k in list(state_dict.keys()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)

model.load_state_dict(state_dict)

hs=load_dataset("Rowan/hellaswag",split="validation")
no_of_examples=len(hs)

print("loaded dataset successfully")

enc=tiktoken.get_encoding('gpt2')

no_of_correct_labels=0

for i in range(no_of_examples):
    idx=enc.encode_ordinary((hs[i])['ctx'])
    inp_1=torch.tensor(idx + enc.encode_ordinary(((hs[i])['endings'])[0]),dtype=torch.long).unsqueeze(0).to(device)
    inp_2=torch.tensor(idx + enc.encode_ordinary(((hs[i])['endings'])[1]),dtype=torch.long).unsqueeze(0).to(device)
    inp_3=torch.tensor(idx + enc.encode_ordinary(((hs[i])['endings'])[2]),dtype=torch.long).unsqueeze(0).to(device)
    inp_4=torch.tensor(idx + enc.encode_ordinary(((hs[i])['endings'])[3]),dtype=torch.long).unsqueeze(0).to(device)


    out_1=inp_1[:,len(idx):inp_1.size(1)].to(device)
    out_2=inp_2[:,len(idx):inp_2.size(1)].to(device)
    out_3=inp_3[:,len(idx):inp_3.size(1)].to(device)
    out_4=inp_4[:,len(idx):inp_4.size(1)].to(device)

    correct_label=int((hs[i])['label'])

    with torch.no_grad():
        _,loss_1=model(inp_1,len(idx),inp_1.size(1),out_1)
        _,loss_2=model(inp_2,len(idx),inp_2.size(1),out_2)
        _,loss_3=model(inp_3,len(idx),inp_3.size(1),out_3)
        _,loss_4=model(inp_4,len(idx),inp_4.size(1),out_4)

    losses=[loss_1.item(),loss_2.item(),loss_3.item(),loss_4.item()]
    output_label= losses.index(min(losses))
    if output_label==correct_label:
        no_of_correct_labels+=1
    if (i+1)%100==0:
        print(f"Completed evaluation on {i+1} steps")

accuracy= (no_of_correct_labels/no_of_examples)*100
print(f"Accuracy of your model on hellaswag is {accuracy}")







    



    






