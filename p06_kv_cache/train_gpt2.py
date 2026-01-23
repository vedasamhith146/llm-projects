import os
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass


device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    def forward(self, idx, targets=None,start_pos=0):
        B, T = idx.size()
        pos = torch.arange(start_pos, start_pos+T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h: x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nondecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nondecay_params, 'weight_decay': 0.0}
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)

    def clear_kv_cache(self):
        for block in self.transformer.h:
            block.attn.kvcache.reset()

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
        self.kvcache=KVCache()
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

        self.kvcache.update(k,v)

        k_hist=self.kvcache.get_cache()["key"]
        v_hist=self.kvcache.get_cache()["value"]

        att=(q @ k_hist.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        
        if T>1:
            mask=torch.tril(torch.ones(T,T,device=x.device)).view(1,1,T,T)
            att=att.masked_fill(mask[:,:,:T,:T]==0,float('-inf'))
        
        att=F.softmax(att,dim=-1)
        y = att @ v_hist
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

class DataLoaderLite:
    def __init__(self, B, T):
        self.B, self.T = B, T
        files = sorted([x for x in os.listdir('.') if x.endswith('.bin')])
        if len(files) == 0: 
            print("WARNING: No .bin files found. Did you run fineweb.py?")
        self.shards = files
        self.reset()

    def reset(self):
        self.shard_idx = 0
        self.tokens = self.load_tokens(self.shards[self.shard_idx])
        self.pos = 0

    def load_tokens(self, filename):
        npt = np.fromfile(filename, dtype=np.uint16)
        return torch.tensor(npt.astype(np.int64), dtype=torch.long)

    def next_batch(self):
        B, T = self.B, self.T
        if self.pos + B * T + 1 > len(self.tokens):
            self.shard_idx = (self.shard_idx + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.shard_idx])
            self.pos = 0
        buf = self.tokens[self.pos : self.pos+B*T+1]
        x, y = buf[:-1].view(B, T), buf[1:].view(B, T)
        self.pos += B * T
        return x, y
    
class KVCache:
    
    def __init__(self):
        self.cache={"key":None,"value":None}

    def reset(self):
        self.cache={"key":None,"value":None}
    
    def update(self,key,value):
        if self.cache["key"] is None:
            self.cache["key"]=key
            self.cache["value"]=value
        else:
            self.cache["key"]=torch.cat([self.cache["key"],key],dim=2)
            self.cache["value"]=torch.cat([self.cache["value"],value],dim=2)
    
    def get_cache(self):
        return self.cache
    
