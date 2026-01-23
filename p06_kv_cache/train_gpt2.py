import os
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass


total_batch_size = 524288 
B = 64              
T = 1024                  
grad_accum_steps = total_batch_size // (B * T)
max_steps = 19073       
learning_rate = 6e-4
warmup_steps = 715

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high') 

# --- MODEL DEFINITION ---
@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50304 # Padded for efficiency
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

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
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
        x=x[:,-1:,:]
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        self.kvcache.update(k,v)
        k,v=self.kvcache.get_cache()["key"],self.kvcache.get_cache()["value"]
        q = q.view(B, 1, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = torch.softmax((q @ k.transpose(2,3))*((C//self.n_head)**-0.5))
        y = y @ v
        y = y.transpose(1, 2).contiguous().view(B, 1, C)
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
    
    def update(self,key,value):
        if self.cache["key"] is None:
            self.cache["key"]=key
            self.cache["value"]=value
        else:
            self.cache["key"]=torch.cat([self.cache["key"],key],dim=1)
            self.cache["value"]=torch.cat([self.cache["value"],value],dim=1)
    
    def get_cache(self):
        return self.cache
    

if __name__ == '__main__':
    train_loader = DataLoaderLite(B=B, T=T)
    model = GPT2(GPT2Config(vocab_size=50304))
    model.to(device)

    start_step = 0
    checkpoint_path = "/workspace/gpt2_step_2000.pt"
    
    if os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        start_step = 2000
        print(f"Success! Resuming training from step {start_step}")
    else:
        print("No checkpoint found. Starting from scratch.")


    model = torch.compile(model) 
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, device=device)
    
    def get_lr(it):
        if it < warmup_steps: return learning_rate * (it + 1) / warmup_steps
        if it > max_steps: return learning_rate * 0.1
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return learning_rate * 0.1 + coeff * (learning_rate * 0.9)

    print("Starting training...")
    

    for step in range(start_step, max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0
        
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
            
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        
        lr = get_lr(step)
        for param_group in optimizer.param_groups: param_group['lr'] = lr
        
        optimizer.step()
        torch.cuda.synchronize()
        
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tok_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (t1 - t0)
        print(f"step {step:4d} | loss: {loss_accum.item():.4f} | lr: {lr:.2e} | dt: {dt:.2f}ms | tok/sec: {tok_sec:.2f}")

        # Save periodically
        if step > 0 and step % 500 == 0:
            torch.save(model.state_dict(), f"gpt2_step_{step}.pt")
            print(f"Saved checkpoint: gpt2_step_{step}.pt")