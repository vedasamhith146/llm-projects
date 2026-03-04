#------------------------------------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import os
import tiktoken
import json
#------------------------------------------------------------------------------------------------------------------------------------------------------
if torch.cuda.is_available():
    device='cuda'
elif torch.backends.mps.is_available():
    device='mps'
else:
    device='cpu'
#------------------------------------------------------------------------------------------------------------------------------------------------------
@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50304 # Padded for efficiency
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512

mask_token_id=50257
B,T=32,128
#------------------------------------------------------------------------------------------------------------------------------------------------------
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

    def forward(self, idx, S, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h: 
            x = block(x,S)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
#------------------------------------------------------------------------------------------------------------------------------------------------------    

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x,S):
        x = x+self.attn(self.ln_1(x),S)
        x = x+self.mlp(self.ln_2(x))
        return x
#------------------------------------------------------------------------------------------------------------------------------------------------------   
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    def forward(self, x,S):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        mask=torch.tril(torch.ones(T,T,dtype=torch.bool,device=x.device))
        mask[:S,:S]=True
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
#------------------------------------------------------------------------------------------------------------------------------------------------------    

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))
    
#------------------------------------------------------------------------------------------------------------------------------------------------------

# importing data 
if not os.path.exists('input.txt'):   
    os.system("curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
with open('input.txt','r',encoding='utf-8') as f:
    text=f.read()

#------------------------------------------------------------------------------------------------------------------------------------------------------
#tokenizing text using GPT2-tokenizer
tokenizer=tiktoken.get_encoding('gpt2')
tokens=torch.tensor(tokenizer.encode(text),dtype=torch.long,device=device)

#------------------------------------------------------------------------------------------------------------------------------------------------------
#implementing dataloader function 

def get_batch(tokens,B,T):
    ix=torch.randint(len(tokens)-B*T-1,(1,)).item()
    input_tokens=tokens[ix:ix+B*T].view(B,T)
    S=torch.randint(1,T,(1,)).item()
    output_tokens=tokens[ix+1:ix+B*T+1].view(B,T).clone()
    if S>1:
        output_tokens[:,:S-1]=-100
    return input_tokens.to(device),output_tokens.to(device),S
#------------------------------------------------------------------------------------------------------------------------------------------------------
model=GPT2(GPT2Config())
model.to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
#------------------------------------------------------------------------------------------------------------------------------------------------------
print("starting training...")

losses=[]
for step in range(5000):
    xb,yb,S=get_batch(tokens,B,T)
    logits,loss=model(xb,S,yb)
    losses.append(loss.item())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step%100==0:
        print(f"Step {step} | Loss {loss.item():.4f}")

with open("prefix_loss.json","w") as f:
    json.dump(losses,f)

print("finished training")
#------------------------------------------------------------------------------------------------------------------------------------------------------


print("\n--- Generating Prefix LM Sample ---")
model.eval()
prompt = "KING RICHARD: \nO, what a"
idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

# S is the length of our initial prompt. The model will treat this fully bidirectionally.
S = idx.size(1) 
max_new_tokens = 50

with torch.no_grad():
    for _ in range(max_new_tokens):
        # Pass both idx and S to your Prefix LM
        logits, _ = model(idx, S) 
        
        logits = logits[:, -1, :] 
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

print(f"Prefix (S={S}):", prompt)
print("Output:\n", tokenizer.decode(idx[0].tolist()))





    






    













