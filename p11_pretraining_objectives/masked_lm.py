#---------------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import os
import tiktoken
import json
#---------------------------------------------------------------------------------------------------------------------------------

if torch.cuda.is_available():
    device='cuda'
elif torch.backends.mps.is_available():
    device='mps'
else:
    device='cpu'

#----------------------------------------------------------------------------------------------------------------------------------

@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50304 ##1
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512

mask_token_id=50257
B,T=32,128

#-----------------------------------------------------------------------------------------------------------------------------------

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
        self.transformer.wte.weight = self.lm_head.weight  ##2

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h: 
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

#-----------------------------------------------------------------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x+self.attn(self.ln_1(x))
        x = x+self.mlp(self.ln_2(x))
        return x
    
#---------------------------------------------------------------------------------------------------------------------------------------
    
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False) ##3
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
#------------------------------------------------------------------------------------------------------------------------------------------   

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

#------------------------------------------------------------------------------------------------------------------------------------------    

if not os.path.exists('input.txt'):   
    os.system("curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
with open('input.txt','r',encoding='utf-8') as f:
    text=f.read()

#------------------------------------------------------------------------------------------------------------------------------------------------

tokenizer=tiktoken.get_encoding('gpt2')
tokens=torch.tensor(tokenizer.encode(text),dtype=torch.long,device=device)

#------------------------------------------------------------------------------------------------------------------------------------------------

def get_batch(tokens,B,T):
    ix=torch.randint(len(tokens)-B*T,(1,)).item()
    input_tokens=tokens[ix:ix+B*T].view(B,T).clone() ##4
    output_tokens=input_tokens.clone()
    prob_mat=torch.rand((B,T),device=device)
    mask_eig=prob_mat<0.80*0.15
    input_tokens[mask_eig]=mask_token_id            #first 80% of the tokens
    mask_ten_one=(prob_mat>0.80*0.15) & (prob_mat<0.90*0.15)
    rand_mat=torch.randint(0,50257,(B,T),device=device)
    input_tokens[mask_ten_one]=rand_mat[mask_ten_one]               #second 10% of the tokens

    mask_eigfif=prob_mat>0.15
    output_tokens[mask_eigfif]=-100

    return input_tokens.to(device),output_tokens.to(device)

#---------------------------------------------------------------------------------------------------------------------------------------------------

model=GPT2(GPT2Config())
model.to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)

#----------------------------------------------------------------------------------------------------------------------------------------------------

print("starting training...")

losses=[]
for step in range(5000):
    xb,yb=get_batch(tokens,B,T)
    logits,loss=model(xb,yb)
    losses.append(loss.item())
    optimizer.zero_grad(set_to_none=True) ##3
    loss.backward()
    optimizer.step()

    if step%100==0:
        print(f"Step {step} | Loss {loss.item():.4f}")

with open("masked_loss_3.json","w") as f:
    json.dump(losses,f)

print("finished training")

#-----------------------------------------------------------------------------------------------------------------------------------------------------

print("\n----Generating masked LM Sample-------")
model.eval()
part1=tokenizer.encode("The king is ")
mask_token=[50257]
part2=tokenizer.encode(" and the queen is sad.")
idx_list=part1+mask_token+part2
idx=torch.tensor(idx_list,dtype=torch.long,device=device).unsqueeze(0)
mask_pos=len(part1)

with torch.no_grad():
    logits=model(idx)
    if isinstance(logits,tuple):
        logits=logits[0]
    mask_logits=logits[0,mask_pos,:]
    predicted_token_id=torch.argmax(mask_logits,dim=-1).item()

idx[0,mask_pos]=predicted_token_id
print("Original: The king is [MASK] and the queen is sad.")
print("Filled :",tokenizer.decode(idx[0].tolist()))









    






    













