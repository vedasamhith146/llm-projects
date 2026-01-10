import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GPT2Config:
   block_size: int=1024
   vocab_size: int=50257
   n_layer: int=12
   n_head: int=12
   n_embd: int=768

  
class MLP(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.c_fc=nn.Linear(config.n_embd,config.n_embd*4)
    self.c_proj=nn.Linear(config.n_embd*4,config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT=1.0
  def forward(self,x):
    x=self.c_fc(x)
    x=F.gelu(x,approximate='tanh')
    x=self.c_proj(x)
    return x

class CasualSelfAttention(nn.Module):
  def __init__(self,config):
    super().__init__()
    assert config.n_embd % config.n_head==0
    self.c_attn=nn.Linear(config.n_embd,config.n_embd*3)
    self.c_proj=nn.Linear(config.n_embd,config.n_embd)
    self.n_embd=config.n_embd
    self.n_head=config.n_head
    self.head_dim=config.n_embd//config.n_head
    self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size),persistent=False)
    self.c_proj.NANOGPT_SCALE_INIT=1.0
  def forward(self,x):
    B,T,C=x.size() 
    qkv=self.c_attn(x)
    q,k,v=qkv.split(self.n_embd,dim=2)
    q=q.view(B,T,self.n_head,self.head_dim).transpose(1,2)
    k=k.view(B,T,self.n_head,self.head_dim).transpose(1,2)
    v=v.view(B,T,self.n_head,self.head_dim).transpose(1,2)

    att=(q@k.transpose(-2,-1))/(self.head_dim**0.5)
    att=att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
    att=F.softmax(att,dim=-1)

    out=att@v 
    out=out.transpose(1,2).contiguous().view(B,T,C)
    out=self.c_proj(out)
    return out

class Block(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.ln_1=nn.LayerNorm(config.n_embd)
    self.attn=CasualSelfAttention(config)
    self.ln_2=nn.LayerNorm(config.n_embd)
    self.mlp=MLP(config)
  def forward(self,x):
    x=x+self.attn(self.ln_1(x))
    x=x+self.mlp(self.ln_2(x))
    return x

class GPT2(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.config=config
    self.transformer=nn.ModuleDict(dict(
        wte=nn.Embedding(config.vocab_size,config.n_embd),
        wpe=nn.Embedding(config.block_size,config.n_embd),
        h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ln_f=nn.LayerNorm(config.n_embd)
    ))
    self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
    self.transformer.wte.weight=self.lm_head.weight

    self.apply(self._init_weights)

  def _init_weights(self,module):
     if isinstance(module,nn.Linear):
        std=0.02
        if hasattr(module,'NANOGPT_SCALE_INIT'):
           std*=(2*self.config.n_layer)**-0.5
        torch.nn.init.normal_(module.weight,mean=0.0,std=std)
        if module.bias is not None:
           torch.nn.init.zeros_(module.bias)
     elif isinstance(module,nn.Embedding):
        torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    
  def forward(self,idx,targets=None):
    B,T=idx.size()
    assert T<=self.config.block_size, f"Cannot forward sequence of length {T}"
    pos=torch.arange(0,T,dtype=torch.long,device=idx.device)
    pos_emb=self.transformer.wpe(pos)
    tok_emb=self.transformer.wte(idx)
    x=pos_emb+tok_emb
    for block in self.transformer.h:
        x=block(x)
    x=self.transformer.ln_f(x)
    logits=self.lm_head(x)
    loss=None
    if targets is not None:
      loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
    return logits,loss
#-------------------------------------------------------------------------------------------------------------------------------
import tiktoken


class DataLoaderLite:
    def __init__(self,B,T):
        self.B=B
        self.T=T

        with open('input.txt','r') as f:
            text=f.read()
        enc=tiktoken.get_encoding('gpt2')
        tokens=enc.encode(text)
        self.tokens=torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)//(B*T)} batches")
        self.current_position=0
    def next_batch(self):
        B,T=self.B,self.T
        buf=self.tokens[self.current_position:self.current_position+B*T+1]
        x=(buf[:-1]).view(B,T)
        y=(buf[1:]).view(B,T)
        self.current_position+=B*T
        if self.current_position+(B*T+1)>len(self.tokens):
            self.current_position=0
        return x,y
    
#--------------------------------------------------------------------------------------------------------------------------------
device="cpu"
if torch.cuda.is_available():
  device="cuda"
elif hasattr(torch.backends,"mps") and torch.mps.is_available():
  device="mps"
print(f"using device {device}")

torch.manual_seed(1337)

train_loader=DataLoaderLite(B=16,T=1024)

torch.set_float32_matmul_precision('high')

model=GPT2(GPT2Config())
model.to(device)

optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
for i in range(50):
  x,y=train_loader.next_batch()
  x,y=x.to(device),y.to(device)
  optimizer.zero_grad()
  with torch.autocast(device_type=device,dtype=torch.bfloat16):
    logits,loss=model(x,y)
  logits,loss=model(x,y)
  loss.backward()
  optimizer.step()
  torch.cuda.synchronize()
  print(f"step {i} loss: {loss.item()}")
     




