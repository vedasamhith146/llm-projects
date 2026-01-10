import torch
import torch.nn as nn
from torch.nn import functional as F

class mlp(nn.Module):
  def __init__(self,n_embd):
    super().__init__()
    self.c_fc=nn.Linear(n_embd,n_embd*4)
    self.c_proj=nn.Linear(n_embd*4,n_embd)
  def forward(self,x):
    x=self.c_fc(x)
    x=F.gelu(x,approximate='tanh')
    x=self.c_proj(x)
    return x

class attn(nn.Module):
  def __init__(self,n_embd,n_head):
    super().__init__()
    self.c_attn=nn.Linear(n_embd,n_embd*3)
    self.c_proj=nn.Linear(n_embd,n_embd)
    self.n_embd=n_embd
    self.n_head=n_head
    self.head_dim=n_embd//n_head
  def forward(self,x):
    B,T,C=x.size() 
    qkv=self.c_attn(x)
    q,k,v=qkv.split(self.n_embd,dim=2)
    q=q.view(B,T,self.n_head,self.head_dim).transpose(1,2)
    k=k.view(B,T,self.n_head,self.head_dim).transpose(1,2)
    v=v.view(B,T,self.n_head,self.head_dim).transpose(1,2)

    att=(q@k.transpose(-2,-1))/(self.head_dim**0.5)
    mask=torch.tril(torch.ones(T,T,device=x.device))
    att=att.masked_fill(mask==0,float('-inf'))
    att=F.softmax(att,dim=-1)

    out=att@v 
    out=out.transpose(1,2).contiguous().view(B,T,C)
    out=self.c_proj(out)
    return out

class Head(nn.Module):
  def __init__(self,n_embd,n_head):
    super().__init__()
    self.ln_1=nn.LayerNorm(n_embd)
    self.attn=attn(n_embd,n_head)
    self.ln_2=nn.LayerNorm(n_embd)
    self.mlp=mlp(n_embd)
  def forward(self,x):
    x=x+self.attn(self.ln_1(x))
    x=x+self.mlp(self.ln_2(x))
    return x

class transformer(nn.Module):
  def __init__(self,vocab_size,block_size,n_embd,n_layers,n_head):
    super().__init__()
    self.wte=nn.Embedding(vocab_size,n_embd)
    self.wpe=nn.Embedding(block_size,n_embd)
    self.h=nn.ModuleList([Head(n_embd,n_head) for _ in range(n_layers)])
    self.ln_f=nn.LayerNorm(n_embd)
  def forward(self,x):
    B,T=x.size()
    te=self.wte(x)
    positions=torch.arange(T,device=x.device).unsqueeze(0)
    pe=self.wpe(positions)
    x=te+pe
    for block in self.h:
        x=block(x)
    x=self.ln_f(x)
    return x

class GPT2(nn.Module):
  def __init__(self,vocab_size,block_size,n_embd,n_layers,n_head):
    super().__init__()
    self.transformer=transformer(vocab_size,block_size,n_embd,n_layers,n_head)
    self.lm_head=nn.Linear(n_embd,vocab_size,bias=False)
    self.lm_head.weight = self.transformer.wte.weight
  def forward(self,x):
    x=self.transformer(x)
    x=self.lm_head(x)
    return x

my_model=GPT2(50257,1024,768,12,12)  


from transformers import GPT2LMHeadModel
hf_model=GPT2LMHeadModel.from_pretrained("gpt2")
hf_state=hf_model.state_dict()
my_state=my_model.state_dict()
hf_keys=set(hf_state.keys())
my_keys=set(my_state.keys())
print("missing in my model",hf_keys-my_keys)
print("extra in my model",my_keys-hf_keys)


new_state = {}
for k in hf_state:
    if k not in my_state:
        continue
    
    hf_w = hf_state[k]
    my_w = my_state[k]
    
    if k.endswith(".weight") and any(x in k for x in ["c_attn", "c_proj", "c_fc"]):
        assert hf_w.dim() == 2, f"Expected 2D weight for {k}, got {hf_w.dim()}"
        new_state[k] = hf_w.T
    
    elif hf_w.shape == my_w.shape:
        new_state[k] = hf_w
    else:
        raise ValueError(f"Shape mismatch at {k}: HF {hf_w.shape}, MY {my_w.shape}")

my_model.load_state_dict(new_state, strict=True) 
hf_model.eval()
x = torch.randint(0, hf_model.config.vocab_size, (1, 10))

with torch.no_grad():
    hf_logits = hf_model(x).logits
    my_logits = my_model(x)

print("Max logit diff:", (hf_logits - my_logits).abs().max())
print(my_model.transformer.wte.weight is my_model.lm_head.weight)




