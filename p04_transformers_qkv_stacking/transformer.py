from bpe_encode import encode
from bpe_decode import decode
import torch
import torch.nn.functional as F
import math
import torch.optim as optim
import torch.nn as nn
torch.set_num_threads(4)
device="cuda" if torch.cuda.is_available() else "cpu"

#initialising of variables
vocab_size=1264
d_model=256
B=64
T=128
h=8
d_head=d_model//h
n_layers=6

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq=nn.Linear(d_model,d_model,bias=False)
        self.Wk=nn.Linear(d_model,d_model,bias=False)
        self.Wv=nn.Linear(d_model,d_model,bias=False)
        self.register_buffer(
            "mask",torch.tril(torch.ones(T,T))
        )
    def forward(self,x):
        B,T,_=x.shape
        Q=self.Wq(x)
        K=self.Wk(x)
        V=self.Wv(x)
        #viewing Q,K,V as (h,B,T,d_head)
        Q=Q.view(B,T,h,d_head).transpose(1,2)
        K=K.view(B,T,h,d_head).transpose(1,2)
        V=V.view(B,T,h,d_head).transpose(1,2)
        att= Q @ K.transpose(-2,-1)/math.sqrt(d_head)
        att=att.masked_fill(self.mask==0,float('-inf'))
        att=torch.softmax(att,dim=-1)
        out= att @ V
        out=out.transpose(1,2).contiguous().view(B,T,d_model)
        return out

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d_model,4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model,d_model),
        )
    def forward(self,x):
        return self.net(x)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1=nn.LayerNorm(d_model)
        self.attn=Attention()
        self.ln2=nn.LayerNorm(d_model)
        self.feedforward=FeedForward()
    def forward(self,x):
        x=x+self.attn(self.ln1(x))
        x=x+self.feedforward(self.ln2(x))
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb=nn.Embedding(vocab_size,d_model)
        self.pos_emb=nn.Embedding(T,d_model)
        self.layers=nn.ModuleList([Layer() for _ in range(n_layers)])
        self.ln_f=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,vocab_size)
    def forward(self,x):
       B,T=x.shape
       pos=torch.arange(T,device=x.device)
       x=self.token_emb(x)+self.pos_emb(pos)
       for layer in self.layers:
           x=layer(x)
       x=self.ln_f(x)
       return self.head(x)

with open("data/test.txt",'r') as f:
    tokens=f.read()
token_ids=encode(tokens)
tokens_per_batch=B*T
num_batches=len(token_ids)//tokens_per_batch
token_ids=torch.tensor(token_ids,dtype=torch.long).to(device)

model=Model().to(device)
optimizer=optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9,0.999),
    eps=1e-8,
    weight_decay=1e-3
)

for step in range(1000):
    i=step % num_batches
    x=token_ids[tokens_per_batch*i:tokens_per_batch*(i+1)].view(B,T).to(device)
    logits= model(x)
    logits=logits[:,:-1,:]
    targets=x[:,1:]
    loss=F.cross_entropy(logits.reshape(-1,vocab_size),targets.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step==2000:
        for param_group in optimizer.param_groups:
            param_group['lr']*=0.1
    if step%200==0:
        print(step,loss.item())





