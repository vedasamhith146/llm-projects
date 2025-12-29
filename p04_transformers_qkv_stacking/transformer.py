from bpe_encode import encode
from bpe_decode import decode
import torch
torch.set_num_threads(4)
device="cuda" if torch.cuda.is_available() else "cpu"
import torch.nn.functional as F
import math
import torch.optim as optim
g=torch.Generator()
g.manual_seed(42)

#initialising of variables
vocab_size=1264
d_model=256
B=64
T=128
h=8
d_head=d_model//h
n_layers=6

class Attention:
    def __init__(self):
        self.Wq=torch.randn((d_model,d_model),generator=g)
        self.Wk=torch.randn((d_model,d_model),generator=g)
        self.Wv=torch.randn((d_model,d_model),generator=g)
        self.register_mask=torch.tril(torch.ones(T,T))
    def forward(self,x):
        #calculation of Q,K,V
        Q= x @ self.Wq
        K= x @ self.Wk
        V= x @ self.Wv
        #viewing Q,K,V as (h,B,T,d_head)
        Q=Q.view(B,T,h,d_head).permute(2,0,1,3)
        K=K.view(B,T,h,d_head).permute(2,0,1,3)
        V=V.view(B,T,h,d_head).permute(2,0,1,3)
        S= Q @ K.transpose(2,3)/math.sqrt(d_head)
        S=S.masked_fill(self.register_mask==0,float('-inf'))
        A=torch.softmax(S,dim=-1)
        outs= A @ V
        outs=outs.transpose(0,1).transpose(1,2).contiguous().view(B,T,d_model)
        return outs
    def parameters(self):
        params=[self.Wq,self.Wk,self.Wv]
        return params
    def __call__(self,x):
        return self.forward(x)

class FeedForward:
    def __init__(self):
        self.Wff_1=torch.randn((d_model,4*d_model),generator=g)
        self.Bff_1=torch.randn((4*d_model),generator=g)
        self.Wff_2=torch.randn((4*d_model,d_model),generator=g)
        self.Bff_2=torch.randn((d_model),generator=g)
    def forward(self,x):
        x= x @ self.Wff_1 + self.Bff_1
        x=F.gelu(x,approximate='tanh')
        x= x @ self.Wff_2 + self.Bff_2
        return x
    def parameters(self):
        params=[self.Wff_1,self.Bff_1,self.Wff_2,self.Bff_2]
        return params
    def __call__(self,x):
        return self.forward(x)

class LayerNorm:
    def __init__(self):
        self.mul_bias=torch.ones((d_model))
        self.add_bias=torch.zeros((d_model))
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        var=(((x-mean)**2).mean(dim=-1,keepdim=True))
        x=(x-mean)/torch.sqrt(var+1e-5)
        return self.mul_bias * x + self.add_bias
    def parameters(self):
        params=[self.mul_bias,self.add_bias]
        return params
    def __call__(self,x):
        return self.forward(x)

class Linear:
    def __init__(self):
        self.Wl=torch.randn((d_model,vocab_size),generator=g)
        self.Bl=torch.randn((vocab_size),generator=g)
    def forward(self,x):
        x= x @ self.Wl + self.Bl
        return x
    def parameters(self):
        params=[self.Wl,self.Bl]
        return params
    def __call__(self,x):
        return self.forward(x)

class Layer:
    def __init__(self):
        self.LayerNorm_1=LayerNorm()
        self.attention=Attention()
        self.LayerNorm_2=LayerNorm()
        self.feedforward=FeedForward()
    def forward(self,x):
        x=x+self.attention(self.LayerNorm_1(x))
        x=x+self.feedforward(self.LayerNorm_2(x))
        return x
    def parameters(self):
        params=[]
        params=params+self.LayerNorm_1.parameters()
        params=params+self.attention.parameters()
        params=params+self.LayerNorm_2.parameters()
        params=params+self.feedforward.parameters()
        return params
    def __call__(self,x):
        return self.forward(x)

class Model:
    def __init__(self):
        self.token_table=torch.randn((vocab_size,d_model),generator=g)
        self.pos_table=torch.randn((T,d_model),generator=g)
        self.layers=[Layer() for _ in range(n_layers)]
        self.LayerNorm_final=LayerNorm()
        self.Linear=Linear()
    def forward(self,x):
        x=self.token_table[x]
        x=x+self.pos_table
        for layer in self.layers:
            x=layer(x)
        x=self.LayerNorm_final(x)
        logits=self.Linear(x)
        return logits
    def parameters(self):
        params=[self.token_table,self.pos_table]
        params=params+self.LayerNorm_final.parameters()
        params=params+self.Linear.parameters()
        for layer in self.layers:
            params=params+layer.parameters()
        return params
    def __call__(self,x):
        return self.forward(x)

with open("data/test.txt",'r') as f:
    tokens=f.read()
token_ids=encode(tokens)
tokens_per_batch=B*T
num_batches=len(token_ids)//tokens_per_batch
token_ids=torch.tensor(token_ids,dtype=torch.long).to(device)

model=Model().to(device)
for p in model.parameters():
    p.requires_grad=True
optimizer=optim.AdamW(
    params=model.parameters(),
    lr=3e-4,
    betas=(0.9,0.999),
    eps=1e-8,
    weight_decay=1e-3
)

for step in range(10000):
    i=step % num_batches
    x=token_ids[tokens_per_batch*i:tokens_per_batch*(i+1)].view(B,T).to(device)
    logits= model(x)
    logits=logits[:,:-1,:]
    targets=x[:,1:]
    loss=F.cross_entropy(logits.reshape(-1,1264),targets.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step==2000:
        for param_group in optimizer.param_groups:
            param_group['lr']*=0.1
    if step%200==0:
        print(step,loss.item())





