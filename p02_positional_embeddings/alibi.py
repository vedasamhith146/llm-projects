import torch
import math
from token_tensor import token_embd
g=torch.Generator()
g.manual_seed(42)
dim=token_embd.size(1)
Wq=torch.randn((dim,dim),generator=g)
Wk=torch.randn((dim,dim),generator=g)
q=token_embd @ Wq
k=token_embd @ Wk
n_heads=4
d_head=dim//n_heads
T=token_embd.size(0)
q=q.view(T,n_heads,d_head).transpose(0,1) #final shape is (n_heads,T,d_heads)
k=k.view(T,n_heads,d_head).transpose(0,1) #final shape is (n_heads,T,d_heads)
attn=q @ k.transpose(1,2)
attn=attn/math.sqrt(d_head)
B=torch.zeros((T,T))

for i in range(T):
    for j in range(T):
        B[i,j]=j-i
mask=torch.tril(torch.ones(T,T))
B=B.masked_fill(mask==0,0.0)

for i in range(n_heads):
    m=pow(pow(2,-8/n_heads),i+1)
    attn_temp=attn[i]+m*B
    attn_temp=attn_temp.masked_fill(mask==0,float('-inf'))
    attn[i]=torch.softmax(attn_temp,dim=-1)










