import torch
from token_tensor import token_embd
from bpe_encode import encode
g=torch.Generator()
g.manual_seed(42)
dim=token_embd.size(1)
P=torch.randn((400,dim),generator=g)
with open("data/test.txt",'r') as f:
    text=f.read()
text=encode(text)

T=token_embd.size(0)
pos_embd=P[torch.arange(T)]
total_embeddings_learned=token_embd+pos_embd






