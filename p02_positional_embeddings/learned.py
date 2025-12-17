import torch
from token_tensor import token_embd
from bpe_encode import encode
g=torch.Generator()
g.manual_seed(42)
def pos_embd_learned(vocab_size,dim,idx):
    vec=torch.randn((vocab_size,dim),generator=g)
    vec=vec[idx]
    return vec


with open("data/test.txt",'r') as f:
    text=f.read()
text=encode(text)

pos_embd=pos_embd_learned(1264,16,text)
total_embeddings_learned=token_embd+pos_embd






