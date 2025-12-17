import torch 
import math
from token_tensor import get_tokens,token_embd
def pos_embd_sin(no_of_pos,d):
    w_pe=[]
    for m in range(no_of_pos):
        pe=[]
        for n in range(d):
            i=n//2
            if n%2==0:
                pe.append(math.sin(m/pow(10000,2*i/d)))
            else:
                pe.append(math.cos(m/pow(10000,2*i/d)))
        w_pe.append(pe)
    w_pe=torch.tensor(w_pe)
    return w_pe 

no_of_pos=token_embd.size(0)
pos_embd=pos_embd_sin(no_of_pos,16)
total_embeddings_sinusoidal=token_embd+pos_embd









