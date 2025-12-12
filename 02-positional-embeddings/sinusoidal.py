import torch 
import math
def pos_embd_sin(pos,d):
    pe=[]
    for i in range(d):
        if i%2==0:
            pe.append(math.sin(pos/pow(10000,2*i/d)))
        else:
            pe.append(math.cos(pos/pow(10000,2*i/d)))
    pe=torch.tensor(pe)
    return pe
print(pos_embd_sin(1,16))
