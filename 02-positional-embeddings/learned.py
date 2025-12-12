import torch
g=torch.Generator()
g.manual_seed(42)
def pos_embd_learned(vocab_size,dim,pos):
    vec=torch.randn((vocab_size,dim),generator=g)
    pos_vec=vec[pos]
    return pos_vec
print(pos_embd_learned(1264,16,1))



