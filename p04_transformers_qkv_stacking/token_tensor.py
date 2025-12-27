import torch
from bpe_encode import encode
g=torch.Generator()
g.manual_seed(123)
def get_tokens(text,d):
    text=(encode(text))
    lookup_table=torch.randn((1264,d),generator=g)
    tokens=lookup_table[text]
    return tokens
with open("data/test.txt") as f:
    text=f.read()
token_embd=get_tokens(text,16)