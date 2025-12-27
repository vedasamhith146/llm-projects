import torch
from bpe_encode import encode
g=torch.Generator()
g.manual_seed(42)
d=16
lookup_table=torch.randn((1264,d),generator=g)
def get_tokens(text):
    text=(encode(text))
    tokens=lookup_table[text]
    return tokens
with open("data/test.txt") as f:
    text=f.read()
token_embd=get_tokens(text)