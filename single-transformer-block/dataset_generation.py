import tiktoken
import torch

enc=tiktoken.get_encoding('gpt2')
with open("data/tiny.txt","r") as f:
    text=f.read()

all_tokens=enc.encode(text)

tokens=torch.tensor(all_tokens,dtype=torch.long)
n=int(0.9*len(tokens))

train_tokens=tokens[:n]
val_tokens=tokens[n:]

torch.save(train_tokens, "train_tokens.pt")
torch.save(val_tokens, "val_tokens.pt")