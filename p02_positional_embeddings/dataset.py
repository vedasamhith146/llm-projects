from datasets import load_dataset
import tiktoken
import torch

enc=tiktoken.get_encoding('gpt2')

dataset=load_dataset("roneneldan/TinyStories",split="train")

def tokenize(example):
    return {
        "ids": enc.encode_ordinary(example["text"])
    }

dataset=dataset.map(tokenize,remove_columns=["text"])

all_tokens=[]

for example in dataset:
    all_tokens.extend(example["ids"])
    all_tokens.append(enc.eot_token)

tokens=torch.tensor(all_tokens,dtype=torch.long)
n=int(0.9*len(tokens))

train_tokens=tokens[:n]
val_tokens=tokens[n:]

torch.save(train_tokens, "train_tokens.pt")
torch.save(val_tokens, "val_tokens.pt")
