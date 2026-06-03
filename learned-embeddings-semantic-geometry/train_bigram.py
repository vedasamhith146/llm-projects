import torch
import torch.nn as nn
import torch.nn.functional as F
import json

if torch.backends.mps.is_available():
    device='mps'
else:
    device='cpu'

with open("data/tiny.txt",'r') as file:
    text=file.read()
    text=text.lower()

chars=sorted(list(set(text)))
stoi={s:i for i,s in enumerate(chars)}

with open("stoi.json","w") as f:
    json.dump(stoi,f)

vocab_size=len(stoi)
embedding_dim=2
max_steps=20
batch_size=4096


class bigram_model(nn.Module):
    def __init__(self,vocab_size,embedding_dim):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.linear=nn.Linear(embedding_dim,vocab_size)
    def forward(self,inputs,targets):
        x=self.embedding(inputs)
        logits=self.linear(x)
        loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits,loss
    
inputs=[]
targets=[]

for ch1,ch2 in zip(text,text[1:]):
    ix1=stoi[ch1]
    ix2=stoi[ch2]
    inputs.append(ix1)
    targets.append(ix2)


model=bigram_model(vocab_size=vocab_size,embedding_dim=embedding_dim)

initial_embeddings=(model.embedding.weight.detach().cpu().clone())
torch.save(initial_embeddings,"initial_embeddings.pt")

model.to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=0.01)
inputs=torch.tensor(inputs,device=device)
targets=torch.tensor(targets,device=device)

losses=[]
for epoch in range(max_steps):
    perm=torch.randperm(len(inputs))
    xs_shuffled=inputs[perm]
    ys_shuffled=targets[perm]
    losses_this_epoch=[]

    for i in range(0,len(inputs),batch_size):
        xs=xs_shuffled[i:i+batch_size]
        ys=ys_shuffled[i:i+batch_size]

        _,loss=model(xs,ys)
        losses_this_epoch.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    losses.append(sum(losses_this_epoch)/len(losses_this_epoch))

    if epoch%10==0:
        print(f"epoch:{epoch} | loss:{loss.item():.4f}")

final_embeddings=(model.embedding.weight.detach().cpu())
torch.save(final_embeddings,"final_embeddings.pt")
torch.save(losses,"losses.pt")



    





