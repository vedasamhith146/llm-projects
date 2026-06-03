import matplotlib.pyplot as plt
import torch
import json

with open("stoi.json","r") as f:
    stoi=json.load(f)
initial_embeddings=torch.load("initial_embeddings.pt")
final_embeddings=torch.load("final_embeddings.pt")

plt.figure(figsize=(8,8))

fig,axes=plt.subplots(1,2,figsize=(14,7))

for ch,idx in stoi.items():
    x=initial_embeddings[idx,0].item()
    y=initial_embeddings[idx,1].item()

    axes[0].scatter(x,y)
    axes[0].annotate(ch,(x,y))

axes[0].set_title("initial embeddings")

for ch,idx in stoi.items():
    x=final_embeddings[idx,0].item()
    y=final_embeddings[idx,1].item()

    axes[1].scatter(x,y)
    axes[1].annotate(ch,(x,y))

axes[1].set_title("final embeddings")

plt.tight_layout()
plt.show()