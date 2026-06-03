import torch
import json

with open("stoi.json","r") as f:
    stoi=json.load(f)
initial_embeddings=torch.load("initial_embeddings.pt")
final_embeddings=torch.load("final_embeddings.pt")

chars=list(stoi.keys())

def cosine_similarity(ch1,ch2,embeddings):
    embed_1=embeddings[stoi[ch1]]
    embed_2=embeddings[stoi[ch2]]
    sim = torch.dot(embed_1,embed_2)/(torch.norm(embed_1)*torch.norm(embed_2))
    return sim.item()


def topk_cosine_similarity_trained(ch1,topk):
    similarities={}
    for ch2 in chars:
        similarities[(ch1,ch2)] = cosine_similarity(ch1,ch2,final_embeddings)
    x=list(similarities.items())
    x.sort(key=lambda x:x[1],reverse=True)
    return x[:topk]

def topk_cosine_similarity_untrained(ch1,topk):
    similarities={}
    for ch2 in chars:
        similarities[(ch1,ch2)] = cosine_similarity(ch1,ch2,initial_embeddings)
    x=list(similarities.items())
    x.sort(key=lambda x:x[1],reverse=True)
    return x[:topk]











