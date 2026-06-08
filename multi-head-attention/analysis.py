import torch
import matplotlib.pyplot as plt
import math
from one_head_model import RoPE_config,one_head_model

device="cuda" if torch.cuda.is_available() else "cpu"

one_head=one_head_model(RoPE_config(n_head=1)).to(device)
two_head=one_head_model(RoPE_config(n_head=2)).to(device)
four_head=one_head_model(RoPE_config(n_head=4)).to(device)
eight_head=one_head_model(RoPE_config(n_head=8)).to(device)
sixteen_head=one_head_model(RoPE_config(n_head=16)).to(device)
thirtytwo_head=one_head_model(RoPE_config(n_head=32)).to(device)

batch_size=16


def get_batch(split,block_size):
        data= train_tokens if split =="train" else val_tokens
        ix=torch.randint(len(data)-block_size-1,(batch_size,))
        x=torch.stack([data[i:i+block_size] for i in ix])
        y=torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x,y  

def estimate_perplexity(split,model,block_size,eval_iters=100):
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split,block_size)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
        return math.exp(sum(losses) / len(losses))

train_tokens=torch.load("train_tokens.pt")
val_tokens=torch.load("val_tokens.pt")
one_head_model_dict=torch.load("one_head_model.pt")
two_head_model_dict=torch.load("two_head_model.pt")
four_head_model_dict=torch.load("four_head_model.pt")
eight_head_model_dict=torch.load("eight_head_model.pt")
sixteen_head_model_dict=torch.load("sixteen_head_model.pt")
thirtytwo_head_model_dict=torch.load("thirtytwo_head_model.pt")

for k in list(one_head_model_dict.keys()):
    if k.startswith('_orig_mod.'):
          one_head_model_dict[k[len('_orig_mod.'):]]=one_head_model_dict.pop(k)

for k in list(two_head_model_dict.keys()):
    if k.startswith('_orig_mod.'):
          two_head_model_dict[k[len('_orig_mod.'):]]=two_head_model_dict.pop(k)

for k in list(four_head_model_dict.keys()):
    if k.startswith('_orig_mod.'):
          four_head_model_dict[k[len('_orig_mod.'):]]=four_head_model_dict.pop(k)

for k in list(eight_head_model_dict.keys()):
    if k.startswith('_orig_mod.'):
          eight_head_model_dict[k[len('_orig_mod.'):]]=eight_head_model_dict.pop(k)

for k in list(sixteen_head_model_dict.keys()):
    if k.startswith('_orig_mod.'):
          sixteen_head_model_dict[k[len('_orig_mod.'):]]=sixteen_head_model_dict.pop(k)

for k in list(thirtytwo_head_model_dict.keys()):
    if k.startswith('_orig_mod.'):
          thirtytwo_head_model_dict[k[len('_orig_mod.'):]]=thirtytwo_head_model_dict.pop(k)

one_head.load_state_dict(one_head_model_dict)
two_head.load_state_dict(two_head_model_dict)
four_head.load_state_dict(four_head_model_dict)
eight_head.load_state_dict(eight_head_model_dict)
sixteen_head.load_state_dict(sixteen_head_model_dict)
thirtytwo_head.load_state_dict(thirtytwo_head_model_dict)

losses=[estimate_perplexity('val',one_head,128),estimate_perplexity('val',two_head,128),estimate_perplexity('val',four_head,128),estimate_perplexity('val',eight_head,128),estimate_perplexity('val',sixteen_head,128),estimate_perplexity('val',thirtytwo_head,128)]

heads=[1,2,4,8,16,32]

plt.figure(figsize=(8,5))
plt.plot(heads,losses,marker='o')
plt.xlabel("Number of heads")
plt.ylabel("Validation perplexity")
plt.title("Perplexity vs Number of attention heads")
plt.grid(True)
plt.savefig("Perplexity_vs_Number_of_attention_heads.png",dpi=300)

plt.figure(figsize=(8,5))
plt.plot(heads, losses, marker='o')
plt.xscale("log", base=2)
plt.xlabel("Number of Heads")
plt.ylabel("Validation Perplexity")
plt.title("Perplexity vs Number of Heads(log scale)")
plt.grid(True)
plt.savefig("Perplexity_vs_Log_of_attention_heads.png",dpi=300)

head_dims=[256//h for h in heads]
plt.figure(figsize=(8,5))
plt.plot(head_dims, losses, marker='o')
plt.xlabel("Head Dimension")
plt.ylabel("Validation Perplexity")
plt.title("Perplexity vs Head Dimension")
plt.grid(True)
plt.savefig("Perplexity_vs_head_dimension.png",dpi=300)

plt.figure(figsize=(8,5))
plt.scatter(heads, losses)
for h,p in zip(heads, losses):
    plt.annotate(f"{h}",(h,p))

plt.xscale("log", base=2)
plt.xlabel("Number of Heads")
plt.ylabel("Validation Perplexity")
plt.title("Perplexity of TinyStories Models")
plt.grid(True)
plt.savefig("Perplexity_of_Tinystories_Models")