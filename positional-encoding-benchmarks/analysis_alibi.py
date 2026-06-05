import torch
import matplotlib.pyplot as plt
import math
from alibi import ALiBi_config,ALiBi_model,Attention_Mask

device= "cuda" if torch.cuda.is_available() else "cpu"

ALiBi_model=ALiBi_model(ALiBi_config()).to(device)
val_tokens=torch.load("val_tokens.pt")

batch_size=1

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
            with torch.autocast(device_type=device,dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses.append(loss.item())
        return math.exp(sum(losses) / len(losses))


ALiBi_model_dict=torch.load("ALiBi_model.pt")

for key in list(ALiBi_model_dict.keys()):
    if key.startswith('_orig_mod.'):
        ALiBi_model_dict[key[len('_orig_mod.'):]]=ALiBi_model_dict.pop(key)

ALiBi_model.load_state_dict(ALiBi_model_dict,strict=False)

block_size_others=[32,64,128,256,512,1024,2048,4096]

ALiBi_model.eval()

perplexity_ALiBi=[]



for block_size in block_size_others:
    per=estimate_perplexity("val",ALiBi_model,block_size)
    perplexity_ALiBi.append(per)

print("ALiBi:", perplexity_ALiBi)


plt.plot(block_size_others, perplexity_ALiBi, label="ALiBi")


plt.xlabel("Context Length")
plt.ylabel("Perplexity")
plt.legend()
plt.savefig("context_length_alibi vs perplexity.png",dpi=300)