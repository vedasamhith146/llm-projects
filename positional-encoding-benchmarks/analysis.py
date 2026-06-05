import torch
import matplotlib.pyplot as plt
import math
from learned import learned_config,learned_model
from sinusoidal import sinusoidal_config,sinusoidal_model,SinusoidalPositionalEmbedding
from rope import RoPE_config,RoPE_model,RotationalPositionalEmbedding
from alibi import ALiBi_config,ALiBi_model,Attention_Mask
from without_position import without_position_config,without_position_model

device= "cuda" if torch.cuda.is_available() else "cpu"

learned_model=learned_model(learned_config()).to(device)
sinusoidal_model=sinusoidal_model(sinusoidal_config()).to(device)
RoPE_model=RoPE_model(RoPE_config()).to(device)
ALiBi_model=ALiBi_model(ALiBi_config()).to(device)
without_position_model=without_position_model(without_position_config()).to(device)

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
            with torch.autocast(device_type=device,dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses.append(loss.item())
        return math.exp(sum(losses) / len(losses))

val_tokens=torch.load("val_tokens.pt")
learned_model_dict=torch.load("learned_model.pt")
sinusoidal_model_dict=torch.load("sinusoidal_model.pt")
RoPE_model_dict=torch.load("RoPE_model.pt")
ALiBi_model_dict=torch.load("ALiBi_model.pt")
without_position_model_dict=torch.load("without_position_model.pt")

for key in list(learned_model_dict.keys()):
    if key.startswith('_orig_mod.'):
        learned_model_dict[key[len('_orig_mod.'):]]=learned_model_dict.pop(key)

for key in list(sinusoidal_model_dict.keys()):
    if key.startswith('_orig_mod.'):
        sinusoidal_model_dict[key[len('_orig_mod.'):]]=sinusoidal_model_dict.pop(key)

for key in list(RoPE_model_dict.keys()):
    if key.startswith('_orig_mod.'):
        RoPE_model_dict[key[len('_orig_mod.'):]]=RoPE_model_dict.pop(key)

for key in list(ALiBi_model_dict.keys()):
    if key.startswith('_orig_mod.'):
        ALiBi_model_dict[key[len('_orig_mod.'):]]=ALiBi_model_dict.pop(key)

for key in list(without_position_model_dict.keys()):
    if key.startswith('_orig_mod.'):
        without_position_model_dict[key[len('_orig_mod.'):]]=without_position_model_dict.pop(key)

learned_model.load_state_dict(learned_model_dict)
sinusoidal_model.load_state_dict(sinusoidal_model_dict)
RoPE_model.load_state_dict(RoPE_model_dict)
ALiBi_model.load_state_dict(ALiBi_model_dict,strict=False)
without_position_model.load_state_dict(without_position_model_dict)

block_size_learned=[32,64,128]
block_size_others=[32,64,128,256,512,1024]

learned_model.eval()
sinusoidal_model.eval()
RoPE_model.eval()
ALiBi_model.eval()
without_position_model.eval()

perplexity_learned=[]
perplexity_sinusoidal=[]
perplexity_RoPE=[]
perplexity_ALiBi=[]
perplexity_without_position=[]

for block_size in block_size_learned:
    per=estimate_perplexity("val",learned_model,block_size)
    perplexity_learned.append(per)

for block_size in block_size_others:
    per=estimate_perplexity("val",sinusoidal_model,block_size)
    perplexity_sinusoidal.append(per)

for block_size in block_size_others:
    per=estimate_perplexity("val",RoPE_model,block_size)
    perplexity_RoPE.append(per)

for block_size in block_size_others:
    per=estimate_perplexity("val",ALiBi_model,block_size)
    perplexity_ALiBi.append(per)

for block_size in block_size_others:
    per=estimate_perplexity("val",without_position_model,block_size)
    perplexity_without_position.append(per)



print("Learned:", perplexity_learned)
print("Sinusoidal:", perplexity_sinusoidal)
print("RoPE:", perplexity_RoPE)
print("ALiBi:", perplexity_ALiBi)
print("Without position:",perplexity_without_position)

plt.plot(block_size_learned,perplexity_learned,label="Learned")
plt.plot(block_size_others, perplexity_sinusoidal, label="Sinusoidal")
plt.plot(block_size_others, perplexity_RoPE, label="RoPE")
plt.plot(block_size_others, perplexity_ALiBi, label="ALiBi")
plt.plot(block_size_others,perplexity_without_position,label="Without position")

plt.yscale("log")

plt.xlabel("Context Length")
plt.ylabel("Perplexity(Log scale)")
plt.legend()
plt.savefig("context_length vs perplexity_log_2.png",dpi=300)











    