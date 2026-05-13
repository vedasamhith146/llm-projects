import torch
import torch.nn.functional as F
from train_gpt2 import GPT2,GPT2Config
import tiktoken
import time

device='mps' if torch.backends.mps.is_available() else 'cpu'

checkpoint="gpt2_step_11500.pt"
state_dict=torch.load(checkpoint,map_location=device)

for k in list(state_dict.keys()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]]=state_dict.pop(k)

model=GPT2(GPT2Config(vocab_size=50304))
model.to(device)

model.load_state_dict(state_dict)

enc=tiktoken.get_encoding('gpt2')


def next_token_generator(logits,temp,top_k,top_p):
    if temp==0:
        next_token=logits.argmax(dim=-1,keepdim=True)
        return next_token
    logits=logits/temp

    if top_k is not None:
        _,topk_indices = torch.topk(logits,k=top_k)
        mask=torch.ones_like(logits,dtype=torch.bool)
        mask[0,topk_indices[0]]=False
        logits[mask]=-float('inf')


    if top_p is not None:
        probs=F.softmax(logits,dim=-1)
        sorted_probs,sorted_indices=torch.sort(probs,descending=True) 
        cumm_prob=torch.cumsum(sorted_probs,dim=-1)
        sorted_indices_to_remove=cumm_prob > top_p
        sorted_indices_to_remove[:,1:]=sorted_indices_to_remove[:,:-1].clone()
        sorted_indices_to_remove[:,0]=False
        gather_indices=sorted_indices[~sorted_indices_to_remove].unsqueeze(0)
        mask=torch.ones_like(logits,dtype=torch.bool)
        mask[0,gather_indices[0]]=False
        logits[mask]=-float('inf')

    probs=F.softmax(logits,dim=-1)
    next_token=torch.multinomial(probs,num_samples=1)
    return next_token
    

def generate_text(prompt,max_new_tokens,temp=0.7,top_k=50,top_p=0.90):

    initial_tokens=enc.encode(prompt)
    tokens=torch.tensor(initial_tokens,dtype=torch.long,device=device).unsqueeze(0) #shape (1,T)
    

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits,_=model(tokens)
            logits=logits[:,-1,:]   
            next_token=next_token_generator(logits,temp=temp,top_k=top_k,top_p=top_p)                        #shape(1,50304)
            tokens=torch.cat((tokens,next_token),dim=-1)
    
    all_tokens=tokens.squeeze(0).tolist()
    generated_tokens=all_tokens[len(initial_tokens):]
    generated_text=enc.decode(generated_tokens)


    print("Output Generated from the model:")
    print(f"{generated_text}")

if __name__=="__main__":
    prompt="The Industrial Revolution happened in"

    generate_text(prompt,max_new_tokens=50)




    

        








