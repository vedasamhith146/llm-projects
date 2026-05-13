import torch
import torch.nn.functional as F
from train_gpt2 import GPT2,GPT2Config
import tiktoken

device='mps' if torch.backends.mps.is_available() else 'cpu'

checkpoint="gpt2_step_11500.pt"
state_dict=torch.load(checkpoint,map_location=device)

for k in list(state_dict.keys()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]]=state_dict.pop(k)

model=GPT2(GPT2Config(vocab_size=50304))
model.to(device)

model.load_state_dict(state_dict)

prompt="The future of artificial intelligence is"

enc=tiktoken.get_encoding('gpt2')

def next_token_generator(logits,temp,top_k,top_p):
    if temp==0:
        next_token=logits.argmax(dim=-1,keepdim=True)
        return next_token
    logits=logits/temp
    probs=F.softmax(logits,dim=-1)  #(1,50304)

    if top_k is not None:
        logits,topk_indices = torch.topk(logits,k=top_k)
        probs=F.softmax(logits,dim=-1) #(1,top_k)

    if top_p is not None:
        sorted_probs,sorted_indices=torch.sort(probs,descending=True) 
        cumm_prob=torch.cumsum(sorted_probs,dim=-1)
        sorted_indices_to_remove=cumm_prob > top_p
        sorted_indices_to_remove[:,1:]=sorted_indices_to_remove[:,:-1].clone()
        sorted_indices_to_remove[:,0]=False
        sorted_probs[sorted_indices_to_remove]=0
        sorted_probs_sum=torch.sum(sorted_probs,dim=-1)
        probs=sorted_probs/sorted_probs_sum
    
    if top_k is None and top_p is None:
        next_token=torch.multinomial(probs,num_samples=1)
        return next_token
    elif top_k is not None and top_p is None:
        next_index=torch.multinomial(probs,num_samples=1)
        next_token=torch.gather(topk_indices,dim=1,index=next_index)
        return next_token
    elif top_k is None and top_p is not None:
        next_index=torch.multinomial(probs,num_samples=1)
        next_token=torch.gather(sorted_indices,dim=1,index=next_index)
        return next_token
    else:
        next_index_1=torch.multinomial(probs,num_samples=1)
        next_index_2=torch.gather(sorted_indices,dim=1,index=next_index_1)
        next_token=torch.gather(topk_indices,dim=1,index=next_index_2)
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

prompt="The Industrial Revolution happened in"

generate_text(prompt,max_new_tokens=50)

    

        








