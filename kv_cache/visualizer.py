import time
import torch
import sys
import os

import sample_kvcache

GREEN='\033[92m'
RED='\033[91m'
RESET='\033[0m'

def visualize_stream(prompt,max_tokens=100):
    model=sample_kvcache.model
    enc=sample_kvcache.enc
    device=sample_kvcache.device

    tokens=torch.tensor(enc.encode(prompt),dtype=torch.long,device=device).unsqueeze(0)
    model.clear_kv_cache()

    with torch.no_grad():
        logits,_=model(tokens,start_pos=0)

    import torch.nn.functional as F
    logits=logits[:,-1,:]
    probs=F.softmax(logits,dim=-1)
    next_token=torch.multinomial(probs,num_samples=1)

    for _ in range(max_tokens):
        os.system('clear')
        print(f"\n---Visualizing KV Cache Hits/Misses---\n")
        print(f"Legend:{GREEN} ||| Cache Hit (Saved){RESET} vs {RED}||| Cache miss (Computed)")
        green_text=enc.decode(tokens[0].tolist())
        red_text=enc.decode(next_token[0].tolist())
        print(f"{GREEN}{green_text}{RED}{red_text}{RESET}",end="",flush=True)
        time.sleep(0.05)
        current_pos=tokens.size(1)-1
        with torch.no_grad():
            logits,_=model(next_token,start_pos=current_pos)
        probs=F.softmax(logits[:,-1,:],dim=1)
        next_token_id=torch.multinomial(probs,num_samples=1)
        tokens=torch.cat((tokens,next_token_id),dim=1)
        next_token=next_token_id
    print("\n\n---Done---")

if __name__=="__main__":
    user_prompt="Artificial Intelligence"
    visualize_stream(user_prompt)