import torch
from torch.nn import functional as F
from model_kvcache import GPT2, GPT2Config, device 
import tiktoken
enc=tiktoken.get_encoding('gpt2')

if torch.backends.mps.is_available():
    device=torch.device("mps")
    print(f"Using Device: MPS")
else:
    device=torch.device("cpu")
    print(f"Using Device:CPU")


checkpoint_path = "gpt2_step_11500.pt" 

config = GPT2Config(vocab_size=50304)
model = GPT2(config)

state_dict = torch.load(checkpoint_path, map_location=device)
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.to(device)
model.eval() 

def generate(prompt, max_tokens=100,top_k=50,top_p=0.9,temp=1.0):
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    print(f"\n--- Generating from prompt: '{prompt}' ---\n")

    with torch.no_grad():
        logits,_=model(tokens)
    
    logits=logits[:,-1,:]/temp
    if top_k is not None:
        v,_=torch.sort(logits,descending=True)
        v=v[:,:top_k]
        logits[logits < v[:, [-1]]] = -float('Inf')
    probs = F.softmax(logits, dim=-1)

    if top_p is not None:
        sorted_probs,sorted_indices=torch.sort(probs,descending=True)
        cumm_probs=torch.cumsum(sorted_probs,dim=-1)
        sorted_indices_to_remove=cumm_probs>top_p
        sorted_indices_to_remove[:,1:]=sorted_indices_to_remove[:,:-1].clone()
        sorted_indices_to_remove[:, 0] = False
        sorted_indices_remove=sorted_indices[sorted_indices_to_remove]
        logits[0,sorted_indices_remove]=-float('Inf')
        probs=F.softmax(logits,dim=-1)
        
    next_token = torch.multinomial(probs, num_samples=1)
    tokens = torch.cat((tokens, next_token), dim=1)
    text=enc.decode(tokens[0].tolist())
    prev_text = enc.decode(tokens[0, :-1].tolist()) 
    print(text[len(prev_text):], end='', flush=True)
    prev_text = text

    for _ in range(max_tokens-1):
        current_pos=tokens.size(1)-1
        with torch.no_grad():
            logits, _ = model(next_token,start_pos=current_pos)
            logits = logits[:, -1, :]/temp

            if top_k is not None:
                v,_=torch.sort(logits,descending=True)
                v=v[:,:top_k]
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)

            if top_p is not None:
                sorted_probs,sorted_indices=torch.sort(probs,descending=True)
                cumm_probs=torch.cumsum(sorted_probs,dim=-1)
                sorted_indices_to_remove=cumm_probs>top_p
                sorted_indices_to_remove[:,1:]=sorted_indices_to_remove[:,:-1].clone()
                sorted_indices_to_remove[:, 0] = False
                sorted_indices_remove=sorted_indices[sorted_indices_to_remove]
                logits[0,sorted_indices_remove]=-float('Inf')
                probs=F.softmax(logits,dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
            text=enc.decode(tokens[0].tolist())
            #print(text)
            print(text[len(prev_text):], end='', flush=True)
            prev_text=text
    print("\n\n--- End ---")

if __name__ == "__main__":

    print("Model loaded! Type your prompt below (or type 'exit' to quit).")
    while True:
        user_input = input("\nPrompt: ")
        
        if user_input.lower() in ['exit', 'quit']:
            break
            
        if user_input.strip() == "":
            continue
        
        model.clear_kv_cache()
        generate(user_input, max_tokens=100)
