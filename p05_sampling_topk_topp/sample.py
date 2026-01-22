import torch
from torch.nn import functional as F
from train_gpt2 import GPT2, GPT2Config, device 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


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

def generate(model,prompt, max_tokens,top_k,top_p,temp):
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    tokens_entropy_history=[]
    generated_text=prompt

    for _ in range(max_tokens):
        with torch.no_grad():
            logits, _ = model(tokens)
            logits = logits[:, -1, :] 

            if temp==0:
                next_token=torch.argmax(logits,dim=-1).unsqueeze(0).unsqueeze(0)
                tokens_entropy_history.append(0.0)
            else:
                logits=logits/temp
                if top_k>0:
                    #v, _ = torch.topk(logits, top_k)
                    v,_=torch.sort(logits,descending=True)
                    v=v[:,:top_k]
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)

                if top_p>0:
                    sorted_probs,sorted_indices=torch.sort(probs,descending=True)
                    cumm_probs=torch.cumsum(sorted_probs,dim=-1)
                    sorted_indices_to_remove=cumm_probs>top_p
                    sorted_indices_to_remove[:,1:]=sorted_indices_to_remove[:,:-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    sorted_indices_remove=sorted_indices[sorted_indices_to_remove]
                    logits[0,sorted_indices_remove]=-float('Inf')
                    probs=F.softmax(logits,dim=-1)

                current_entropy = -(probs * torch.log(probs + 1e-9)).sum().item()
                tokens_entropy_history.append(current_entropy)

                next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat((tokens, next_token), dim=1)
            word=enc.decode(next_token[0].tolist())
            generated_text+=word

    return generated_text,tokens_entropy_history

#print("Model loaded! Type your prompt below (or type 'exit' to quit).")

#while True:
    #user_input = input("\nPrompt: ")
    
    #if user_input.lower() in ['exit', 'quit']:
        #break
        
    #if user_input.strip() == "":
        #continue

    #generate(model,user_input, 100,None,0.9,1.5)
