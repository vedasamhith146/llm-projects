import torch
import torch.nn.functional as F
from model_kvcache import GPT2, GPT2Config
import tiktoken

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

checkpoint = "gpt2_instruction_tuned_transposed.pt"
state_dict = torch.load(checkpoint, map_location=device)

# Handle compiled model weights if necessary
for k in list(state_dict.keys()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)

model = GPT2(GPT2Config(vocab_size=50257))
model.to(device)
model.load_state_dict(state_dict)

enc = tiktoken.get_encoding('gpt2')

# GPT-2 EOS token ID in tiktoken is 50256
EOS_TOKEN_ID = 50256 
REPETITION_PENALTY = 1.2 # Prevents "eucalyptus, eucalyptus" loops

def next_token_generator(logits, temp, top_k, top_p):
    if temp == 0:
        next_token = logits.argmax(dim=-1, keepdim=True)
        return next_token
    
    logits = logits / temp

    if top_k is not None:
        _, topk_indices = torch.topk(logits, k=top_k)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[0, topk_indices[0]] = False
        logits[mask] = -float('inf')

    if top_p is not None:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True) 
        cumm_prob = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumm_prob > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        gather_indices = sorted_indices[~sorted_indices_to_remove].unsqueeze(0)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[0, gather_indices[0]] = False
        logits[mask] = -float('inf')

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

def generate_text(prompt, max_new_tokens, temp=0.5, top_k=50, top_p=0.95):
    model.clear_kv_cache()
    model.eval()

    initial_tokens = enc.encode(prompt)
    tokens = torch.tensor(initial_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Track generated tokens to apply repetition penalty
    generated_tokens = set(initial_tokens) 

    print("Output Generated from the model:")
    print(prompt, end="", flush=True)

    stop_reason = "max_new_tokens"
    generated_count = 0

    with torch.no_grad():
        if tokens.size(1) > 0:
            logits, _ = model(tokens, start_pos=0)
            logits = logits[:, -1, :]
        else:
            logits = None

        for _ in range(max_new_tokens):
            if logits is None:
                next_token = torch.tensor([[enc.encode(" ")[0]]], dtype=torch.long, device=device)
            else:
                # --- FIX 1: APPLY REPETITION PENALTY ---
                if REPETITION_PENALTY > 1.0:
                    for token_id in generated_tokens:
                        if logits[0, token_id] > 0:
                            logits[0, token_id] /= REPETITION_PENALTY
                        else:
                            logits[0, token_id] *= REPETITION_PENALTY

                next_token = next_token_generator(logits, temp=temp, top_k=top_k, top_p=top_p)

            # --- FIX 2: STOP AT EOS TOKEN ---
            if next_token.item() == EOS_TOKEN_ID:
                stop_reason="eos_token"
                break

            # Add new token to our tracking set
            generated_count+=1
            generated_tokens.add(next_token.item())

            tokens = torch.cat((tokens, next_token), dim=-1)
            token_text = enc.decode(next_token.squeeze(0).tolist())
            print(token_text, end="", flush=True)

            logits, _ = model(next_token, start_pos=tokens.size(1) - 1)
            logits = logits[:, -1, :]

    print("\n" + "-" * 50)

    if stop_reason == "eos_token":
        print(f"Generation stopped because the model generated the EOS token ({EOS_TOKEN_ID}).")
    else:
        print(f"Generation stopped because max_new_tokens ({max_new_tokens}) was reached.")

    print(f"Generated {generated_count} new tokens.")

if __name__ == "__main__":
    try:
        while True:
            user_input = input("Enter prompt (or type exit to quit): ")
            if user_input.strip().lower() == "exit":
                print("Exiting.")
                break
            if user_input.strip() == "":
                print("Please enter a non-empty prompt or type exit to quit.")
                continue
            
            # Format the prompt exactly like the training data
            formatted_prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
            
            generate_text(formatted_prompt, max_new_tokens=150)
    except KeyboardInterrupt:
        print("\nExiting.")