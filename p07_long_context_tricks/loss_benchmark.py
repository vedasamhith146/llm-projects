import torch
import torch.nn.functional as F
import math
import model
import model_sliding
import tiktoken
device='mps' if torch.backends.mps.is_available() else 'cpu'

enc=tiktoken.get_encoding('gpt2')

curr_config=model_sliding.GPT2Config
curr_model=model_sliding.GPT2(curr_config(vocab_size=50304))

with open('text.txt','r') as f:
    text=f.read()
tokens=enc.encode(text)
tokens=torch.tensor(tokens,dtype=torch.long).to(device)

def evaluate_long_document(model,tokens,ctx_len=1024,stride=512):
    curr_model.eval()
    print(f"Total tokens in the document : {len(tokens)}")
    nlls=[]
    total_valid_tokens=0
    for i in range(0,tokens.size(0)-1,stride):
        end_loc=min(i+ctx_len,tokens.size(0)-1)
        input_ids=tokens[i:end_loc].unsqueeze(0)
        target_ids=tokens[i+1:end_loc+1]
        with torch.no_grad():
            _,loss=curr_model(input_ids,target_ids)
        nlls.append(loss)
        if end_loc==tokens.size(0)-1:
            break
    avg_loss=torch.stack(nlls).mean()
    perplexity=torch.exp(avg_loss)

    return avg_loss.item(),perplexity.item()








