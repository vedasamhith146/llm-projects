import torch
import matplotlib.pyplot as plt
import tiktoken
import math

enc=tiktoken.get_encoding('gpt2')
text="The cat sat on the mat"
input=enc.encode(text)
input_tensor=torch.tensor((input),dtype=torch.long)

torch.manual_seed(42)
tok_embd_table=torch.randn((50257,2))
Wq=torch.randn((2,2))
Wk=torch.randn((2,2))

tok_embd=tok_embd_table[input_tensor]
Q=tok_embd @ Wq
K=tok_embd @ Wk
attn_1= torch.tril((Q @ K.transpose(0,1))/math.sqrt(2))

new_input = [3797,464,2603,319,3332,262]
new_input_tensor=torch.tensor((new_input))
new_tok_embd=tok_embd_table[new_input_tensor]
Q_new = new_tok_embd @ Wq
K_new= new_tok_embd @ Wk
attn_2= torch.tril((Q_new @ K_new.transpose(0,1))/math.sqrt(2))

vmin=min(attn_1.min(),attn_2.min())
vmax=max(attn_1.max(),attn_2.max())

plt.imshow(attn_1,cmap="viridis",vmin=vmin,vmax=vmax)
plt.title("Attention map for the tokens in order(0,1,2,3,4,5)")
plt.colorbar()

plt.imshow(attn_2,cmap="viridis",vmin=vmin,vmax=vmax)
plt.title("Attention map for the tokens in order(1,0,5,3,2,4)")
plt.colorbar()

torch.set_printoptions(precision=4,sci_mode=False)
print(attn_1)
print(attn_2)
