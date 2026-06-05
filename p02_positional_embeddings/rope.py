import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F

torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high') 

batch_size=64
learning_rate=3e-4
max_steps=10000


class RotationalPositionalEmbedding(nn.Module):
    def __init__(self,d_model,max_len=1024):
        super().__init__()
        position=torch.arange(max_len).unsqueeze(1)
        thetha_term=torch.exp(torch.repeat_interleave(torch.arange(0,d_model,2),repeats=2)*(-torch.log(torch.tensor(10000))/d_model))
        cos_terms=torch.cos(position*thetha_term)
        sin_terms=torch.sin(position*thetha_term)

        self.register_buffer("cos_term",cos_terms)
        self.register_buffer("sin_term",sin_terms)

    def reverse_tensor(self,input_tensor):
        B,N,T,D=input_tensor.size()
        modified_tensor=input_tensor.view(B,N,T,D//2,2)
        flipped_tensor=torch.flip(modified_tensor,dims=[-1])
        reversed_tensor= (flipped_tensor*torch.tensor((-1,1),device=input_tensor.device)).view(B,N,T,D)
        return reversed_tensor
    
    def forward(self,input_tensor):
        T=input_tensor.size(-2)
        cos_term=self.cos_term[:T]
        sin_term=self.sin_term[:T]
        reversed_tensor=self.reverse_tensor(input_tensor)
        final_tensor=(input_tensor*cos_term)+(reversed_tensor*sin_term)
        return final_tensor



@dataclass
class RoPE_config:
    block_size: int=128
    vocab_size: int=50257
    n_layer: int=4
    n_head: int=4
    n_embd: int=256


class RoPE_model(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd)
        ))
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.transformer.wte.weight=self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self,idx,targets=None):
        B, T = idx.size()
        x= self.transformer.wte(idx) 
        for block in self.transformer.h: x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.rope=RotationalPositionalEmbedding(config.n_embd//config.n_head)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q=self.rope(q)
        k=self.rope(k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))
    
if __name__=="__main__":
    config=RoPE_config()
    model=RoPE_model(config)
    model.to(device)
    model = torch.compile(model) 
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate,betas=(0.9, 0.95),weight_decay=0.1)

    train_tokens=torch.load("train_tokens.pt")
    val_tokens=torch.load("val_tokens.pt")

    def get_batch(split):
        data= train_tokens if split =="train" else val_tokens
        ix=torch.randint(len(data)-config.block_size-1,(batch_size,))
        x=torch.stack([data[i:i+config.block_size] for i in ix])
        y=torch.stack([data[i+1:i+config.block_size+1] for i in ix])
        return x,y   

    @torch.no_grad()

    def estimate_loss(split, eval_iters=50):
        model.eval()
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device,dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    for step in range(max_steps):
        x,y=get_batch("train")
        x,y=x.to(device), y.to(device)
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            logits,loss=model(x,y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step%500==0:
            train_loss=estimate_loss("train")
            val_loss=estimate_loss("val")
            print(f"step {step} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

        
    torch.save(model.state_dict(),"RoPE_model.pt")
    print("Saved sinusoidal_model.pt")
