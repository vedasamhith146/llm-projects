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

def Attention_Mask(block_size,n_heads):
    t=2**(-8/n_heads)
    exponents= t**torch.arange(1,n_heads+1).view(n_heads,1)
    temp_1=torch.arange(block_size).unsqueeze(0).unsqueeze(0).expand(n_heads,-1,-1)
    temp_2=torch.arange(block_size).unsqueeze(1).unsqueeze(0).expand(n_heads,-1,-1)
    t_by_t=temp_1 - temp_2
    t_by_t=torch.tril(t_by_t)
    final_mask= t_by_t*exponents.unsqueeze(1)
    return final_mask.unsqueeze(0)


@dataclass
class ALiBi_config:
    block_size: int=128
    vocab_size: int=50257
    n_layer: int=4
    n_head: int=4
    n_embd: int=256


class ALiBi_model(nn.Module):
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
        t = 2 ** (-8 / config.n_head)
        slopes = t ** torch.arange(1,config.n_head + 1,dtype=torch.float32)
        self.register_buffer("slopes", slopes)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        idx = torch.arange(T, device=x.device)
        distance = torch.tril(idx[None, :] - idx[:, None]).float()
        mask = (distance.unsqueeze(0)* self.slopes.view(self.n_head, 1, 1)).unsqueeze(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,attn_mask=mask)
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
    config=ALiBi_config()
    model=ALiBi_model(config)
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

        
    torch.save(model.state_dict(),"ALiBi_model.pt")
    print("Saved ALiBi_model.pt")









