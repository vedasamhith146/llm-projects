import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
import sys


device='cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

batch_size=4
learning_rate=3e-4
max_steps=200


@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

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

class mini_former(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.miniformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            #rmsnorm_f=RMSNorm(config)
        ))
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.lm_head.weight=self.miniformer.wte.weight
        self.per_layer_activations={}
        self.per_layer_activations_mean={}
        self.per_layer_activations_std={}
        self.per_layer_max={}
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
        self.per_layer_activations={}
        self.per_layer_activations_mean={}
        self.per_layer_activations_std={}
        i=0
        x=self.miniformer.wte(idx)
        for block in self.miniformer.h:
            x=block(x)
            self.per_layer_activations[f"h.{i}"]=torch.sqrt(torch.sum(x**2)).item()
            self.per_layer_activations_mean[f"h.{i}"]=torch.mean(x).item()
            self.per_layer_activations_std[f"h.{i}"]=math.sqrt(torch.var(x).item())
            self.per_layer_max[f"h.{i}"]=torch.max((torch.abs(x))).item()
            i+=1
        #x=self.miniformer.rmsnorm_f(x)
        logits=self.lm_head(x)
        loss=None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        #self.rmsnorm_1=RMSNorm(config)
        self.attn=CausalSelfAttention(config)
        #self.rmsnorm_2=RMSNorm(config)
        self.mlp=SwiGLU_MLP(config)

    def forward(self,x):
        #x=x+self.attn(self.rmsnorm_1(x))
        #x=x+self.mlp(self.rmsnorm_2(x))
        x=x+self.attn(x)
        x=x+self.mlp(x)
        return x
    
class RMSNorm(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.alpha=nn.Parameter(torch.ones((config.n_embd)))
        self.eps=1e-5

    def forward(self,x):
        rms= ((torch.sum((x**2),dim=-1))/x.size(-1)).unsqueeze(-1)
        output = self.alpha*(x/torch.sqrt(rms+self.eps)) 
        return output
    
class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj=nn.Linear(config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1.0
        self.n_embd=config.n_embd
        self.n_head=config.n_head
        self.rope=RotationalPositionalEmbedding(config.n_embd//config.n_head)
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)))

    def forward(self,x):
        B,T,C=x.size()
        q,k,v=self.c_attn(x).split(self.n_embd,dim=2)
        q=q.view(B,T,self.n_head,self.n_embd//self.n_head).transpose(1,2)
        k=k.view(B,T,self.n_head,self.n_embd//self.n_head).transpose(1,2)
        v=v.view(B,T,self.n_head,self.n_embd//self.n_head).transpose(1,2)
        q=self.rope(q)
        k=self.rope(k)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.n_embd//self.n_head)
        att = att.masked_fill(self.bias[:T,:T]==0,float('-inf'))
        att=F.softmax(att,dim=-1)
        y= att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
    
class SwiGLU_MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.W=nn.Linear(config.n_embd,4*config.n_embd)
        self.V=nn.Linear(config.n_embd,4*config.n_embd)
        self.P=nn.Linear(4*config.n_embd,config.n_embd)
        self.P.NANOGPT_SCALE_INIT=1.0
    def forward(self,x):
        input_1=self.W(x)
        input_2=self.V(x)
        swish=F.silu(input_1)
        output=self.P(swish * input_2)
        return output
    
if __name__=="__main__":
    config=Config()
    model=mini_former(config)
    model.to(device)
    model=torch.compile(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,betas=(0.9,0.95),weight_decay=0.1)
    train_tokens=torch.load("train_tokens.pt")
    val_tokens=torch.load("val_tokens.pt")

    def get_batch(split):
        data= train_tokens if split =="train" else val_tokens
        ix=torch.randint(len(data)-config.block_size-1,(batch_size,))
        x=torch.stack([data[i:i+config.block_size] for i in ix])
        y=torch.stack([data[i+1:i+config.block_size+1] for i in ix])
        return x,y   
    
    def estimate_loss(split, eval_iters=50):
        model.eval()
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split)
            x, y = x.to(device), y.to(device)
            #with torch.autocast(device_type=device,dtype=torch.bfloat16):
            _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)
    
    losses=[]
    gradient_norm=[]
    per_layer_activation_norm=[]
    per_layer_activation_mean=[]
    per_layer_activation_std=[]
    per_layer_gradient_norm=[]
    per_layer_max_value=[]
    for step in range(max_steps):
        x,y=get_batch("train")
        x,y=x.to(device), y.to(device)
        #with torch.autocast(device_type=device,dtype=torch.bfloat16):
        logits,loss=model(x,y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step%10==0:
            train_loss=estimate_loss("train")
            params_grad_norm=[(torch.sum(param.grad**2)).item() for param in list(model.parameters())]
            total_grad_norm=math.sqrt(sum(params_grad_norm))
            losses.append(train_loss)
            gradient_norm.append(total_grad_norm)
            per_layer_activation_norm.append(model.per_layer_activations.copy())
            per_layer_activation_mean.append(model.per_layer_activations_mean.copy())
            per_layer_activation_std.append(model.per_layer_activations_std.copy())
            per_layer_max_value.append(model.per_layer_max.copy())
            layer_grad_squares={}
            for name,param in model.named_parameters():
                split_name=name.split('.')
                if len(split_name)>=4 and split_name[2]=="h":
                    layer_name=f"h.{split_name[3]}"
                    if layer_name not in layer_grad_squares:
                        layer_grad_squares[layer_name]=0.0
                    layer_grad_squares[layer_name]+=(torch.sum(param.grad**2)).item()
            per_layer_gradient={}
            for layer_name,grad_square_norm in layer_grad_squares.items():
                per_layer_gradient[layer_name]=math.sqrt(grad_square_norm)
            per_layer_gradient_norm.append(per_layer_gradient)   
            for name, param in model.named_parameters():
                if 'wte' in name:
                    print(f"{name} | weight norm: {param.data.norm().item():.4f} | grad norm: {param.grad.norm().item():.4f}") 
            print(f"step {step} | train loss {train_loss:.4f} | gradient_norm {total_grad_norm:.4f}")

    torch.save(losses,"losses.pt")
    torch.save(gradient_norm,"gradient_norm.pt")
    torch.save(per_layer_activation_norm,"per_layer_activation_norm.pt")
    torch.save(per_layer_activation_mean,"per_layer_activation_mean.pt")
    torch.save(per_layer_activation_std,"per_layer_activation_std.pt")
    torch.save(per_layer_gradient_norm,"per_layer_gradient_norm.pt")
    torch.save(per_layer_max_value,"per_layer_max_value.pt")







    

