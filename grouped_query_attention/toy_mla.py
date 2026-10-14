import torch
import torch.nn as nn

class ToyLatentKVCompression(nn.Module):
    def __init__(self, d_model=768, latent_dim=128):
        super().__init__()
        self.down_proj = nn.Linear(d_model,latent_dim,bias=False)
        self.up_proj_k = nn.Linear(latent_dim,d_model,bias=False)
        self.up_proj_v = nn.Linear(latent_dim,d_model,bias=False)

    def forward(self, x):
        latent = self.down_proj(x)
        k = self.up_proj_k(latent)
        v = self.up_proj_v(latent)
        
        return latent, k, v