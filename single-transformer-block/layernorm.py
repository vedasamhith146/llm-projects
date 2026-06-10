import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self,input):
        super().__init__()
        self.alpha=torch.ones((input.size(-1)))
        self.beta=torch.zeros((input.size(-1)))
        self.eps=1e-12
    def forward(self,x):
        mean=torch.mean(x,dim=-1).unsqueeze(-1)
        var= ((torch.sum(((x - mean)**2),dim=-1))/x.size(-1)).unsqueeze(-1)
        output = self.alpha*((x-mean)/torch.sqrt(var+self.eps)) + self.beta
        return output
    
if __name__=="__main__":
    input=torch.randn((2,4,8))
    ln=LayerNorm(input)
    output=ln(input)
    ln_torch=nn.LayerNorm(input.size(-1))
    output_torch=ln_torch(input)
    if torch.sum(((output-output_torch)**2)).item()<0.01:
        print(torch.mean(output,dim=-1))
        print(torch.var(output,dim=-1,unbiased=False))
        print("Yayyy you have implemented it correctly")


