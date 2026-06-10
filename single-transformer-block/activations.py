import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def GeLU(x):
    return (1+F.tanh((math.sqrt(2/math.pi))*(x + 0.044715*(x**3))))*(0.5*x)

class SwiGLU_MLP(nn.Module):
    def __init__(self,x):
        super().__init__()
        self.W=torch.randn((x.size(-1),4*x.size(-1)))
        self.V=torch.randn((x.size(-1),4*x.size(-1)))
        self.P=torch.randn((4*x.size(-1),x.size(-1)))

    def forward(self,x):
        input_1=x @ self.W
        input_2=x @ self.V
        swish=(input_1)*F.sigmoid(input_1)
        output= (swish * input_2) @ self.P
        return output
    

if __name__=="__main__":
    input=torch.randn((2,4,8))
    output_1=GeLU(input)
    output_2=F.gelu(input)
    if torch.sum(((output_1-output_2)**2)).item()<0.01:
        print("Yayyy you have implemented GeLU correctly")
    swiglu=SwiGLU_MLP(input)
    print(f"output from SwiGLU_MLP : {swiglu(input)}")
    
