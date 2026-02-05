import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm:
    def __init__(self,feature_dim):
        self.gamma=torch.ones((feature_dim,))#shape of gamma and beta will be of (feature_dim,)
        self.beta=torch.zeros((feature_dim,))
        self.epsilon=1e-5
    def forward(self,x):
        B,T,C=x.size()
        mean=(x.sum(dim=-1)/C).unsqueeze(-1)
        y=(x-mean)**2
        std=((y.sum(dim=-1)/C).unsqueeze(-1)+self.epsilon)**0.5
        return ((x-mean)/std)*self.gamma +self.beta
    
class RMSNorm:
    def __init__(self,feature_dim):
        self.gamma=torch.ones((feature_dim,))
        self.epsilon=1e-5
    def forward(self,x):
        B,T,C=x.size()
        y=(x)**2
        std=((y.sum(dim=-1)/C).unsqueeze(-1)+self.epsilon)**0.5
        return (x/std)*self.gamma 
    
class SwiGLUMLP:
    def __init__(self,feature_dim,output_dim):
        self.W1=torch.randn((feature_dim,output_dim))
        self.W2=torch.randn((feature_dim,output_dim))
        self.W3=torch.randn((output_dim,feature_dim))
    def forward(self,x):
        y1=x@self.W1
        y2=x@self.W2
        y2=y2/(1+torch.exp(-y2))
        y=y1*y2
        return y @ self.W3
    
class GeLUMLP:
    def __init__(self,feature_dim,output_dim):
        self.W1=torch.randn((feature_dim,output_dim))
        self.W2=torch.randn((output_dim,feature_dim))
    def forward(self,x):
        y=x@self.W1
        y=0.5*y*(1+torch.tanh(((2/math.pi)**0.5)*(y+0.044715*(y**3))))
        return y@self.W2
    




    

    



