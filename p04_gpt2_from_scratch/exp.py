import torch
p=torch.tensor([[10,9,8,7,6]])
cumm_p=torch.cumsum(p,dim=-1)
c=cumm_p>18
c[:,1:]=c[:,:-1].clone()
print(c)