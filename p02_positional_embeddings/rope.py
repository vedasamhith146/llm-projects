from token_tensor import token_embd
import torch
import math
g=torch.Generator()
g.manual_seed(42)
dim=token_embd.size(1)
Wq=torch.randn((dim,dim),generator=g)
Wk=torch.randn((dim,dim),generator=g)
q=token_embd @ Wq
k=token_embd @ Wk

def cos_temp(pos,dim):
    cos_temp=[]
    for i in range(dim//2):
        temp=math.cos(pos*pow(10000,-2*i/dim))
        cos_temp.append(temp)
        cos_temp.append(temp)
    cos_temp=torch.tensor(cos_temp).unsqueeze(0)
    return cos_temp

def sin_temp(pos,dim):
    sin_temp=[]
    for i in range(dim//2):
        temp=math.sin(pos*pow(10000,-2*i/dim))
        sin_temp.append(temp)
        sin_temp.append(temp)
    sin_temp=torch.tensor(sin_temp).unsqueeze(0)
    return sin_temp

def pos_reverse(pos_vec):
    new_vec=torch.zeros((1,dim))
    for i in range(dim//2):
        new_vec[0,2*i]=-pos_vec[0,2*i+1]
        new_vec[0,2*i+1]=pos_vec[0,2*i]
    return new_vec

def rotation(rot_mat):
    for i in range(rot_mat.size(0)):
        rot_temp=rot_mat[i].unsqueeze(0)
        cos_tab=cos_temp(i,dim)
        sin_tab=sin_temp(i,dim)
        rot_temp_rev=pos_reverse(rot_temp)
        rot_temp=(rot_temp*cos_tab)+(rot_temp_rev*sin_tab)
        rot_mat[i]=rot_temp
    return rot_mat

q=rotation(q)
k=rotation(k)




   
    