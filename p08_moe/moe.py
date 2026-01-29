import torch
import torch.nn as nn
import torch.nn.functional as F


din=128
noe=8
topk=2

class MOElayer(nn.Module):
    def __init__(self,din,noe):
        super().__init__()
        self.noe=noe
        self.experts=nn.ModuleList([nn.Linear(din,din) for _ in range(noe)])
        self.router=nn.Linear(din,noe,bias=False)
        self.expert_history=[]
        
    def forward(self,x,topk):
        batch_size,dim=x.shape
        router_logits=self.router(x)
        routing_probs,selected_indices=torch.topk(router_logits,k=topk,dim=-1)
        self.expert_history.append(selected_indices.detach().cpu().flatten())
        routing_weights=F.softmax(routing_probs,dim=-1)
        final_output=torch.zeros_like(x)
        for expert_idx in range(self.noe):
            expert_layer=self.experts[expert_idx]
            batch_indices,k_indices=torch.where(selected_indices==expert_idx)
            if len(batch_indices)==0:
                continue
            expert_input=x[batch_indices]
            expert_output=expert_layer(expert_input)
            weight_factor=routing_weights[batch_indices,k_indices].unsqueeze(-1)
            weighted_output=expert_output*weight_factor
            final_output.index_add_(0,batch_indices,weighted_output)
        return final_output
    









