import torch
import torch.nn as nn

torch.manual_seed(42)
vocab_size=10
embedding_dim=2

weight_matrix=torch.randn(vocab_size,embedding_dim)

def one_hot_encode(token_id):
    output=torch.zeros((vocab_size,))
    output[token_id]=1.
    final_output = output.unsqueeze(0) @ weight_matrix
    return final_output

def direct_encode(token_id):
    direct_encoding=nn.Embedding(vocab_size,embedding_dim)
    with torch.no_grad():
        direct_encoding.weight.copy_(weight_matrix)
    output=direct_encoding((token_id))
    return output

output_from_one = one_hot_encode(2)
output_from_two = direct_encode(torch.tensor(2))

print(output_from_one)
print(output_from_two)

