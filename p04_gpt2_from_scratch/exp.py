from transformers import GPT2LMHeadModel

# Load GPT-2 (124M)
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Print all named parameters
for name, param in model.named_parameters():
    print(name, param.shape)