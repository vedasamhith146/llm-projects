from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
import time

model_name="mistralai/Mistral-7B-Instruct-v0.2"


device='cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device : {device}")

tokenizer=AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token=tokenizer.eos_token

model=AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",torch_dtype=torch.float16)

prompts=["What is artificial intelligence?","What is gravity?","What is machine learning?","What is vibe coding?"]*25

inputs=tokenizer(prompts,return_tensors="pt",padding=True).to(device)

with torch.inference_mode():
    _ = model.generate(**inputs, max_new_tokens=10)

torch.cuda.synchronize()

torch.cuda.reset_peak_memory_stats()


with torch.inference_mode():
    start=time.time()
    outputs=model.generate(**inputs,max_new_tokens=100)
    torch.cuda.synchronize()
    end=time.time()

max_memory= torch.cuda.max_memory_allocated()

input_length=inputs["input_ids"].shape[1]
batch_size=outputs.shape[0]

total_new_tokens=0
for i in range(batch_size):
    generated=outputs[i,:input_length]
    eos_position=(generated==tokenizer.eos_token_id).nonzero()
    if len(eos_position)>0:
        actual_len=eos_position[0].item()+1
    else:
        actual_len=len(generated)

    total_new_tokens+=actual_len

print(tokenizer.decode(outputs[4],skip_special_tokens=True))

print(f"Latency: {end-start:.3f}sec")

print(f"Throughput: {(total_new_tokens/(end-start)):.2f}tok/sec")

print(f"Maximum memory allocated:{((max_memory)/(1024**3)):.2f}GB")

