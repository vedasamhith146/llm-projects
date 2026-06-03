from vllm import LLM,SamplingParams
import torch
import time


model_name="mistralai/Mistral-7B-Instruct-v0.2"

device='cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device : {device}")

sampling_params=SamplingParams(temperature=1.0,max_tokens=100)

llm=LLM(model_name)

prompts=["What is artificial intelligence?","What is gravity?","What is machine learning?","What is vibe coding?"]*25


_ = llm.generate(prompts, SamplingParams(max_tokens=10))

torch.cuda.synchronize()

torch.cuda.reset_peak_memory_stats()


start = time.time()
outputs = llm.generate(prompts, sampling_params)
torch.cuda.synchronize()
end = time.time()

max_memory = torch.cuda.max_memory_allocated()

total_new_tokens=sum(len(out.outputs[0].token_ids) for out in outputs)

print(outputs[0].outputs[0].text)

print(f"Latency: {end-start:.3f}sec")

print(f"Throughput: {(total_new_tokens/(end-start)):.2f}tok/sec")

print(f"Maximum memory allocated:{((max_memory)/(1024**3)):.2f}GB")








