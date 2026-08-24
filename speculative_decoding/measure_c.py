import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import time

target_model_path='mlx-community/Qwen2.5-1.5B-Instruct-4bit'
draft_model_path='mlx-community/Qwen2.5-0.5B-Instruct-4bit'
print("loading model...")

target_model,tokenizer=load(target_model_path)
draft_model,tokenizer=load(draft_model_path)

messages=[
    {"role":"system","content":"You are a helpful and concise assistant."},
    {"role":"user","content":"What is the capital of France?"}
]

input_ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors="np")
input_vector=mx.array(input_ids)

target_cache=make_prompt_cache(target_model)
draft_cache=make_prompt_cache(draft_model)

def measure_model_time(model, input_vector, cache, num_runs=100):

    # Warmup
    for _ in range(10):
        logits = model(input_vector, cache=cache)
        mx.eval(logits)

    start = time.perf_counter()

    for _ in range(num_runs):
        logits = model(input_vector, cache=cache)
        mx.eval(logits)

    end = time.perf_counter()

    return (end - start) / num_runs


# Measure draft
draft_time = measure_model_time(
    draft_model,
    input_vector[:, -1:],
    draft_cache
)


# Measure target
target_time = measure_model_time(
    target_model,
    input_vector[:, -1:],
    target_cache
)


c = draft_time / target_time


print(f"Draft model latency: {draft_time*1000:.4f} ms")
print(f"Target model latency: {target_time*1000:.4f} ms")
print(f"c = {c:.4f}")

#print(f"Input sequence length:{input_vector.shape[1]} tokens")
#print(f"Logits shape:{logits.shape}")