import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

NUM_TOKENS = 100
WARMUP_RUNS = 10
MEASURED_RUNS = 30


messages = [
    {
        "role": "system",
        "content": "You are a helpful and concise assistant.",
    },
    {
        "role": "user",
        "content": (
            "Write a detailed science-fiction story about a spacecraft "
            "that lands on a mysterious planet after losing communication "
            "with Earth. Describe the alien landscape, the spacecraft "
            "interior, the sounds and atmosphere around the crew, and the "
            "discovery that something unexpected is moving beneath the "
            "surface. Build the story gradually and end with a surprising "
            "discovery."
        ),
    },
]


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("Loading model...")

model, tokenizer = load(MODEL_PATH)


# --------------------------------------------------
# Prepare prompt ONCE
# --------------------------------------------------

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="np",
)

prompt = mx.array(input_ids)


# --------------------------------------------------
# Target-only generation
# --------------------------------------------------

def generate_tokens_target(num_tokens):

    # Create KV cache
    cache = make_prompt_cache(model)

    # -----------------------------
    # Prefill
    # -----------------------------

    logits = model(
        prompt,
        cache=cache,
    )

    # Select first token greedily.
    token = mx.argmax(
        logits[:, -1, :],
        axis=-1,
    )[:, None]

    mx.eval(token)

    # -----------------------------
    # Decode
    # -----------------------------

    start = time.perf_counter()

    for _ in range(num_tokens):

        logits = model(
            token,
            cache=cache,
        )

        token = mx.argmax(
            logits[:, -1, :],
            axis=-1,
        )[:, None]

    # Force all lazy MLX computation to finish
    mx.eval(token)

    end = time.perf_counter()

    return end - start


# --------------------------------------------------
# Warmup
# --------------------------------------------------

print("Warming up...")

for _ in range(WARMUP_RUNS):
    generate_tokens_target(NUM_TOKENS)


# --------------------------------------------------
# Measurement
# --------------------------------------------------

print("Measuring...")

times = []

for i in range(MEASURED_RUNS):

    elapsed = generate_tokens_target(NUM_TOKENS)

    times.append(elapsed)

    print(
        f"Run {i + 1:2d}: "
        f"{NUM_TOKENS / elapsed:.2f} tok/s"
    )


# --------------------------------------------------
# Statistics
# --------------------------------------------------

avg_time = sum(times) / len(times)

tok_per_sec = NUM_TOKENS / avg_time

latency_ms = avg_time / NUM_TOKENS * 1000


print("\n========== RESULTS ==========")

print(f"Model: {MODEL_PATH}")
print(f"Generated tokens: {NUM_TOKENS}")
print(f"Measured runs: {MEASURED_RUNS}")

print(f"\nAverage time: {avg_time:.6f} s")
print(f"Throughput: {tok_per_sec:.2f} tok/s")
print(f"Latency: {latency_ms:.3f} ms/token")