import time
import statistics

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


# ============================================================
# Configuration
# ============================================================

TARGET_MODEL_PATH = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DRAFT_MODEL_PATH = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

NUM_WARMUPS = 10
NUM_MEASUREMENTS = 100

NEXT_TOKEN_ID = 100

messages = [
    {
        "role": "system",
        "content": "You are a helpful and concise assistant."
    },
    {
        "role": "user",
        "content": "What is the capital of France?"
    }
]


# ============================================================
# Load models
# ============================================================

print("Loading models...")

target_model, target_tokenizer = load(TARGET_MODEL_PATH)
draft_model, draft_tokenizer = load(DRAFT_MODEL_PATH)

print("Models loaded.")


# ============================================================
# Prepare input for each model
# ============================================================

def prepare_input(tokenizer):

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="np"
    )

    return mx.array(input_ids)


target_input = prepare_input(target_tokenizer)
draft_input = prepare_input(draft_tokenizer)


# ============================================================
# Prefill
# ============================================================

def prefill(model, input_vector):

    cache = make_prompt_cache(model)

    logits = model(
        input_vector,
        cache=cache
    )

    # Force computation to finish.
    mx.eval(logits)

    return cache


# ============================================================
# Measure one cached decode step
# ============================================================

def measure_model(
    model,
    input_vector,
    num_warmups=NUM_WARMUPS,
    num_measurements=NUM_MEASUREMENTS
):

    # --------------------------------------------------------
    # Fixed decode token
    # --------------------------------------------------------

    next_token = mx.array(
        [[NEXT_TOKEN_ID]],
        dtype=mx.uint32
    )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    for _ in range(num_warmups):

        cache = prefill(
            model,
            input_vector
        )

        logits = model(
            next_token,
            cache=cache
        )

        mx.eval(logits)

    # --------------------------------------------------------
    # Measurements
    # --------------------------------------------------------

    times = []

    for _ in range(num_measurements):

        # Fresh cache for every measurement so that
        # sequence length is identical every time.
        cache = prefill(
            model,
            input_vector
        )

        start = time.perf_counter()

        logits = model(
            next_token,
            cache=cache
        )

        # Important: MLX execution is lazy, so force the
        # computation to finish before stopping the timer.
        mx.eval(logits)

        end = time.perf_counter()

        times.append(end - start)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times)

    tok_per_sec = 1.0 / mean_time

    return {
        "mean_time": mean_time,
        "std_time": std_time,
        "tok_per_sec": tok_per_sec,
        "times": times,
    }


# ============================================================
# Measure draft
# ============================================================

print("\nMeasuring draft model...")

draft_results = measure_model(
    draft_model,
    draft_input
)


# ============================================================
# Measure target
# ============================================================

print("Measuring target model...")

target_results = measure_model(
    target_model,
    target_input
)


# ============================================================
# Calculate c
# ============================================================

draft_time = draft_results["mean_time"]
target_time = target_results["mean_time"]

c = draft_time / target_time


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 65)
print("DRAFT vs TARGET DECODE COST")
print("=" * 65)

print(
    f"Draft model : "
    f"{draft_time * 1000:.4f} ± "
    f"{draft_results['std_time'] * 1000:.4f} ms/token"
)

print(
    f"Target model: "
    f"{target_time * 1000:.4f} ± "
    f"{target_results['std_time'] * 1000:.4f} ms/token"
)

print()

print(
    f"Draft throughput : "
    f"{draft_results['tok_per_sec']:.2f} tok/s"
)

print(
    f"Target throughput: "
    f"{target_results['tok_per_sec']:.2f} tok/s"
)

print()

print(f"c = {c:.4f}")

print("=" * 65)

print(
    "\nInterpretation:"
)

print(
    f"The draft decode step costs approximately "
    f"{c * 100:.2f}% of the target decode step."
)