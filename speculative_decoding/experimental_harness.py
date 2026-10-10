import time
from statistics import mean, stdev

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

# EXPERIMENT CONFIGURATION

# Models
TARGET_MODEL_PATH = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DRAFT_MODEL_PATH = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# Prompt
MESSAGES = [
    {
        "role": "system",
        "content": "You are an helpful assistant.",
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

# Generation
NUM_TOKENS = 150
TEMPERATURE = 1.0
TOP_P = None
TOP_K = None

# Gamma sweep
GAMMA_VALUES = [1, 2, 4, 6, 8, 12, 16]

# Benchmark settings
WARMUP_RUNS = 10
MEASUREMENT_RUNS = 30

# Set to None for non-deterministic runs.
# Example: SEED = 1234
SEED = 1234

# LOAD MODELS
print()
print("Loading models...")
target_model, tokenizer = load(TARGET_MODEL_PATH)
draft_model, _ = load(DRAFT_MODEL_PATH)
print("Models loaded.")

# RANDOM SEED
if SEED is not None:
    mx.random.seed(SEED)

# TOKENIZATION
def tokenize_prompt(messages):
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="np",
    )
    return mx.array(input_ids, dtype=mx.uint32)

# SAMPLING UTILITIES
def filter_logits(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits.astype(mx.float32)

    # Greedy
    if temperature == 0:
        return logits

    # Temperature
    logits = logits / temperature

    # Top-k
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        topk_vals = mx.topk(logits, k=top_k, axis=-1)
        kth_value = mx.min(topk_vals, axis=-1, keepdims=True)
        logits = mx.where(logits < kth_value, mx.array(-float("inf")), logits)

    # Top-p
    if top_p is not None and top_p < 1.0:
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        probs = mx.softmax(logits, axis=-1)
        sorted_indices = mx.argsort(-probs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
        cumulative_prob = mx.cumsum(sorted_probs, axis=-1)
        sorted_remove = cumulative_prob > top_p

        # Keep the first token that crosses the threshold.
        sorted_remove = mx.concatenate(
            [mx.zeros_like(sorted_remove[..., :1]), sorted_remove[..., :-1]],
            axis=-1,
        )

        unsort_order = mx.argsort(sorted_indices, axis=-1)
        remove_mask = mx.take_along_axis(sorted_remove, unsort_order, axis=-1)
        logits = mx.where(remove_mask, mx.array(-float("inf")), logits)

    return logits


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None):
    # Greedy
    if temperature == 0:
        token = mx.argmax(logits, axis=-1, keepdims=True).astype(mx.uint32)
        probs = mx.softmax(logits.astype(mx.float32), axis=-1)
        return token, probs

    # Filter
    filtered_logits = filter_logits(
        logits, temperature=temperature, top_k=top_k, top_p=top_p,
    )
    probs = mx.softmax(filtered_logits, axis=-1)

    # Sample
    token = mx.random.categorical(mx.log(probs))
    token = token[:, None].astype(mx.uint32)
    return token, probs

# TARGET-ONLY BENCHMARK
def benchmark_target_only(messages, num_tokens, temperature, top_k, top_p):
    input_vector = tokenize_prompt(messages)
    cache = make_prompt_cache(target_model)

    # Prompt prefill
    logits = target_model(input_vector, cache=cache)
    mx.eval(logits, [c.state for c in cache])

    # First target token
    last_token, _ = sample_from_logits(
        logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p,
    )
    mx.eval(last_token)
    generated_tokens = 1

    if num_tokens <= 1:
        return 0.0

    # Timed decode
    start = time.perf_counter()

    while generated_tokens < num_tokens:
        logits = target_model(last_token, cache=cache)
        last_token, _ = sample_from_logits(
            logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p,
        )

        # IMPORTANT:
        # MLX is lazy, so force completion inside timer.
        mx.eval(last_token)
        generated_tokens += 1

    end = time.perf_counter()
    return end - start

# SPECULATIVE DECODING
def benchmark_speculative(messages, num_tokens, gamma, temperature, top_k, top_p):
    input_vector = tokenize_prompt(messages)
    draft_cache = make_prompt_cache(draft_model)
    target_cache = make_prompt_cache(target_model)

    draft_logits = draft_model(input_vector, cache=draft_cache)
    target_logits = target_model(input_vector, cache=target_cache)
    mx.eval(
        draft_logits, target_logits,
        [c.state for c in draft_cache],
        [c.state for c in target_cache],
    )

    last_token, _ = sample_from_logits(
        target_logits[:, -1, :],
        temperature=temperature, top_k=top_k, top_p=top_p,
    )
    mx.eval(last_token)
    generated_tokens = 1

    if num_tokens <= 1:
        return {
            "time": 0.0, "throughput": 0.0, "alpha": 0.0,
            "accepted": 0, "proposed": 0, "blocks": 0,
            "draft_time": 0.0, "target_verification_time": 0.0,
            "speculative_time": 0.0,
        }

    accepted_draft_tokens = 0
    proposed_draft_tokens = 0
    block_count = 0
    draft_generation_time = 0.0
    target_verification_time = 0.0
    speculative_block_time = 0.0
    draft_y = last_token

    total_start = time.perf_counter()

    while generated_tokens < num_tokens:
        block_count += 1
        current_gamma = min(gamma, num_tokens - generated_tokens)
        block_start = time.perf_counter()

        # ---------------- DRAFT PHASE ----------------
        draft_start = time.perf_counter()
        draft_tokens = []
        draft_probs = []
        y = draft_y

        for _ in range(current_gamma):
            logits = draft_model(y, cache=draft_cache)
            token, probs = sample_from_logits(
                logits[:, -1, :],
                temperature=temperature, top_k=top_k, top_p=top_p,
            )
            mx.eval(token, probs)
            draft_tokens.append(token)
            draft_probs.append(probs)
            y = token

        draft_token_sequence = mx.concatenate(draft_tokens, axis=1)
        draft_probability_matrix = mx.concatenate(draft_probs, axis=0)
        mx.eval(draft_token_sequence, draft_probability_matrix)
        draft_generation_time += time.perf_counter() - draft_start

        # ---------------- TARGET VERIFICATION PHASE ----------------
        verification_input = mx.concatenate([last_token, draft_token_sequence], axis=1)
        verification_start = time.perf_counter()

        target_logits = target_model(verification_input, cache=target_cache)
        target_probs = []
        for i in range(current_gamma + 1):
            filtered = filter_logits(
                target_logits[:, i, :],
                temperature=temperature, top_k=top_k, top_p=top_p,
            )
            target_probs.append(mx.softmax(filtered, axis=-1))
        target_probability_matrix = mx.concatenate(target_probs, axis=0)
        mx.eval(target_probability_matrix)
        target_verification_time += time.perf_counter() - verification_start

        # ---------------- ACCEPT / REJECT ----------------
        accepted_in_block = 0
        rejected = False
        correction_token = None

        if temperature == 0:
            for i in range(current_gamma):
                draft_id = int(draft_tokens[i].item())
                target_token = int(mx.argmax(target_probability_matrix[i]).item())
                proposed_draft_tokens += 1
                if draft_id == target_token:
                    accepted_in_block += 1
                    accepted_draft_tokens += 1
                else:
                    rejected = True
                    correction_token = mx.array([[target_token]], dtype=mx.uint32)
                    break
        else:
            for i in range(current_gamma):
                draft_id = int(draft_tokens[i].item())
                q_value = float(draft_probability_matrix[i, draft_id].item())
                p_value = float(target_probability_matrix[i, draft_id].item())
                proposed_draft_tokens += 1

                acceptance_probability = (
                    0.0 if q_value <= 0.0 else min(1.0, p_value / q_value)
                )
                u = float(mx.random.uniform().item())

                if u < acceptance_probability:
                    accepted_in_block += 1
                    accepted_draft_tokens += 1
                else:
                    rejected = True
                    residual = mx.maximum(
                        target_probability_matrix[i] - draft_probability_matrix[i],
                        0.0,
                    )
                    residual_mass = float(mx.sum(residual).item())

                    if residual_mass <= 0.0:
                        correction_token, _ = sample_from_logits(
                            target_logits[:, i, :],
                            temperature=temperature, top_k=top_k, top_p=top_p,
                        )
                        mx.eval(correction_token)
                    else:
                        correction_token = mx.random.categorical(
                            mx.log(residual / residual_mass)
                        )
                        correction_token = mx.reshape(
                            correction_token, (1, 1)
                        ).astype(mx.uint32)
                        mx.eval(correction_token)
                    break

        # ---------------- CACHE COMMIT ----------------
        if rejected:
            num_unaccepted = current_gamma - accepted_in_block
            trim_prompt_cache(target_cache, num_unaccepted)
            trim_prompt_cache(draft_cache, max(num_unaccepted - 1, 0))
            last_token = correction_token
            draft_y = last_token
            generated_tokens += accepted_in_block + 1
        else:
            accepted_draft_tokens_in_block = accepted_in_block
            if generated_tokens + accepted_draft_tokens_in_block >= num_tokens:
                generated_tokens += accepted_draft_tokens_in_block
                speculative_block_time += time.perf_counter() - block_start
                break

            bonus_token, _ = sample_from_logits(
                target_logits[:, current_gamma, :],
                temperature=temperature, top_k=top_k, top_p=top_p,
            )
            mx.eval(bonus_token)
            last_accepted_draft_token = draft_tokens[-1]
            draft_y = mx.concatenate([last_accepted_draft_token, bonus_token], axis=1)
            last_token = bonus_token
            generated_tokens += accepted_draft_tokens_in_block + 1

        speculative_block_time += time.perf_counter() - block_start

    mx.eval(last_token)
    elapsed = time.perf_counter() - total_start
    timed_tokens = num_tokens - 1

    return {
        "time": elapsed,
        "throughput": timed_tokens / elapsed,
        "alpha": (
            accepted_draft_tokens / proposed_draft_tokens
            if proposed_draft_tokens > 0 else 0.0
        ),
        "accepted": accepted_draft_tokens,
        "proposed": proposed_draft_tokens,
        "blocks": block_count,
        "draft_time": draft_generation_time,
        "target_verification_time": target_verification_time,
        "speculative_time": speculative_block_time,
    }


def summarize(values):
    avg = mean(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    return avg, sd

# TARGET-ONLY BASELINE
def run_target_baseline():
    print()
    print("=" * 78)
    print("TARGET-ONLY BASELINE")
    print("=" * 78)

    # Warmup
    for i in range(WARMUP_RUNS):
        print(f"\rWarmup {i + 1}/{WARMUP_RUNS}", end="", flush=True)
        benchmark_target_only(MESSAGES, NUM_TOKENS, TEMPERATURE, TOP_K, TOP_P)
    print()

    # Measurements
    times = []
    for i in range(MEASUREMENT_RUNS):
        print(f"\rMeasurement {i + 1}/{MEASUREMENT_RUNS}", end="", flush=True)
        elapsed = benchmark_target_only(
            MESSAGES, NUM_TOKENS, TEMPERATURE, TOP_K, TOP_P,
        )
        times.append(elapsed)
    print()

    avg_time, std_time = summarize(times)
    timed_tokens = NUM_TOKENS - 1
    throughput = (timed_tokens * MEASUREMENT_RUNS) / sum(times)

    return {
        "throughput": throughput,
        "avg_time": avg_time,
        "std_time": std_time,
    }

# SPECULATIVE EXPERIMENT FOR ONE GAMMA
def run_gamma(gamma):
    print()
    print(f"Gamma = {gamma}")

    for i in range(WARMUP_RUNS):
        print(f"\rWarmup {i + 1}/{WARMUP_RUNS}", end="", flush=True)
        benchmark_speculative(
            MESSAGES, NUM_TOKENS, gamma,
            TEMPERATURE, TOP_K, TOP_P,
        )
    print()

    times = []
    alphas = []
    accepted = []
    proposed = []
    blocks = []
    draft_times = []
    target_verification_times = []
    speculative_times = []

    for i in range(MEASUREMENT_RUNS):
        print(f"\rMeasurement {i + 1}/{MEASUREMENT_RUNS}", end="", flush=True)
        result = benchmark_speculative(
            MESSAGES, NUM_TOKENS, gamma,
            TEMPERATURE, TOP_K, TOP_P,
        )
        times.append(result["time"])
        alphas.append(result["alpha"])
        accepted.append(result["accepted"])
        proposed.append(result["proposed"])
        blocks.append(result["blocks"])
        draft_times.append(result["draft_time"])
        target_verification_times.append(result["target_verification_time"])
        speculative_times.append(result["speculative_time"])
    print()

    avg_time, std_time = summarize(times)
    avg_alpha, std_alpha = summarize(alphas)
    throughput = ((NUM_TOKENS - 1) * MEASUREMENT_RUNS) / sum(times)
    total_blocks = sum(blocks)

    return {
        "gamma": gamma,
        "throughput": throughput,
        "avg_time": avg_time,
        "std_time": std_time,
        "alpha": avg_alpha,
        "alpha_std": std_alpha,
        "accepted": sum(accepted),
        "proposed": sum(proposed),
        "blocks": total_blocks,
        "draft_time": sum(draft_times),
        "target_verification_time": sum(target_verification_times),
        "speculative_time": sum(speculative_times),
        "draft_time_per_block": sum(draft_times) / total_blocks,
        "target_verification_time_per_block": sum(target_verification_times) / total_blocks,
        "speculative_time_per_block": sum(speculative_times) / total_blocks,
    }


def print_results_table(target_throughput, results):
    print()
    print("=" * 125)
    print("                    SPECULATIVE DECODING TIMING RESULTS")
    print("=" * 125)

    print(
        f"{'Gamma':>7}"
        f"{'Alpha':>10}"
        f"{'Target tok/s':>16}"
        f"{'Spec tok/s':>16}"
        f"{'Speedup':>11}"
        f"{'Draft ms/block':>18}"
        f"{'Target verify ms/block':>24}"
        f"{'Spec ms/block':>17}"
    )

    print("-" * 125)

    for result in results:
        speedup = result["throughput"] / target_throughput

        print(
            f"{result['gamma']:>7}"
            f"{result['alpha']:>10.4f}"
            f"{target_throughput:>16.2f}"
            f"{result['throughput']:>16.2f}"
            f"{speedup:>11.3f}x"
            f"{result['draft_time_per_block'] * 1000:>18.3f}"
            f"{result['target_verification_time_per_block'] * 1000:>24.3f}"
            f"{result['speculative_time_per_block'] * 1000:>17.3f}"
        )

    print("=" * 125)
    print()
    print(f"Target-only baseline: {target_throughput:.2f} tok/s")
    print()
    print("Timing definitions:")
    print("  Draft ms/block           = draft generation + sampling + forced MLX evaluation.")
    print("  Target verify ms/block   = target forward + probability processing + forced MLX evaluation inside speculative decoding.")
    print("  Spec ms/block            = complete speculative block, including acceptance/rejection, correction/bonus sampling, cache operations, and synchronization.")
    print()

# MAIN EXPERIMENT
def main():
    print()
    print("=" * 80)
    print("SPECULATIVE DECODING EXPERIMENTAL HARNESS")
    print("=" * 80)

    print()
    print("Configuration")
    print("-" * 80)

    print(f"Target model:       {TARGET_MODEL_PATH}")
    print(f"Draft model:        {DRAFT_MODEL_PATH}")
    print(f"Generated tokens:   {NUM_TOKENS}")
    print(f"Gamma values:       {GAMMA_VALUES}")
    print(f"Temperature:        {TEMPERATURE}")
    print(f"Top-k:              {TOP_K}")
    print(f"Top-p:              {TOP_P}")
    print(f"Warmup runs:        {WARMUP_RUNS}")
    print(f"Measurement runs:   {MEASUREMENT_RUNS}")
    print(f"Random seed:        {SEED}")
    print()

    # TARGET BASELINE
    baseline = run_target_baseline()
    target_throughput = baseline["throughput"]

    print()
    print(f"Target-only throughput: {target_throughput:.2f} tok/s")

    # GAMMA SWEEP
    results = []

    for gamma in GAMMA_VALUES:
        # FULL SPECULATIVE DECODING
        result = run_gamma(gamma)
        results.append(result)

        speedup = result["throughput"] / target_throughput
        print(
            f"Gamma {gamma:>2}: "
            f"alpha={result['alpha']:.4f}, "
            f"spec={result['throughput']:.2f} tok/s, "
            f"speedup={speedup:.3f}x, "
            f"draft={result['draft_time_per_block'] * 1000:.3f} ms/block, "
            f"verify={result['target_verification_time_per_block'] * 1000:.3f} ms/block, "
            f"spec_block={result['speculative_time_per_block'] * 1000:.3f} ms/block, "
        )

    # FINAL TABLE
    print_results_table(
        target_throughput,
        results,
    )

    # EXTRA STATISTICS
    print()
    print("=" * 95)
    print("ADDITIONAL STATISTICS")
    print("=" * 95)

    for result in results:
        print(f"\nGamma = {result['gamma']}")
        print(f"  Alpha:              {result['alpha']:.4f} +/- {result['alpha_std']:.4f}")
        print(f"  Accepted tokens:    {result['accepted']}")
        print(f"  Tested positions:   {result['proposed']}")
        print(f"  Speculative blocks: {result['blocks']}")
        print(f"  Mean block time:    {result['avg_time'] * 1000:.3f} ms")
        print(f"  Block time std:     {result['std_time'] * 1000:.3f} ms")

    print()
    print("=" * 95)
    print("EXPERIMENT COMPLETE")
    print("=" * 95)

# ENTRY POINT
if __name__ == "__main__":
    main()