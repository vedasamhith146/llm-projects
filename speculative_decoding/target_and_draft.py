import time
from statistics import mean, stdev

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

# ============================================================
# CONFIGURATION
# ============================================================
TARGET_MODEL_PATH = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DRAFT_MODEL_PATH = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

NUM_TOKENS = 150
GAMMA = 1

TEMPERATURE = 1.0
TOP_K = None
TOP_P = None

WARMUP_RUNS = 10
MEASUREMENT_RUNS = 30

# ============================================================
# LOAD MODELS
# ============================================================
print("Loading models...")
target_model, tokenizer = load(TARGET_MODEL_PATH)
draft_model, _ = load(DRAFT_MODEL_PATH)

# ============================================================
# PROMPT
# ============================================================
messages = [
    {
        "role": "system",
        "content": "You are an helpful assistant."
    },
    {
        "role": "user",
        "content":
        "Write a detailed science-fiction story about a spacecraft "
        "that lands on a mysterious planet after losing communication "
        "with Earth. Describe the alien landscape, the spacecraft "
        "interior, the sounds and atmosphere around the crew, and the "
        "discovery that something unexpected is moving beneath the "
        "surface. Build the story gradually and end with a surprising "
        "discovery."
    }
]

# ============================================================
# TOKEN SAMPLER
# ============================================================
def next_token_generator(logits, temp=1.0, top_k=None, top_p=None):
    """
    Returns:
        next_token: shape (1, 1), dtype uint32
        probs:      shape (1, vocab_size)
    """

    # --------------------------------------------------------
    # Greedy
    # --------------------------------------------------------
    if temp == 0:
        next_token = mx.argmax(logits, axis=-1, keepdims=True).astype(mx.uint32)
        probs = mx.softmax(logits, axis=-1)
        return next_token, probs

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------
    logits = logits / temp

    # --------------------------------------------------------
    # Top-k
    # --------------------------------------------------------
    if top_k is not None:
        topk_vals = mx.topk(logits, k=top_k, axis=-1)
        kth_value = mx.min(topk_vals, axis=-1, keepdims=True)
        logits = mx.where(logits < kth_value, mx.array(-float("inf")), logits)

    # --------------------------------------------------------
    # Top-p
    # --------------------------------------------------------
    if top_p is not None and top_p < 1.0:
        probs = mx.softmax(logits, axis=-1)
        sorted_indices = mx.argsort(-probs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
        cumulative_prob = mx.cumsum(sorted_probs, axis=-1)
        sorted_remove = cumulative_prob > top_p

        # Keep the first token that crosses the threshold.
        sorted_remove = mx.concatenate(
            [mx.zeros_like(sorted_remove[..., :1]), sorted_remove[..., :-1]],
            axis=-1
        )

        unsort_order = mx.argsort(sorted_indices, axis=-1)
        remove_mask = mx.take_along_axis(sorted_remove, unsort_order, axis=-1)
        logits = mx.where(remove_mask, mx.array(-float("inf")), logits)

    # --------------------------------------------------------
    # Final probability distribution
    # --------------------------------------------------------
    probs = mx.softmax(logits, axis=-1)
    next_token = mx.random.categorical(mx.log(probs))

    # MLX categorical gives shape (batch,)
    # Convert to (batch, 1)
    next_token = next_token[:, None].astype(mx.uint32)

    return next_token, probs

# ============================================================
# SPECULATIVE DECODING
# ============================================================
def generate_tokens_speculative(messages, num_tokens, gamma, temp=1.0, top_k=None, top_p=None):

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------
    beta_values = []
    accepted_draft_tokens = 0
    proposed_draft_tokens = 0

    # --------------------------------------------------------
    # Tokenize prompt
    # --------------------------------------------------------
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="np"
    )
    input_vector = mx.array(input_ids, dtype=mx.uint32)

    # --------------------------------------------------------
    # Create caches
    # --------------------------------------------------------
    draft_cache = make_prompt_cache(draft_model)
    target_cache = make_prompt_cache(target_model)

    # --------------------------------------------------------
    # PREFILL
    #
    # Both caches now contain the entire prompt.
    # --------------------------------------------------------
    draft_model(input_vector, cache=draft_cache)
    target_logits = target_model(input_vector, cache=target_cache)

    # --------------------------------------------------------
    # Generate first target token
    #
    # This is the first output token.
    #
    # IMPORTANT:
    # Its forward pass is part of prompt prefill and therefore
    # is outside the decode timer.
    # --------------------------------------------------------
    last_token, _ = next_token_generator(
        target_logits[:, -1, :], temp=temp, top_k=top_k, top_p=top_p
    )
    mx.eval(last_token)

    # --------------------------------------------------------
    # We count this first token as generated.
    #
    # Therefore NUM_TOKENS = total output tokens.
    # The timer measures the remaining NUM_TOKENS - 1 tokens.
    # --------------------------------------------------------
    generated_tokens = 1

    if num_tokens <= 1:
        return {
            "time": 0.0,
            "throughput": 0.0,
            "latency": 0.0,
            "expected_alpha": 0.0,
            "realized_alpha": 0.0,
            "accepted_tokens": 0,
            "proposed_tokens": 0
        }

    # --------------------------------------------------------
    # Pending draft token
    #
    # When all gamma draft tokens are accepted, the last draft
    # token has been verified by target but has NOT yet been
    # processed by the draft cache.
    #
    # We keep it here so that the next draft forward can process:
    #
    # [last_accepted_draft_token, next_target_token]
    #
    # together.
    # --------------------------------------------------------
    pending_draft_token = None

    # ========================================================
    # START DECODE TIMER
    # ========================================================
    start = time.perf_counter()

    while generated_tokens < num_tokens:
        remaining_tokens = num_tokens - generated_tokens
        current_gamma = min(gamma, remaining_tokens)

        # ----------------------------------------------------
        # IMPORTANT STATE:
        #
        # last_token = token that target/draft caches have NOT
        #              processed yet.
        #
        # target_cache contains everything BEFORE last_token.
        # ----------------------------------------------------
        base_token = last_token

        # ----------------------------------------------------
        # DRAFT GENERATION
        # ----------------------------------------------------
        draft_tokens = []
        draft_probs_list = []
        first_draft_step = True

        for _ in range(current_gamma):
            if first_draft_step and pending_draft_token is not None:
                # The draft cache is missing the previous block's
                # final accepted draft token.
                #
                # Process:
                #
                # [pending_draft_token, base_token]
                #
                # in one forward pass.
                draft_input = mx.concatenate([pending_draft_token, base_token], axis=1)
                draft_logits = draft_model(draft_input, cache=draft_cache)

                # Last position predicts the first new draft token.
                draft_logits = draft_logits[:, -1, :]
                pending_draft_token = None
            else:
                draft_logits = draft_model(base_token, cache=draft_cache)
                draft_logits = draft_logits[:, -1, :]

            first_draft_step = False

            draft_token, draft_probs = next_token_generator(
                draft_logits, temp=temp, top_k=top_k, top_p=top_p
            )
            draft_tokens.append(draft_token)
            draft_probs_list.append(draft_probs)

            # The newly sampled draft token becomes the input
            # for the next draft prediction.
            base_token = draft_token

        # ----------------------------------------------------
        # Stack once
        # ----------------------------------------------------
        draft_token_probabilities = mx.concatenate(draft_probs_list, axis=0)
        draft_token_sequence = mx.concatenate(draft_tokens, axis=1)

        # ====================================================
        # TARGET VERIFICATION
        # ====================================================

        # last_token is the token immediately before the draft
        # sequence and has NOT yet been processed by target_cache.
        #
        # Therefore target sees:
        #
        # [last_token, draft_1, ..., draft_gamma]
        #
        # Target cache already contains everything before last_token.
        verification_input = mx.concatenate([last_token, draft_token_sequence], axis=1)
        target_logits = target_model(verification_input, cache=target_cache)
        target_token_probabilities = mx.softmax(target_logits, axis=-1)

        # ====================================================
        # ACCEPT / REJECT
        # ====================================================
        rejection_index = None

        for i in range(current_gamma):
            q_i = draft_token_probabilities[i]
            p_i = target_token_probabilities[0, i]

            # ------------------------------------------------
            # Expected acceptance probability
            #
            # beta_i = sum_x min(p_i(x), q_i(x))
            # ------------------------------------------------
            beta_i = mx.sum(mx.minimum(p_i, q_i))
            beta_values.append(beta_i)

            # ------------------------------------------------
            # Probability of the actual draft token
            # ------------------------------------------------
            token = draft_tokens[i]
            q_token = mx.take_along_axis(q_i[None, :], token, axis=-1).squeeze()
            p_token = mx.take_along_axis(p_i[None, :], token, axis=-1).squeeze()

            ratio = mx.minimum(mx.array(1.0), p_token / q_token)
            random_value = mx.random.uniform()

            # Force ONLY the scalar decision to evaluate.
            mx.eval(ratio, random_value)

            proposed_draft_tokens += 1

            if random_value.item() <= ratio.item():
                # ACCEPT
                accepted_draft_tokens += 1
                generated_tokens += 1
            else:
                # REJECT
                rejection_index = i
                break

        # ====================================================
        # REJECTION
        # ====================================================
        if rejection_index is not None:
            i = rejection_index

            # ------------------------------------------------
            # Residual distribution:
            #
            # r(x) ∝ max(p(x) - q(x), 0)
            # ------------------------------------------------
            residual = (
                target_token_probabilities[0, i]
                - draft_token_probabilities[i]
            )
            residual = mx.maximum(residual, mx.array(0.0))
            residual_mass = mx.sum(residual)

            # Numerical safety.
            residual = mx.where(
                residual_mass > 0,
                residual / residual_mass,
                target_token_probabilities[0, i]
            )

            correction_token = mx.random.categorical(mx.log(residual))
            correction_token = mx.reshape(correction_token, (1, 1)).astype(mx.uint32)

            # ------------------------------------------------
            # Cache rewind
            #
            # Target cache currently contains:
            #
            # prompt + last_token + all gamma draft tokens
            #
            # We want:
            #
            # prompt + last_token + accepted draft tokens
            #
            # Therefore remove:
            #
            # gamma - accepted
            # ------------------------------------------------
            num_unaccepted_draft = current_gamma - i

            trim_prompt_cache(target_cache, num_unaccepted_draft)

            # Draft cache has NOT processed the final draft
            # token, so it has one fewer speculative token.
            trim_prompt_cache(draft_cache, max(num_unaccepted_draft - 1, 0))

            # ------------------------------------------------
            # Correction token is now the next unprocessed token.
            # ------------------------------------------------
            last_token = correction_token
            pending_draft_token = None
            generated_tokens += 1

        # ====================================================
        # ALL DRAFT TOKENS ACCEPTED
        # ====================================================
        else:
            # All current_gamma draft tokens are accepted.

            # If we've already generated the requested number,
            # there is no need for the extra target token.
            if generated_tokens >= num_tokens:
                break

            # ------------------------------------------------
            # Sample the "bonus" target token.
            #
            # target_logits[-1] predicts the token AFTER the
            # final draft token.
            # ------------------------------------------------
            extra_token, _ = next_token_generator(
                target_logits[:, -1, :], temp=temp, top_k=top_k, top_p=top_p
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # target_cache already contains:
            #
            # prompt + last_token + all accepted drafts
            #
            # extra_token is NOT in target_cache.
            #
            # draft_cache is missing the final accepted draft
            # token. Save it as pending so that next iteration
            # can process:
            #
            # [final_draft_token, extra_token]
            #
            # together.
            # ------------------------------------------------
            pending_draft_token = draft_tokens[-1]
            last_token = extra_token
            generated_tokens += 1

    # ========================================================
    # END TIMER
    # ========================================================
    mx.eval(last_token)
    end = time.perf_counter()

    # ========================================================
    # METRICS
    # ========================================================
    elapsed = end - start

    if beta_values:
        beta_array = mx.stack(beta_values)
        mx.eval(beta_array)
        expected_alpha = beta_array.mean().item()
    else:
        expected_alpha = 0.0

    if proposed_draft_tokens > 0:
        realized_alpha = accepted_draft_tokens / proposed_draft_tokens
    else:
        realized_alpha = 0.0

    # Number of timed tokens:
    #
    # Initial target token is outside timer.
    #
    # Therefore timer covers num_tokens - 1 tokens.
    timed_tokens = num_tokens - 1
    throughput = timed_tokens / elapsed
    latency = elapsed / timed_tokens

    return {
        "time": elapsed,
        "throughput": throughput,
        "latency": latency,
        "expected_alpha": expected_alpha,
        "realized_alpha": realized_alpha,
        "accepted_tokens": accepted_draft_tokens,
        "proposed_tokens": proposed_draft_tokens
    }

# ============================================================
# EXPERIMENT
# ============================================================
if __name__ == "__main__":

    print("\n========== EXPERIMENT CONFIG ==========")
    print(f"Target model:       {TARGET_MODEL_PATH}")
    print(f"Draft model:        {DRAFT_MODEL_PATH}")
    print(f"Generated tokens:   {NUM_TOKENS}")
    print(f"Gamma:              {GAMMA}")
    print(f"Temperature:        {TEMPERATURE}")
    print(f"Top-k:              {TOP_K}")
    print(f"Top-p:              {TOP_P}")
    print(f"Warmup runs:        {WARMUP_RUNS}")
    print(f"Measurement runs:   {MEASUREMENT_RUNS}")

    # ========================================================
    # WARMUP
    # ========================================================
    print("\n========== WARMUP ==========")

    for i in range(WARMUP_RUNS):
        print(f"Warmup {i + 1}/{WARMUP_RUNS}")
        generate_tokens_speculative(messages, NUM_TOKENS, GAMMA, TEMPERATURE, TOP_K, TOP_P)

    # ========================================================
    # MEASUREMENTS
    # ========================================================
    times = []
    expected_alphas = []
    realized_alphas = []
    accepted_counts = []
    proposed_counts = []

    print("\n========== MEASUREMENTS ==========")

    for i in range(MEASUREMENT_RUNS):
        print(f"Run {i + 1}/{MEASUREMENT_RUNS}")

        result = generate_tokens_speculative(
            messages, NUM_TOKENS, GAMMA, TEMPERATURE, TOP_K, TOP_P
        )

        times.append(result["time"])
        expected_alphas.append(result["expected_alpha"])
        realized_alphas.append(result["realized_alpha"])
        accepted_counts.append(result["accepted_tokens"])
        proposed_counts.append(result["proposed_tokens"])

    # ========================================================
    # AGGREGATE RESULTS
    # ========================================================
    avg_time = mean(times)

    # Aggregate throughput is better than simply averaging
    # per-run throughput.
    total_timed_tokens = (NUM_TOKENS - 1) * MEASUREMENT_RUNS
    total_time = sum(times)
    aggregate_throughput = total_timed_tokens / total_time
    latency_per_token = total_time / total_timed_tokens

    avg_expected_alpha = mean(expected_alphas)
    avg_realized_alpha = mean(realized_alphas)

    # ========================================================
    # VARIABILITY
    # ========================================================
    if MEASUREMENT_RUNS > 1:
        time_std = stdev(times)
    else:
        time_std = 0.0

    # ========================================================
    # RESULTS
    # ========================================================
    print("\n")
    print("==============================================")
    print("       SPECULATIVE DECODING RESULTS")
    print("==============================================")

    print(f"Generated tokens:       {NUM_TOKENS}")
    print(f"Timed tokens:           {NUM_TOKENS - 1}")
    print(f"Gamma:                  {GAMMA}")
    print(f"Temperature:            {TEMPERATURE}")
    print(f"Top-k:                  {TOP_K}")
    print(f"Top-p:                  {TOP_P}")
    print(f"Warmup runs:            {WARMUP_RUNS}")
    print(f"Measurement runs:       {MEASUREMENT_RUNS}")

    print("----------------------------------------------")

    print(f"Average time:           {avg_time:.4f} s")
    print(f"Std. dev. time:         {time_std:.4f} s")
    print(f"Throughput:             {aggregate_throughput:.2f} tok/s")
    print(f"Latency:                {latency_per_token * 1000:.3f} ms/token")

    print("----------------------------------------------")

    print(f"Expected alpha:         {avg_expected_alpha:.4f}")
    print(f"Realized alpha:         {avg_realized_alpha:.4f}")

    print("----------------------------------------------")

    print(f"Accepted draft tokens:  {sum(accepted_counts)}")
    print(f"Proposed draft tokens:  {sum(proposed_counts)}")

    print("==============================================")