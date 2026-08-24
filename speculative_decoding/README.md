# Speculative Decoding Implementation Study

## Overview

Started with reading the paper:

> **"Fast Inference from Transformers via Speculative Decoding"**

Implemented the speculative decoding algorithm from scratch.

This was a very interesting algorithm to implement because there are several edge cases that need to be handled carefully. Writing the implementation helped me understand where speculative decoding provides benefits and where it breaks down.

---

## Model Selection

For this implementation, I selected:

| Model | Role |
|---|---|
| Qwen2.5-1.5B-Instruct-4bit | Target Model ($M_p$) |
| Qwen2.5-0.5B-Instruct-4bit | Draft Model ($M_q$) |

Initially, I knew this might not be the optimal choice.

The paper mentions:

> "Choosing the draft model to be around two orders of magnitude smaller than the target model usually performed best."

They also mention that the cost coefficient:

$$
c = \frac{\text{time taken for one run of draft model }(M_q)}
{\text{time taken for one run of target model }(M_p)}
$$

was always less than **0.05** in their experiments.

The value of `c` has a direct impact on the achievable speedup of speculative decoding.

However, considering the available hardware:

- MacBook M3
- 8GB RAM

these models seemed like reasonable choices.

Even if no speedup is observed, this experiment will help understand **where speculative decoding fails and why**.

---

# Calculating Cost Coefficient (c)

The first step is calculating:

$$
c = \frac{T_q}{T_p}
$$

where:

- $T_q$ = time taken by draft model
- $T_p$ = time taken by target model

For my setup:

- Averaged across 5 independent runs
- Each model timing averaged over 100 executions

Obtained:

$$
\boxed{c = 0.3328}
$$

This is much higher compared to the paper's experiments where:

$$
c < 0.05
$$

Therefore, the draft model is relatively expensive compared to the target model.

---

# Calculating Acceptance Probability (α)

The next important quantity is:

$$
\alpha
$$

According to Corollary 3.9 from the paper:

If:

$$
\alpha > c
$$

then there exists some value of:

$$
\gamma
$$

for which speculative decoding provides speedup.

---

# How Speculative Sampling Works

Instead of directly sampling:

$$
x \sim p(x)
$$

we first sample:

$$
x \sim q(x)
$$

where:

- $p(x)$ = target model distribution
- $q(x)$ = draft model distribution

The sampled token is accepted if:

$$
q(x) \leq p(x)
$$

Otherwise, it is rejected with probability:

$$
1-\frac{p(x)}{q(x)}
$$

and resampled from:

$$
p'(x)=
\frac{\max(0,p(x)-q(x))}
{\sum_x \max(0,p(x)-q(x))}
$$

The paper proves that this process still produces:

$$
x \sim p(x)
$$

Refer to Appendix A.1 of the paper for the proof.

---

# What is α?

The value:

$$
\alpha = E(\beta)
$$

where:

## β - Acceptance Rate for a Prefix

For a given prefix, β represents the probability that a randomly sampled draft token will be accepted.

Mathematically:

$$
\beta = \sum_x \min(p(x),q(x))
$$

Therefore:

$$
\alpha = E(\sum_x \min(p(x),q(x)))
$$

α measures how closely the draft model approximates the target model.

Higher α means:

- Draft model predictions are closer to target model predictions
- More tokens are accepted
- Higher possible speedup

---

# Measuring α

To estimate α:

1. Calculate β for multiple prefixes.
2. Average β values.

I evaluated α across:

- Factual prompts
- Creative writing prompts
- Code completion prompts
- Explanatory prompts

For each:

- Generated 100 tokens
- Calculated β values
- Averaged the results

Obtained:

$$
\boxed{\alpha = 0.6184}
$$

---

# Condition Check

We have:

$$
\alpha = 0.6184
$$

and:

$$
c = 0.3328
$$

Therefore:

$$
\alpha > c
$$

Hence according to Corollary 3.9:

There exists a value of γ where speculative decoding should provide improvement.

---

# Choosing γ

The theoretical speedup formula from the paper:

$$
Speedup =
\frac{1-\alpha^{\gamma+1}}
{(1-\alpha)(\gamma c+1)}
$$

Testing different γ values:

## γ = 1

$$
Speedup =
\frac{1-(0.6184)^2}
{(1-0.6184)(1(0.3328)+1)}
$$

Result:

$$
\boxed{1.214x}
$$

Expected:

**21.4% faster than target model alone**

---

## γ = 2

$$
\boxed{1.201x}
$$

---

## γ = 3

$$
\boxed{1.119x}
$$

---

## γ = 4

$$
\boxed{1.022x}
$$

Almost no improvement.

---

## γ = 5

$$
\boxed{0.928x}
$$

Slower than target model alone.

---

These are theoretical speedups.

Next step is validating experimentally.

---

# Experimental Speedup Measurement

For:

$$
\gamma = 1
$$

I selected the following prompt:

```
You are a creative storyteller. Write a detailed, 150-word story about a robot who discovers a hidden garden on a deserted planet. Describe the sights, sounds, and emotions.
```

Generated:

- 150 tokens
- Averaged execution time over 5 runs

Results:

| Method | Time |
|---|---:|
| Target Model | 1.8380s |
| Speculative Decoding | 1.9131s |

Measured speedup:

$$
\frac{1.8380}{1.9131}
$$

$$
\boxed{0.9607x}
$$

Therefore speculative decoding was actually slower.

---

# Measured α For This Experiment

For this prompt:

$$
\alpha = 0.5836
$$

Using this value:

$$
Speedup_{theoretical}
=
\frac{1-(0.5836)^2}
{(1-0.5836)(1(0.3328)+1)}
$$

Result:

$$
\boxed{1.188x}
$$

However, experimental speedup was:

$$
0.9607x
$$

The difference occurs because the theoretical equation does not include:

- KV cache management overhead
- Additional memory operations
- Synchronization overhead
- Python implementation overhead

Therefore, for these particular models:

> We should not expect practical speedup.

---

# Investigating α Variation

Next, I measured α for different prompt categories.

## Code Completion

Obtained:

$$
\boxed{\alpha = 0.4146}
$$

For:

- Creative writing
- Explanatory prompts

I also obtained α values around:

$$
0.40
$$

---

# Observation

Previously:

$$
\alpha = 0.6184
$$

was obtained when measuring across:

- Multiple prompts
- γ = 5
- 100 generated tokens

However, during individual experiments:

$$
\alpha \approx 0.40
$$

was observed.

---

# Question

Why is there such a difference?

If α represents the similarity between draft and target models, shouldn't measured α values remain close?

Possible reasons being investigated:

- Difference in token distributions across prefixes
- Different γ values affecting observed acceptance
- Prompt-dependent model alignment
- Sampling randomness
- Measurement methodology differences

Further analysis is required to understand where the discrepancy originates.