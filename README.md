# LLM Projects

A collection of 16 hands-on LLM engineering projects.

Each folder (`01-...` to `16-...`) is one self-contained mini-project that focuses on a single core concept:
build → plot → break → learn.

---

## Project Index

### 01 — Tokenization & Embeddings
Build a simple tokenizer (e.g. BPE), map text to token IDs, and compare one-hot vs learned embeddings using cosine similarity.

### 02 — Positional Embeddings
Experiment with different positional schemes (sinusoidal, learned, RoPE, ALiBi) and see how removing positions breaks attention.

### 03 — Self-Attention & Multi-Head Attention
Implement dot-product attention for a single token, scale to multi-head attention, and visualize attention weight heatmaps with causal masking.

### 04 — Transformers, QKV & Stacking
Combine attention, residual connections, and normalization into a transformer block, stack multiple blocks into a “mini-former”, and experiment with Q/K/V roles.

### 05 — Sampling Parameters: Temperature / Top-k / Top-p
Build a small sampling playground to see how temperature, top-k, and top-p change entropy, diversity, and repetition in generated text.

### 06 — KV Cache (Fast Inference)
Implement key/value caching for autoregressive generation and measure the speed/memory trade-offs for different sequence lengths.

### 07 — Long-Context Tricks
Explore sliding-window or other long-context attention tricks, measure loss/perplexity vs context length, and find where performance collapses.

### 08 — Mixture of Experts (MoE)
Add a simple router with two or more experts, visualize expert usage, and compare sparse vs dense compute (FLOPs and quality).

### 09 — Grouped Query Attention (GQA)
Convert a mini-transformer to grouped query attention and measure speed/latency vs standard multi-head attention.

### 10 — Normalization & Activations
Implement LayerNorm, RMSNorm, GELU, SwiGLU, etc., and ablate them to see the effect on training stability, loss, and activation distributions.

### 11 — Pretraining Objectives
Compare masked LM, causal LM, and prefix LM on toy text data, looking at loss curves and sample quality from each objective.

### 12 — Finetuning vs Instruction Tuning vs RLHF
Fine-tune a base model on tasks, instruction-tune with formatted prompts, and run a tiny RLHF-style loop with a reward model + PPO steps.

### 13 — Scaling Laws & Model Capacity
Train tiny/small/medium models on the same dataset and plot loss vs model size, as well as VRAM usage, throughput, and training time.

### 14 — Quantization
Apply post-training quantization (PTQ) and/or quantization-aware training (QAT), export to common formats, and plot accuracy vs model size.

### 15 — Inference / Training Stacks
Run the same model across different inference/training stacks (e.g. DeepSpeed, vLLM, ExLlama) and compare throughput, VRAM, and latency.

### 16 — Synthetic Data
Generate synthetic datasets, add noise and deduplication, create eval splits, and compare learning curves on real vs synthetic data.

---

## Layout

```text
llm-projects/
  01-tokenization-embeddings/
  02-positional-embeddings/
  ...
  16-synthetic-data/
  README.md
