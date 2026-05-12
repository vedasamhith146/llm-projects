# GPT-2 124M Pretrained on Fineweb-edu

Reproduction of GPT-2 (124M parameters ) pretrained on the 
[Fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset
using a single NVIDIA RTX 5090 GPU

This project closely follows the architecture and training methodology from
[Andrej Karpathy's nanoGPT](https://github.com/karpathy/build-nanogpt), adapted
for single - GPU training

## Results

Evaluated at training step **11,500** out of planned 19073.

| Metric | Value | Notes |
|--------|-------|-------|
| **Train Loss** | 3.1252 | Measured on a seen training shard |
| **Validation Loss** | 3.1580 | Measured on held-out shard `099` (unseen during training) |
| **Perplexity** | 23.52 | `exp(val_loss)` |
| **HellaSwag Accuracy** | 28.24% | Full validation split, 10,042 examples |
| **Train–Val Gap** | 0.0328 | Indicates healthy generalization (no overfitting) |

### Comparison with Baselines

| Model | Val Loss | HellaSwag |
|-------|----------|-----------|
| Random Baseline | ~10.82 | 25.00% |
| **This Model (step 11,500)** | **3.1580** | **28.24%** |
| GPT-2 124M (OpenAI) | ~3.11 | ~29.55% |

The model achieves performance close to the original GPT-2 124M while being
trained for only **~60% of the planned steps** (11,500 / 19,073).

---

## Model Architecture 

| Hyperparameter | Value |
|----------------|-------|
| Architecture | GPT-2 (decoder-only transformer) |
| Parameters | ~124M |
| Layers | 12 |
| Attention Heads | 12 |
| Embedding Dimension | 768 |
| Context Length | 1024 |
| Vocabulary Size | 50,304 (padded from 50,257 for efficiency) |
| Tokenizer | GPT-2 BPE (`tiktoken`) |
| Activation | GELU (tanh approximation) |
| Position Embeddings | Learned |


## Training Configuration

| Item | Value |
|------|-------|
| Dataset | FineWeb-Edu `sample-10BT` |
| Total Tokens Available | ~10B |
| Tokens Seen During Training | ~6.03B |
| Steps Completed | 11,500 / 19,073 |
| Effective Batch Size | 524,288 tokens/step |
| Micro Batch Size (B) | 64 |
| Sequence Length (T) | 1024 |
| Gradient Accumulation Steps | 8 |
| Optimizer | AdamW (β1=0.9, β2=0.95, ε=1e-8) |
| Weight Decay | 0.1 |
| Peak Learning Rate | 6e-4 |
| LR Schedule | Cosine decay with warmup |
| Warmup Steps | 715 |
| Min LR | 6e-5 (10% of peak) |
| Gradient Clipping | 1.0 |
| Precision | Mixed (BF16 autocast) |
| Compilation | `torch.compile` |

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 5090 (32 GB VRAM) |
| RAM | 60 GB |




