# KV Cache for GPT-2-Style Generation

An implementation-focused experiment with autoregressive generation, layer-wise key/value caching, and cache-storage precision on Apple Silicon. The project uses a compact GPT-2-style Transformer and the checkpoint saved as `gpt2_step_11500.pt`.

> The headline result: for the prompt **“Artificial Intelligence”** and 500 generated tokens, cached decoding reached **56.50 tokens/s**, compared with **4.79 tokens/s** without caching — an **11.80× speedup** in this local benchmark.

## What this project studies

During autoregressive generation, each new token attends to the complete prefix. A naïve decoder recomputes the key and value projections for that prefix on every step. KV caching stores those projections once and reuses them, changing the repeated work from “reprocess the growing sequence” to “process the new token and attend to the stored history.”

This repository focuses on three practical questions:

1. How much faster is the existing model when generation reuses a KV cache?
2. How does the cache grow with sequence length?
3. What storage and latency trade-offs appear when cached K/V tensors use FP16, BF16, or INT8 storage?

The project does **not** include a separate cached-vs-uncached numerical-correctness experiment. The comparisons here are performance and storage experiments using the repository’s existing implementations.

## Model and cache design

The model in `model.py` and `model_kvcache.py` uses the following configuration:

| Parameter | Value |
|---|---:|
| Transformer layers | 12 |
| Attention heads | 12 |
| Hidden size | 768 |
| Head dimension | 64 |
| Vocabulary size | 50,304 |
| Maximum position/block size | 1,024 |
| Default model/weight dtype | FP32 |

`model.py` performs the uncached forward pass. `model_kvcache.py` adds a `KVCache` to every attention block:

- the prompt is processed as a prefill pass;
- the resulting keys and values are retained per layer;
- each decode step sends only the newest token through the model;
- `start_pos` keeps positional embeddings aligned with the growing sequence;
- `clear_kv_cache()` resets all layer caches between generations.

The current cache grows dynamically with `torch.cat` along the sequence dimension. This keeps the implementation easy to follow, but repeated appends allocate and copy tensors; those costs are part of the current implementation’s behavior and are not hidden by the benchmarks.

## KV-cache memory equation

For batch size `B`, sequence length `T`, `L` layers, `H` heads, head dimension `D`, and `s` bytes per stored element:

```text
KV bytes = 2 × L × B × H × T × D × s
```

For this model, `L=12`, `H=12`, and `D=64`. At batch size 1 and FP32 storage, the theoretical cost is approximately **0.0703 MiB per cached token**, before allocator behavior and any additional runtime memory. The INT8 experiment also stores FP32 scale metadata, so its total cache is larger than its raw INT8 payload.

## Benchmark results

### Cached versus uncached generation

`benchmark.py` compares the existing `sample.py` and `sample_kvcache.py` generation paths. The recorded workload was:

- prompt: `Artificial Intelligence`
- generated tokens: 500
- device: Apple Silicon MPS
- metric: generated tokens per second

| Path | Throughput |
|---|---:|
| No KV cache | 4.79 tok/s |
| KV cache | 56.50 tok/s |
| Speedup | 11.80× |

This is an end-to-end result for the repository’s sampling path, not a claim about all GPT-2 implementations or all hardware. Sampling settings, PyTorch version, device backend, checkpoint, and prompt length can change the measured ratio.

### Cache growth

`profile_memory.py` reports the MPS allocation difference after generating a cache for the 12-layer, 12-head, hidden-size-768 model. At 1,000 cached tokens, the reported cache growth was approximately **70.39 MiB**. The script’s printed labels say “MB,” but its conversion divides by (1024^2), so **MiB** is the technically accurate unit.

The result is close to the FP32 theoretical estimate of about 70.31 MiB for 1,000 tokens. The small difference is consistent with allocator/runtime effects and the way the profile measures backend allocation rather than isolating every process allocation.

### Cache storage precision on MPS

The precision experiment kept model weights in FP32 and used the following workload:

- device: `mps`
- PyTorch: `2.10.0`
- context length: 512
- generated tokens: 32
- total cached sequence for the final measurement: 544 tokens

| Storage | Status | Cache storage | Raw K/V payload | Prefill | Decode | Total |
|---|---|---:|---:|---:|---:|---:|
| FP32 | supported | 38.25 MiB | 38.25 MiB | 181.913 ms | 15.685 ms/token | 683.823 ms |
| FP16 | supported | 19.125 MiB | 19.125 MiB | 175.174 ms | 16.587 ms/token | 705.956 ms |
| BF16 | supported | 19.125 MiB | 19.125 MiB | 179.381 ms | 15.529 ms/token | 676.298 ms |
| INT8 | supported | 10.160 MiB | 9.563 MiB | 191.093 ms | 20.695 ms/token | 853.348 ms |
| FP8 E4M3 | unsupported | — | — | — | — | — |

The INT8 cache uses per-token/per-head FP32 scales. Its raw payload is therefore smaller than the total cache storage, which includes scale metadata.

These are storage-format comparisons, not native low-precision attention comparisons. For every supported format, stored K/V tensors are converted back to FP32 before attention, and that conversion/dequantization overhead is included in decode timing. The FP8 row was recorded as unsupported because this MPS run raised:

```text
RuntimeError: Undefined type Float8_e4m3fn
```

Within this run, BF16 had the lowest total latency among the supported formats, while INT8 used the least total cache storage but had the highest decode latency because of quantization and dequantization work.

## Interpretation

- KV caching produced the largest performance gain in this project because uncached generation repeatedly forwards an increasingly long sequence.
- Cache memory grows linearly with sequence length and batch size under the current representation.
- FP16 and BF16 approximately halve the K/V payload relative to FP32, but they do not automatically improve end-to-end latency when attention still consumes FP32 tensors.
- INT8 substantially reduces payload size, but scale metadata and conversion overhead matter; smaller storage did not mean faster generation in this measurement.
- MPS memory values are backend-observable allocations. They should not be presented as the total unified memory used by macOS or as a hardware capacity limit.

Batch-size scaling was also explored on MPS, but the run was stopped by the benchmark’s conservative backend-observable allocation budget at batch size 2. Because that measurement is not a total unified-memory measurement, it is intentionally not used as a headline result here.

## Reproduce the experiments

Run commands from the repository directory:

```bash
cd /Users/vedasamhith/llm-projects/kv_cache
```

### Install dependencies

Use a Python environment with a PyTorch build appropriate for your device. The scripts import PyTorch, `tiktoken`, NumPy, and optionally Matplotlib for plots:

```bash
python3 -m pip install torch tiktoken numpy matplotlib
```

The checkpoint is expected at `gpt2_step_11500.pt` unless a script argument overrides its path.

### Throughput comparison

`benchmark.py` uses its in-file workload (`Artificial Intelligence`, 500 generated tokens) and loads both generation modules:

```bash
python3 benchmark.py
```

### Memory-growth profile

The existing profile defaults to 1,000 tokens sampled in steps of 100:

```bash
python3 profile_memory.py
```

### Detailed cached/uncached profile

`profile_kv_cache.py` exposes synchronized timing on MPS and can report prefill, decode, sampling, and cache-update behavior:

```bash
python3 profile_kv_cache.py \
  --device auto \
  --contexts 32 64 128 256 \
  --new-tokens 8 \
  --warmup 2 \
  --repeats 5
```

Useful options include `--prompt`, `--profile-ops`, `--profile-context`, `--uncached`, `--cached`, and `--checkpoint`.

### Cache-precision comparison

The precision script defaults to `fp32,fp16,bf16,fp8,int8`, context length 32, eight new tokens, one warmup, and three repetitions. The following mirrors the longer MPS workload used for the results above:

```bash
python3 benchmark_cache_precision.py \
  --formats fp32,fp16,bf16,fp8,int8 \
  --context-length 512 \
  --new-tokens 32 \
  --warmup 1 \
  --repetitions 3 \
  --device mps \
  --output precision_results.csv \
  --plot
```

The CSV records supported and unsupported formats. When plotting is enabled, the script writes a PNG beside the requested CSV path.

### Interactive generation

The two sampling programs provide simple interactive prompts:

```bash
python3 sample.py
python3 sample_kvcache.py
```

Type `exit` or `quit` to stop. `sample_kvcache.py` clears its cache before each interactive prompt.

## Repository map

| File | Role |
|---|---|
| `model.py` | GPT-2-style Transformer without KV caching |
| `model_kvcache.py` | Same model with per-layer dynamic KV caches |
| `sample.py` | Interactive uncached generation |
| `sample_kvcache.py` | Interactive cached generation |
| `benchmark.py` | Throughput comparison for 500 generated tokens |
| `profile_memory.py` | Simple MPS/CUDA cache-growth profile |
| `profile_kv_cache.py` | Synchronized, configurable timing/profile tool |
| `benchmark_cache_precision.py` | FP32/FP16/BF16/FP8/INT8 cache-storage experiment |
| `visualizer.py` | Visualization helper for benchmark data |
| `gpt2_step_11500.pt` | Model checkpoint used by the scripts |
| `precision_results.csv` | Recorded precision benchmark output |
| `precision_results.png` | Plot generated from the precision results |

## Limitations and next steps

The current implementation is intentionally small and transparent, but several improvements would make it a stronger production-style cache:

- preallocate cache capacity or use a paged/block cache to avoid repeated `torch.cat` allocation and copying;
- separate prompt prefill, decode, sampling, and cache-update timings in every headline comparison;
- add explicit numerical validation against the uncached path before optimizing storage formats;
- test device-native low-precision attention kernels rather than converting every stored format to FP32;
- evaluate longer contexts and multiple batch sizes with a clearly defined memory metric;
- record environment details such as macOS version, hardware, PyTorch build, and sampling configuration alongside each result.

The numbers in this README are one reproducible snapshot of this repository’s checkpoint and MPS environment, not universal hardware benchmarks.
