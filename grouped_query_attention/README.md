# Grouped-Query Attention: KV-Cache and Decode-Latency Experiments

This project implements and benchmarks several attention configurations that
trade key/value (KV) cache size against decoding latency:

- Multi-Head Attention (MHA): one key head and one value head per query head.
- Grouped-Query Attention (GQA): groups of query heads share one key/value
  head.
- Multi-Query Attention (MQA): all query heads share one key/value head.

The repository contains both a small attention-module benchmark and a
randomly initialized GPT-style model with a KV cache. The experiments are
runtime experiments, not language-quality evaluations: there is no trained
checkpoint, tokenizer pipeline, validation dataset, or training script in
this directory.

## questions

The code explores three practical questions:

1. How does reducing the number of KV heads change theoretical KV-cache
   storage?
2. Does the local attention implementation become faster when fewer KV heads
   are used?
3. Does that behavior remain visible in a full model's autoregressive decode
   loop?

The measured results are specific to the implementation, PyTorch version,
device, tensor shapes, and timing methodology described below. They should not
be interpreted as a claim that GQA is universally faster than MHA.

## Attention architectures

Let `Hq` be the number of query heads and `Hkv` the number of key/value heads.
The grouping factor is:

```text
group_size = Hq / Hkv
```

| Architecture | Query heads | KV heads | Group size | KV-cache implication |
|---|---:|---:|---:|---|
| MHA | `Hq` | `Hq` | 1 | Every query head has its own K/V pair. |
| GQA | `Hq` | `1 < Hkv < Hq` | `Hq/Hkv` | Several query heads share each K/V pair. |
| MQA | `Hq` | 1 | `Hq` | All query heads share one K/V pair. |

In this project, GQA is implemented by producing fewer K/V heads and then
repeating each cached KV head across its query-head group before attention.
MQA is the `n_kv_head=1` case of the same implementation.

## KV-cache memory

The theoretical KV-cache storage estimate is:

```text
KV-cache memory in bytes =
  2 * number_of_layers * batch_size * sequence_length
    * number_of_KV_heads * head_dimension * bytes_per_element
```

The terms are:

- `2`: one tensor for keys and one for values.
- `number_of_layers`: the number of Transformer blocks.
- `batch_size`: number of sequences cached at once.
- `sequence_length`: number of tokens stored in the cache.
- `number_of_KV_heads`: `Hkv`.
- `head_dimension`: embedding dimension divided by query-head count.
- `bytes_per_element`: for example, 4 for FP32 and 2 for FP16/BF16.

This estimates KV-cache storage only. It does not include model weights,
activations, temporary attention tensors, logits, allocator overhead, or
other process memory. A smaller cache can reduce memory traffic, but the
latency effect depends on the attention kernel, tensor layout, synchronization,
and hardware.

For the model-level configuration in `decode_latency.py`, the defaults are
12 layers, 12 query heads, embedding dimension 768, and therefore a head
dimension of 64. At batch size 1, sequence length 1024, and FP32, the cache
estimate is:

| Configuration | KV heads | Group size | Cache size | Reduction vs MHA |
|---|---:|---:|---:|---:|
| MHA | 12 | 1 | 72.00 MiB | 0% |
| GQA | 6 | 2 | 36.00 MiB | 50% |
| GQA | 4 | 3 | 24.00 MiB | 66.7% |
| GQA | 2 | 6 | 12.00 MiB | 83.3% |
| MQA | 1 | 12 | 6.00 MiB | 91.7% |

The model-level decode benchmark actually grows the cache from the prompt
through generated tokens. The table above is a normalized calculation at a
fixed sequence length, not a measurement of total process memory.

## Project structure

| File | Purpose |
|---|---|
| `benchmark.py` | One attention module with configurable query/KV heads; measures cached single-token attention latency. |
| `decode_latency.py` | Full 12-layer GPT-style model, KV cache, random-token prompt, and autoregressive decode benchmark. |
| `model.py` | Earlier/duplicate GPT-style implementation with the same broad design, but currently not constructible because its QKV Linear dimension uses `/` instead of integer division. |
| `plots.py` | Runs an ablation over 32 query heads and KV-head counts `[1, 2, 4, 8, 16, 32]`, then displays a plot. It does not save a plot file. |
| `toy_mla.py` | Small latent K/V compression toy module; it is not used by the MHA/GQA/MQA benchmarks. |

There is no project-local `README`, `requirements.txt`, `pyproject.toml`,
training script, checkpoint, CSV result file, or saved plot before this
README. The Python dependencies used by the scripts are PyTorch, NumPy, and
Matplotlib; `decode_latency.py` imports NumPy but does not use it in the
benchmark path.

## Model implementation

`decode_latency.py` defines the following default GPT-style configuration:

```text
block_size = 1024
vocab_size = 50304
n_layer = 12
n_head = 12
n_embd = 768
n_kv_head = 4
```

The model ties the token embedding and language-model head weights. Each
attention block projects full-width queries and reduced-width keys/values,
stores the latter in a per-layer cache, repeats cached KV heads with
`torch.repeat_interleave`, and calls
`torch.nn.functional.scaled_dot_product_attention`.

The cache appends each new token using `torch.cat`. This is simple and easy to
inspect, but repeated concatenation reallocates and copies the cache as the
sequence grows; it is not an optimized production KV-cache implementation.

## Running the experiments

Run from the repository root, or change into this directory first:

```bash
cd grouped_query_attention
python3 benchmark.py
python3 decode_latency.py
python3 plots.py
```

### `benchmark.py`: attention-module latency

The default command measures a single-token query against a synthetic cache
using:

- `n_head=8`, `n_kv_head=8` for MHA and `n_kv_head=2` for GQA;
- batch size 64;
- cached history length 1024;
- embedding dimension 768;
- 10 warm-up iterations and 100 measured iterations;
- `torch.no_grad()`;
- CUDA if available, otherwise Apple MPS if available, otherwise CPU;
- device synchronization before and after timing on CUDA/MPS.

The benchmark includes the attention module's QKV projection, attention
operation, output projection, and cache reads. It does not run the full GPT
model, and it does not measure prefill or tokenization.

For the safe local measurement reported below, the same function was called
with batch size 1 and history length 128 to avoid an unnecessarily large CPU
workload. That run still used the script's 10 warm-ups and 100 measured
iterations.

### `decode_latency.py`: full-model decode latency

`benchmark_decode` constructs a fresh randomly initialized 115,026,432-
parameter GPT-style model for each KV-head configuration. It:

1. Runs a random prompt through the model to fill the cache.
2. Performs 10 unmeasured greedy-generation warm-up steps.
3. Clears and refills the cache.
4. Measures `gen_tokens` autoregressive steps and reports milliseconds per
   generated token.

The default benchmark uses batch size 1, prompt length 128, 100 generated
tokens, and KV-head counts `[12, 6, 4, 2, 1]`. The prompt prefill is not part
of the timed interval. The timed loop includes model forward passes, logits,
`argmax`, cache updates, and KV repetition. CUDA synchronization is performed
around the measured interval when CUDA is active; this execution environment
was CPU-only.

The model is randomly initialized and the prompt consists of random token
IDs. These measurements describe implementation runtime only. They do not
measure generation quality or compare trained models.

### `plots.py`: KV-head ablation

The script fixes `n_head=32` and evaluates `n_kv_head` in
`[1, 2, 4, 8, 16, 32]`, corresponding to group sizes `[32, 16, 8, 4, 2,
1]`. It calls the default `benchmark.run_benchmark`, so each point uses batch
size 64, history length 1024, 10 warm-ups, and 100 measured iterations. It
then calls `plt.show()`; no image is written to disk.

I did not run `plots.py` with its default workload during this audit because
it launches six large CPU benchmark cases and only displays the resulting
figure. The same ablation function was evaluated safely with batch size 1 and
history length 128 using the underlying benchmark function, producing the
table below.

## Results collected during this audit

Execution environment:

- Python 3.13.5
- PyTorch 2.10.0
- CPU-only: CUDA unavailable and MPS unavailable
- 10 warm-up iterations and 100 measured iterations for `benchmark.py`
- Results below are single benchmark invocations, not repeated-run means or
  confidence intervals.

### Synthetic attention-module latency

Safe run: `n_embd=768`, batch size 1, cached history 128, 8 query heads,
100 measured iterations. “Relative speedup” is MHA latency divided by the
configuration latency; values below 1 mean the configuration was slower than
the MHA baseline.

| Attention case | Query heads | KV heads | Group size | Latency (ms) | Relative speedup vs MHA |
|---|---:|---:|---:|---:|---:|
| MHA | 8 | 8 | 1 | 0.111609 | 1.0000× |
| GQA | 8 | 2 | 4 | 0.199100 | 0.5606× |

In this CPU run, the GQA case was slower. The implementation reshapes the
queries into an extra group dimension and relies on broadcasting in scaled
dot-product attention; that layout is not evidence that fewer KV heads will
always improve runtime.

### Synthetic 32-query-head ablation

Safe run: `n_head=32`, `n_embd=768`, batch size 1, cached history 128. The
MHA baseline is the `n_kv_head=32` row.

| KV heads | Group size | Latency (ms) | Relative speedup vs MHA |
|---:|---:|---:|---:|
| 1 | 32 | 0.190994 | 0.5694× |
| 2 | 16 | 0.253119 | 0.4296× |
| 4 | 8 | 0.254522 | 0.4273× |
| 8 | 4 | 0.260142 | 0.4180× |
| 16 | 2 | 0.278860 | 0.3900× |
| 32 | 1 | 0.108747 | 1.0000× |

The non-monotonic values and the slower GQA cases demonstrate why the
theoretical cache reduction should not be converted directly into a latency
claim. No profiler evidence was collected, so this README does not attribute
the result to memory bandwidth.

### Full-model decode latency

Safe run: `decode_latency.benchmark_decode`, CPU, batch size 1, prompt length
16, 5 generated tokens, with a fresh random 12-layer model per row. This is a
shortened run relative to the script defaults of prompt length 128 and 100
generated tokens.

| Attention case | Query heads | KV heads | Group size | Decode latency (ms/token) | Relative speedup vs MHA |
|---|---:|---:|---:|---:|---:|
| MHA | 12 | 12 | 1 | 10.345167 | 1.0000× |
| GQA | 12 | 6 | 2 | 10.039442 | 1.0305× |
| GQA | 12 | 4 | 3 | 9.504558 | 1.0884× |
| GQA | 12 | 2 | 6 | 9.587642 | 1.0790× |
| MQA | 12 | 1 | 12 | 9.440075 | 1.0959× |

These results show a small improvement in this particular short CPU run, but
they are not directly comparable with the synthetic attention-module table:
the full-model benchmark includes all Transformer blocks, MLPs, projections,
logits, greedy selection, and cache management. It also uses a different
prompt length, token count, and head count.

## Quality evaluation status

### Validation loss

Not implemented as an experiment. The GPT class accepts optional `targets` and
can calculate token-level cross-entropy for a supplied batch, but the project
contains no validation dataset loader, tokenizer, checkpoint-loading path, or
evaluation loop. The random-token prompt used by `decode_latency.py` cannot
produce a meaningful validation-loss result.

### Perplexity

Not implemented. If a future evaluation computes average token-level
cross-entropy using natural logarithms, perplexity can be reported as:

```text
perplexity = exp(validation_loss)
```

No perplexity number is reported here.

### Batch throughput

Not implemented. The scripts report latency for a single generated token or
an average attention call; they do not measure tokens/second over a defined
batch-throughput workload. Throughput should be measured separately rather
than inferred from the latency tables.

## Implementation limitations and known issues

The following limitations are present in the inspected code:

- `model.py` is not constructible with its default configuration because
  `CausalSelfAttention` passes a floating-point output dimension to
  `nn.Linear` via `/`. `decode_latency.py` contains a working duplicate that
  uses integer division `//`. This README does not silently fix the duplicate
  implementation.
- `decode_latency.py` repeats the full cached KV tensors with
  `torch.repeat_interleave` before every attention call. This materializes
  repeated KV data and can obscure the benefit of a compact cache.
- `KVCache.update` appends with `torch.cat` on every token, causing repeated
  allocation and copying as the sequence grows.
- The full-model benchmark uses random initialization and random input IDs;
  it provides no predictive-quality evidence.
- `benchmark.py` and `plots.py` benchmark only an attention module, not the
  full GPT model. Their results must not be conflated with model-level decode
  latency.
- `plots.py` displays a plot but does not save it, and its default workload is
  substantially larger than the safe CPU run reported above.
- CUDA synchronization is handled in the benchmark paths, but there is no
  repeated-run aggregation, variance report, profiler trace, or kernel-level
  attribution.
- `benchmark.py` uses a different default head count (`n_head=8`) from the
  GPT-style model (`n_head=12`), while `plots.py` uses 32 query heads. The
  workloads are therefore not interchangeable.
- The code assumes that `n_head` is divisible by `n_kv_head`; the benchmark
  configurations used here satisfy that requirement, but there is no explicit
  validation error for invalid combinations.

## Future work

Useful next steps would be:

1. Replace repeated `torch.cat` with preallocated or paged KV-cache storage.
2. Use a native grouped-query attention kernel or an attention API that keeps
   KV heads compact instead of materializing repeated keys and values.
3. Add a real tokenizer, checkpoint-loading path, validation dataset, and
   deterministic loss/perplexity evaluation.
4. Add repeated trials, variance summaries, CPU thread controls, and profiler
   traces for each workload.
5. Save ablation plots and machine-readable result tables.
6. Add a separate batch-throughput benchmark with explicit tokens/second and
   batch-size reporting.
7. Consolidate `model.py` and `decode_latency.py` so there is one tested model
   implementation.

## References

The implementation is a compact educational experiment. No external paper or
reference is cited in the existing project files. For background when
extending it, the relevant concepts are the original Transformer multi-head
attention mechanism and grouped/multi-query attention variants; any future
claims should be tied to the exact kernel and hardware being measured.
