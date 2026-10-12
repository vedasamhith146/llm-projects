#!/usr/bin/env python3
"""
Profile the supplied GPT-2 implementation with and without its KV cache.

Run this next to model.py, model_kvcache.py, and gpt2_step_11500.pt, or pass
--uncached, --cached, and --checkpoint explicitly.  The script deliberately
does not edit either model.  It measures synchronized wall-clock time on MPS,
where an ordinary timer can otherwise measure only command enqueue time.

Results are divided into:
  * end-to-end generation and per-token decode latency;
  * prefill and model-forward latency;
  * sampling/postprocessing latency (greedy by default);
  * torch.cat cache-update time (instrumented separately);
  * analytical FLOP, cache-byte, and cache-copy estimates; and
  * an optional torch.profiler operator table.

Measured timings are observations on this machine.  FLOPs and bytes are
estimates based on the GPT-2 architecture and are not hardware counters.
Apple MPS does not expose a complete public arithmetic-vs-bandwidth split
through PyTorch, so this script cannot make that attribution exactly.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Optional

import torch
import torch.nn.functional as F


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def clock(device: torch.device) -> float:
    synchronize(device)
    return time.perf_counter()


def elapsed(start: float, device: torch.device) -> float:
    synchronize(device)
    return time.perf_counter() - start


def median_ms(values: list[float]) -> float:
    return 1000.0 * statistics.median(values)


def percentile_ms(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0] * 1000.0
    index = (len(ordered) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    value = ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return value * 1000.0


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false")
    return device


def load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dict in {checkpoint}, got {type(state)}")
    cleaned = {}
    for key, value in state.items():
        cleaned[key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. Missing={missing}; unexpected={unexpected}")
    model.to(device).eval()


def make_model(module: ModuleType, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    config = module.GPT2Config(vocab_size=50304)
    model = module.GPT2(config)
    load_checkpoint(model, checkpoint, device)
    return model


def token_ids(length: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    # Avoid tokenizer/CPU-to-MPS noise in the measured region.  The repeated
    # pattern is only a deterministic prompt; use --prompt for real text.
    values = [(i * 7919 + 17) % min(vocab_size, 50304) for i in range(length)]
    return torch.tensor(values, dtype=torch.long, device=device).unsqueeze(0)


def prompt_ids(prompt: Optional[str], length: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    if prompt:
        try:
            import tiktoken

            ids = tiktoken.get_encoding("gpt2").encode(prompt)
            if not ids:
                raise ValueError("prompt tokenized to zero tokens")
            ids = (ids * ((length + len(ids) - 1) // len(ids)))[:length]
            return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        except Exception as exc:
            print(f"warning: tokenizer unavailable ({exc}); using deterministic token IDs", file=sys.stderr)
    return token_ids(length, vocab_size, device)


@contextlib.contextmanager
def no_grad() -> Iterator[None]:
    with torch.no_grad():
        yield


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    # argmax avoids random-number and top-k/top-p differences between runs.
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def clear_cache(model: torch.nn.Module) -> None:
    clear = getattr(model, "clear_kv_cache", None)
    if clear is not None:
        clear()


@dataclass
class StepResult:
    prefill_ms: float
    decode_ms: list[float]
    model_ms: list[float]
    sampling_ms: list[float]
    total_ms: float
    cache_update_ms: list[float]


class CacheUpdateTimer:
    """Wrap KVCache.update without changing the cache algorithm."""

    def __init__(self, model: torch.nn.Module, device: torch.device, sync_each: bool):
        self.device = device
        self.sync_each = sync_each
        self.times: list[float] = []
        self.originals: list[tuple[Any, Callable[..., Any]]] = []
        for block in getattr(model.transformer, "h", []):
            cache = getattr(block.attn, "kvcache", None)
            if cache is None:
                continue
            original = cache.update

            def wrapped(key: torch.Tensor, value: torch.Tensor, original=original) -> None:
                if self.sync_each:
                    synchronize(self.device)
                start = time.perf_counter()
                original(key, value)
                if self.sync_each:
                    synchronize(self.device)
                self.times.append(time.perf_counter() - start)

            cache.update = wrapped
            self.originals.append((cache, original))

    def restore(self) -> None:
        for cache, original in self.originals:
            cache.update = original


def run_generation(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    new_tokens: int,
    device: torch.device,
    cached: bool,
    instrument_cache: bool = False,
) -> StepResult:
    clear_cache(model)
    update_timer = CacheUpdateTimer(model, device, sync_each=instrument_cache) if cached else None
    generated = tokens
    decode_ms: list[float] = []
    model_ms: list[float] = []
    sampling_ms: list[float] = []

    with no_grad():
        total_start = clock(device)
        if cached:
            start = clock(device)
            logits, _ = model(generated, start_pos=0)
            prefill_ms = elapsed(start, device) * 1000.0
            next_token = greedy_sample(logits)
            generated = torch.cat((generated, next_token), dim=1)
        else:
            # This is only a warmup/prefill measurement.  The uncached loop
            # below intentionally forwards the entire growing prefix.
            start = clock(device)
            logits, _ = model(generated)
            prefill_ms = elapsed(start, device) * 1000.0
            next_token = greedy_sample(logits)

        for _ in range(new_tokens):
            if cached:
                step_input = next_token
                start = clock(device)
                logits, _ = model(step_input, start_pos=generated.size(1) - 1)
            else:
                step_input = torch.cat((generated, next_token), dim=1)
                start = clock(device)
                logits, _ = model(step_input)
            forward_s = elapsed(start, device)

            sample_start = clock(device)
            next_token = greedy_sample(logits)
            sample_s = elapsed(sample_start, device)

            decode_ms.append((forward_s + sample_s) * 1000.0)
            model_ms.append(forward_s * 1000.0)
            sampling_ms.append(sample_s * 1000.0)
            generated = torch.cat((generated, next_token), dim=1)
        total_ms = elapsed(total_start, device) * 1000.0

    if update_timer is not None:
        update_timer.restore()
    return StepResult(prefill_ms, decode_ms, model_ms, sampling_ms, total_ms, update_timer.times if update_timer else [])


def warmup(model: torch.nn.Module, ids: torch.Tensor, device: torch.device, cached: bool) -> None:
    with no_grad():
        clear_cache(model)
        synchronize(device)
        if cached:
            logits, _ = model(ids, start_pos=0)
            model(logits.new_zeros((1, 1), dtype=torch.long), start_pos=ids.size(1))
        else:
            model(ids)
            model(torch.cat((ids, ids[:, :1]), dim=1))
        synchronize(device)


def architecture(model: torch.nn.Module) -> tuple[int, int, int, int, int]:
    c = model.config
    return c.n_layer, c.n_head, c.n_embd, c.block_size, c.vocab_size


def estimates(model: torch.nn.Module, context: int, dtype_bytes: int) -> dict[str, float]:
    layers, heads, d, _, _ = architecture(model)
    dff = 4 * d
    # Multiply-add = 2 FLOPs.  Attention uses 2*T^2*d for QK and the same
    # again for AV.  These are architecture estimates, not profiler counters.
    uncached = layers * (6 * context * d * d + 4 * context * context * d + 2 * context * d * d + 4 * context * d * dff)
    cached = layers * (6 * d * d + 4 * context * d + 2 * d * d + 4 * d * dff)
    cache_bytes = 2 * layers * context * d * dtype_bytes
    # torch.cat allocates a new [past + 1] tensor and copies old K and V.
    cat_copy_bytes = 2 * layers * max(context - 1, 0) * d * dtype_bytes
    return {"uncached_flops": float(uncached), "cached_flops": float(cached), "cache_bytes": float(cache_bytes), "cat_copy_bytes": float(cat_copy_bytes), "heads": float(heads)}


def print_operator_profile(model: torch.nn.Module, ids: torch.Tensor, device: torch.device, cached: bool, activities: list[Any]) -> None:
    clear_cache(model)
    with torch.no_grad():
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
            if cached:
                logits, _ = model(ids, start_pos=0)
                model(torch.argmax(logits[:, -1:, :], dim=-1), start_pos=ids.size(1))
            else:
                logits, _ = model(ids)
                nxt = torch.argmax(logits[:, -1:, :], dim=-1)
                model(torch.cat((ids, nxt), dim=1))
        synchronize(device)
    print("\nPyTorch operator profile (one prefill + one next-token step):")
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=25))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uncached", type=Path, default=Path("model.py"))
    parser.add_argument("--cached", type=Path, default=Path("model_kvcache.py"))
    parser.add_argument("--checkpoint", type=Path, default=Path("gpt2_step_11500.pt"))
    parser.add_argument("--device", default="auto", help="auto, mps, cpu, or cuda")
    parser.add_argument("--contexts", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prompt", default=None, help="optional real text; repeated/truncated to each context length")
    parser.add_argument("--profile-ops", action="store_true", help="also print torch.profiler operator timing")
    parser.add_argument("--profile-context", type=int, default=None)
    parser.add_argument("--dtype-bytes", type=int, default=4, help="4 for float32, 2 for float16; match checkpoint/model dtype")
    args = parser.parse_args()

    for path in (args.uncached, args.cached, args.checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    device = choose_device(args.device)
    uncached_mod = load_module(args.uncached.resolve(), "profile_uncached_model")
    cached_mod = load_module(args.cached.resolve(), "profile_cached_model")
    uncached = make_model(uncached_mod, args.checkpoint, device)
    cached = make_model(cached_mod, args.checkpoint, device)
    if architecture(uncached) != architecture(cached):
        raise RuntimeError(f"Architecture mismatch: {architecture(uncached)} vs {architecture(cached)}")
    layers, heads, embd, block_size, vocab = architecture(cached)
    contexts = [n for n in args.contexts if 1 <= n < block_size]
    if not contexts:
        raise ValueError(f"No context fits block_size={block_size}")

    print(f"device={device}; torch={torch.__version__}; MPS available={torch.backends.mps.is_available()}")
    print(f"architecture: layers={layers}, heads={heads}, hidden={embd}, block_size={block_size}, vocab={vocab}")
    print("timing: synchronized wall clock; generation uses argmax; no printing/token decoding in measured loops")
    print("note: MPS timings include device execution after synchronization, but PyTorch cannot expose exact Apple GPU bandwidth/ALU attribution")

    for context in contexts:
        ids = prompt_ids(args.prompt, context, vocab, device)
        for _ in range(args.warmup):
            warmup(uncached, ids, device, cached=False)
            warmup(cached, ids, device, cached=True)
        uncached_runs: list[StepResult] = []
        cached_runs: list[StepResult] = []
        for _ in range(args.repeats):
            uncached_runs.append(run_generation(uncached, ids, args.new_tokens, device, cached=False))
            cached_runs.append(run_generation(cached, ids, args.new_tokens, device, cached=True, instrument_cache=False))
        u = uncached_runs[-1]
        c = cached_runs[-1]
        u_decode = [x / 1000.0 for r in uncached_runs for x in r.decode_ms]
        c_decode = [x / 1000.0 for r in cached_runs for x in r.decode_ms]
        u_model = [x / 1000.0 for r in uncached_runs for x in r.model_ms]
        c_model = [x / 1000.0 for r in cached_runs for x in r.model_ms]
        u_sample = [x / 1000.0 for r in uncached_runs for x in r.sampling_ms]
        c_sample = [x / 1000.0 for r in cached_runs for x in r.sampling_ms]
        e = estimates(cached, context, args.dtype_bytes)
        print(f"\n=== context={context}, generated={args.new_tokens}, repeats={args.repeats} ===")
        print(f"prefill median ms: uncached={median_ms([r.prefill_ms/1000 for r in uncached_runs]):.3f} cached={median_ms([r.prefill_ms/1000 for r in cached_runs]):.3f} (same full prompt work)")
        print(f"decode/token median ms: uncached={median_ms(u_decode):.3f} cached={median_ms(c_decode):.3f} speedup={median_ms(u_decode)/median_ms(c_decode):.2f}x")
        print(f"decode/token p90 ms:    uncached={percentile_ms(u_decode,.90):.3f} cached={percentile_ms(c_decode,.90):.3f}")
        print(f"model forward median ms: uncached={median_ms(u_model):.3f} cached={median_ms(c_model):.3f}")
        print(f"sampling median ms:     uncached={median_ms(u_sample):.3f} cached={median_ms(c_sample):.3f}")
        print(f"full measured run ms:   uncached={statistics.median([r.total_ms for r in uncached_runs]):.3f} cached={statistics.median([r.total_ms for r in cached_runs]):.3f}")
        print(f"estimated decode FLOPs:  uncached={e['uncached_flops']:.3e} cached={e['cached_flops']:.3e} ratio={e['uncached_flops']/e['cached_flops']:.2f}x")
        print(f"estimated KV bytes:      {e['cache_bytes']/1024**2:.3f} MiB; one-step cat old-data copy={e['cat_copy_bytes']/1024**2:.3f} MiB")

        # Synchronizing around every torch.cat is intentionally a separate,
        # intrusive measurement: it gives a bound for cache-update work but
        # should not be confused with normal end-to-end timing.
        timed = run_generation(cached, ids, 1, device, cached=True, instrument_cache=True)
        if timed.cache_update_ms:
            print(f"instrumented KVCache.update: {len(timed.cache_update_ms)} calls; total synchronized cat/update={sum(timed.cache_update_ms)*1000:.3f} ms (intrusive)")
        else:
            print("instrumented KVCache.update: no cache update calls observed")

        if args.profile_ops and (args.profile_context is None or args.profile_context == context):
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "mps":
                activities.append(torch.profiler.ProfilerActivity.MPS)
            elif device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            print_operator_profile(uncached, ids, device, False, activities)
            print_operator_profile(cached, ids, device, True, activities)

    print("\nInterpretation:")
    print("- If model-forward time dominates the saved time, KV caching is removing repeated Transformer work, not merely speeding up sampling.")
    print("- Cached forward time should grow roughly with context because the new query still reads historical K/V; it is not constant-time attention.")
    print("- A large instrumented cat/update time identifies the current torch.cat implementation as a measurable overhead; preallocation can reduce it.")
    print("- Operator tables identify PyTorch operators, but MPS does not provide a complete public compute-throughput vs memory-bandwidth attribution here.")


if __name__ == "__main__":
    main()
