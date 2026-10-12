"""Compare storage precision for the existing GPT-2 KV-cache path.

Model weights remain FP32.  Only the stored K/V tensors change representation;
attention converts historical K/V back to FP32 before PyTorch SDPA.  Therefore
conversion/dequantization overhead is part of decode timing.
"""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import types
import time
from pathlib import Path

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
FLOAT_FORMATS = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--formats", default="fp32,fp16,bf16,fp8,int8", help="Comma-separated formats (default: %(default)s)")
    p.add_argument("--context-length", type=int, default=32)
    p.add_argument("--new-tokens", type=int, default=8)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    p.add_argument("--checkpoint", type=Path, default=HERE / "gpt2_step_11500.pt")
    p.add_argument("--output", type=Path, default=HERE / "cache_precision_results.csv")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def choose_device(name):
    if name == "auto":
        if torch.backends.mps.is_available(): return torch.device("mps")
        if torch.cuda.is_available(): return torch.device("cuda")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def synchronize(device):
    if device.type == "mps": torch.mps.synchronize()
    elif device.type == "cuda": torch.cuda.synchronize(device)


def device_memory_mib(device):
    if device.type == "mps": return torch.mps.driver_allocated_memory() / (1024 ** 2)
    if device.type == "cuda": return torch.cuda.memory_allocated(device) / (1024 ** 2)
    return None


def reset_memory(device):
    if device.type == "mps": torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


class PrecisionCache:
    """Drop-in cache storage with optional FP8 or per-token/head INT8 payloads."""

    def __init__(self, fmt: str):
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.key = self.value = None
        self.key_scale = self.value_scale = None

    def _quantize(self, tensor):
        scale = tensor.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
        payload = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
        return payload, scale.to(torch.float32)

    def update(self, key, value):
        if self.fmt == "int8":
            key, key_scale = self._quantize(key)
            value, value_scale = self._quantize(value)
            if self.key is None:
                self.key, self.value = key, value
                self.key_scale, self.value_scale = key_scale, value_scale
            else:
                self.key = torch.cat((self.key, key), dim=2)
                self.value = torch.cat((self.value, value), dim=2)
                self.key_scale = torch.cat((self.key_scale, key_scale), dim=2)
                self.value_scale = torch.cat((self.value_scale, value_scale), dim=2)
        else:
            key, value = key.to(FLOAT_FORMATS[self.fmt]), value.to(FLOAT_FORMATS[self.fmt])
            if self.key is None:
                self.key, self.value = key, value
            else:
                self.key = torch.cat((self.key, key), dim=2)
                self.value = torch.cat((self.value, value), dim=2)

    def for_attention(self, dtype):
        if self.fmt == "int8":
            return (self.key.to(dtype) * self.key_scale.to(dtype), self.value.to(dtype) * self.value_scale.to(dtype))
        return self.key.to(dtype), self.value.to(dtype)

    def storage_bytes(self):
        tensors = (self.key, self.value, self.key_scale, self.value_scale)
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)

    def payload_bytes(self):
        tensors = (self.key, self.value)
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)

    def get_cache(self):
        # Compatibility with GPT2.clear_kv_cache().
        return {"key": self.key, "value": self.value}


def install_precision_cache(model, fmt: str):
    """Replace only attention forwards/caches; the Transformer remains the existing one."""
    for block in model.transformer.h:
        attn = block.attn
        attn.kvcache = PrecisionCache(fmt)

        def forward(self, x):
            batch, tokens, channels = x.size()
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            q = q.view(batch, tokens, self.n_head, channels // self.n_head).transpose(1, 2)
            k = k.view(batch, tokens, self.n_head, channels // self.n_head).transpose(1, 2)
            v = v.view(batch, tokens, self.n_head, channels // self.n_head).transpose(1, 2)
            self.kvcache.update(k, v)
            k_hist, v_hist = self.kvcache.for_attention(q.dtype)
            # The existing supported path is prefill followed by one-token decode.
            y = F.scaled_dot_product_attention(q, k_hist, v_hist, is_causal=tokens > 1)
            y = y.transpose(1, 2).contiguous().view(batch, tokens, channels)
            return self.c_proj(y)

        attn.forward = types.MethodType(forward, attn)


def load_model(checkpoint, device):
    from model_kvcache import GPT2, GPT2Config
    if not checkpoint.exists(): raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = GPT2(GPT2Config(vocab_size=50304)).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    prefix = "_orig_mod."
    for key, value in list(state.items()):
        if key.startswith(prefix): state[key[len(prefix):]] = state.pop(key)
    model.load_state_dict(state)
    model.eval()
    return model


def prompt_tokens(length, device):
    import tiktoken
    base = tiktoken.get_encoding("gpt2").encode("Artificial Intelligence systems use attention and memory. ")
    ids = (base * ((length + len(base) - 1) // len(base)))[:length]
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def theoretical_bytes(model, batch, length, fmt):
    c = model.config
    elements = 2 * c.n_layer * batch * c.n_head * length * (c.n_embd // c.n_head)
    if fmt == "int8":
        return elements + 2 * c.n_layer * batch * c.n_head * length * 4
    return elements * torch.empty((), dtype=FLOAT_FORMATS[fmt]).element_size()


def run_once(model, prompt, new_tokens, device):
    model.clear_kv_cache()
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode(): logits, _ = model(prompt, start_pos=0)
    synchronize(device)
    prefill = time.perf_counter() - start
    workload_peak = device_memory_mib(device)
    token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    decode = 0.0
    for step in range(new_tokens):
        synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode(): logits, _ = model(token, start_pos=prompt.size(1) + step)
        synchronize(device)
        decode += time.perf_counter() - start
        observed = device_memory_mib(device)
        if workload_peak is None:
            workload_peak = observed
        elif observed is not None:
            workload_peak = max(workload_peak, observed)
        token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    caches = [block.attn.kvcache for block in model.transformer.h]
    return prefill, decode, sum(c.storage_bytes() for c in caches), sum(c.payload_bytes() for c in caches), workload_peak


def run_format(model, fmt, prompt, args, device):
    row = {"format": fmt, "support_status": "unsupported", "model_weight_dtype": str(next(model.parameters()).dtype), "cache_storage": fmt, "attention_dtype": "torch.float32"}
    try:
        install_precision_cache(model, fmt)
        for _ in range(args.warmup):
            run_once(model, prompt, args.new_tokens, device)
            model.clear_kv_cache()
        reset_memory(device)
        baseline = device_memory_mib(device)
        measurements = []
        peak = baseline
        for _ in range(args.repetitions):
            prefill, decode, stored, payload, observed = run_once(model, prompt, args.new_tokens, device)
            peak = max(x for x in (peak, observed) if x is not None) if peak is not None or observed is not None else None
            if not torch.isfinite(torch.tensor([prefill, decode], dtype=torch.float64)).all():
                raise RuntimeError("non-finite timing result")
            measurements.append((prefill, decode, stored, payload, observed))
            model.clear_kv_cache()
        prefill = statistics.median(x[0] for x in measurements)
        decode = statistics.median(x[1] for x in measurements)
        stored = statistics.median(x[2] for x in measurements)
        payload = statistics.median(x[3] for x in measurements)
        peak = max((x[4] for x in measurements if x[4] is not None), default=None)
        row.update({"support_status": "supported", "theoretical_cache_mib": theoretical_bytes(model, 1, args.context_length + args.new_tokens, fmt) / (1024 ** 2), "measured_cache_mib": stored / (1024 ** 2), "raw_payload_mib": payload / (1024 ** 2), "baseline_memory_mib": baseline, "peak_memory_mib": peak, "prefill_latency_ms": prefill * 1000, "decode_latency_ms_per_token": decode / args.new_tokens * 1000 if args.new_tokens else 0.0, "total_latency_ms": (prefill + decode) * 1000, "notes": "FP8/INT8 K/V are converted to FP32 for attention; conversion is included in timing." if fmt in ("fp8", "int8") else "Stored K/V are converted to FP32 for attention; conversion is included in timing."})
    except (RuntimeError, TypeError, ValueError, NotImplementedError) as exc:
        row["notes"] = f"{type(exc).__name__}: {exc}"
    finally:
        model.clear_kv_cache(); gc.collect(); reset_memory(device)
    return row


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def plot(path, rows):
    try: import matplotlib.pyplot as plt
    except ImportError:
        print("Plot skipped: matplotlib is not installed."); return
    good = [r for r in rows if r["support_status"] == "supported"]
    if not good: return
    labels = [r["format"] for r in good]
    plt.bar(labels, [r["measured_cache_mib"] for r in good]); plt.ylabel("Cache storage (MiB)"); plt.title("KV-cache storage by format"); plt.tight_layout(); plt.savefig(path); plt.close()


def main():
    args = parse_args()
    formats = [x.strip().lower() for x in args.formats.split(",") if x.strip()]
    valid = set(FLOAT_FORMATS) | {"int8"}
    unknown = set(formats) - valid
    if unknown: raise ValueError(f"Unknown formats: {sorted(unknown)}")
    if args.context_length < 1 or args.context_length + args.new_tokens > 1024: raise ValueError("workload exceeds the model position table")
    if args.repetitions < 1 or args.warmup < 0: raise ValueError("invalid warmup/repetition count")
    device = choose_device(args.device)
    print(f"Device: {device} | PyTorch: {torch.__version__} | weights: torch.float32")
    print(f"Workload: context={args.context_length}, new_tokens={args.new_tokens}; storage comparison includes conversion overhead")
    model = load_model(args.checkpoint, device)
    prompt = prompt_tokens(args.context_length, device)
    rows = []
    for fmt in formats:
        print(f"Testing {fmt}...", flush=True)
        row = run_format(model, fmt, prompt, args, device)
        row.update({"device": str(device), "context_length": args.context_length, "new_tokens": args.new_tokens, "torch_version": torch.__version__})
        rows.append(row)
        print(f"  {row['support_status']}: {row.get('notes', '')}")
    write_csv(args.output, rows)
    if args.plot: plot(args.output.with_suffix(".png"), rows)
    print(f"Results written to {args.output}")
    print("Measured device memory is backend allocation, not total unified system memory; CPU reports no device-memory value.")


if __name__ == "__main__":
    main()
