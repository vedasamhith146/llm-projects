import time
import torch
import contextlib
import io
import sys

import sample
import sample_kvcache

PROMPT="Artificial Intelligence"
TOKENS_TO_GENERATE=500

@contextlib.contextmanager
def supress_stdout():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def run_benchmark(module_name,module_obj,prompt,max_tokens):
    print(f"\nBenchmarking:{module_name}")

    device=getattr(module_obj,'device',torch.device('cpu'))

    if hasattr(module_obj.model,'clear_kv_cache'):
        module_obj.model.clear_kv_cache()

    with supress_stdout():
        module_obj.generate(prompt,max_tokens=1)
        if hasattr(module_obj.model,'clear_kv_cache'):
            module_obj.model.clear_kv_cache()

    if device.type=='mps':
        torch.mps.synchronize()
    elif device.type=='cuda':
        torch.cuda.synchronize()

    start_time=time.perf_counter()

    with supress_stdout():
        module_obj.generate(prompt,max_tokens=max_tokens)

    if device.type=='mps':
        torch.mps.synchronize()
    elif device.type=='cuda':
        torch.cuda.synchronize()

    end_time=time.perf_counter()

    duration=end_time-start_time
    tps=max_tokens/duration

    print("Completed!")
    print(f"Time Taken: {duration:.4f}seconds")
    print(f"Speed: {tps:.2f}tokens/sec")

    return tps

if __name__=="__main__":
    print(f"\n STARTING BENCHMARK")
    print(f"prompt:'{PROMPT}'")
    print(f"Generating: {TOKENS_TO_GENERATE}tokens")

    tps_no_kv=run_benchmark("No KV Cache",sample,PROMPT,TOKENS_TO_GENERATE)
    tps_kv=run_benchmark("With KV Cache",sample_kvcache,PROMPT,TOKENS_TO_GENERATE)

    print("\n"+"="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"No Cache:{tps_no_kv:.2f}tok/s")
    print(f"KV Cache:{tps_kv:.2f}tok/s")

    if tps_no_kv>0:
        speedup=tps_kv/tps_no_kv
        print(f"\nSpeedUp:{speedup:.2f}x faster")
    print("="*40)


