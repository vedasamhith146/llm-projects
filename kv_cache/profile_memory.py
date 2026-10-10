import torch
import time
import sys
import os

import sample_kvcache

def get_memory_usage(device):

    if device.type=='mps':
        return torch.mps.current_allocated_memory()/1024/1024
    elif device.type=='cuda':
        return torch.cuda.memory_allocated()/1024/1024
    else:
        return 0.0
    
def profile_kv_growth(max_length=1024,step_size=100):
    print(f"\n===Profiling KV Cache Memory Growth===")
    print(f"Device:{sample_kvcache.device.type.upper}")
    model=sample_kvcache.model
    device=sample_kvcache.device

    model.clear_kv_cache()
    if device.type=='mps':
        torch.mps.empty_cache()
    if device.type=='cuda':
        torch.cuda.empty_cache()

    baseline_mem=get_memory_usage(device)
    print(f"Baseline memory (Model weights):{baseline_mem:.2f}MB")
    print("\n"+"="*45)
    print(f"| {'Seq length':<12} |{'Total Mem (MB)': <15} | {'KV Cache Only':<13} |")
    print("="*45)

    dummy_token=torch.tensor([[50256]],dtype=torch.long,device=device)

    with torch.no_grad():
        model(dummy_token,start_pos=0)
    for pos in range(1,max_length+1):
        with torch.no_grad():
            model(dummy_token,start_pos=pos)

        if pos%step_size==0 or pos==10:
            current_mem=get_memory_usage(device)
            kv_cost=current_mem-baseline_mem

            print(f"|{pos:<12} | {current_mem:<15.2f} | {kv_cost:<13.2f} |")

    print("="*45)

    final_mem=get_memory_usage(device)
    total_growth=final_mem-baseline_mem
    print(f"\n Total KV Cache size for {max_length} tokens:{total_growth:.2f}MB")
    print(f"Average cost per token:{(total_growth/max_length)*1000:.2f}KB/token")


if __name__=="__main__":
    profile_kv_growth(max_length=1000,step_size=100)
    

