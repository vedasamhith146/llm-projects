import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
 
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".")

def tokenize(doc):
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    return tokens_np.astype(np.uint16)

def write_datafile(filename, tokens_np):
    tokens_np.tofile(filename)


if __name__ == '__main__':

    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True) 

    nprocs = max(1, os.cpu_count() // 2)
    print(f"Tokenizing with {nprocs} processes...")

    with mp.Pool(nprocs) as pool:
        shard_size = int(1e8) 
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
            else:
                remainder = shard_size - token_count
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_train_{shard_index:06d}.bin")
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                token_count = len(tokens) - remainder
                all_tokens_np[0:token_count] = tokens[remainder:]
                

        if token_count != 0:
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_train_{shard_index:06d}.bin")
            write_datafile(filename, all_tokens_np[:token_count])

    print("Done.")