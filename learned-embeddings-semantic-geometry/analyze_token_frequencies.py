import numpy as np
import tiktoken
import json
import os

enc = tiktoken.get_encoding("gpt2")
freq = np.zeros(enc.n_vocab, dtype=np.int64)
DATA_DIR = "."

print("Counting token frequencies...")

for shard_idx in range(60):

    filename = os.path.join(DATA_DIR,f"edufineweb_train_{shard_idx:06d}.bin")
    print(f"Processing {filename}")
    tokens = np.fromfile(filename,dtype=np.uint16)
    freq += np.bincount(tokens,minlength=enc.n_vocab)

print("Finished counting.")
print(f"Total tokens counted: {freq.sum():,}")

np.save("token_frequencies_60shards.npy", freq)

top200 = np.argsort(freq)[::-1][:200]

results = []

for rank, token_id in enumerate(top200, start=1):

    token_str = enc.decode([int(token_id)])

    results.append({
        "rank": rank,
        "token_id": int(token_id),
        "token": token_str,
        "repr": repr(token_str),
        "count": int(freq[token_id])
    })

with open("top200_tokens_first60shards.json","w",encoding="utf-8") as f:
    json.dump(results,f,ensure_ascii=False,indent=2)

print("Saved top200_tokens_first60shards.json")


print("\nTop 20 Tokens:\n")

for item in results[:20]:
    print(
        f"{item['rank']:3d} | "
        f"id={item['token_id']:5d} | "
        f"count={item['count']:15,d} | "
        f"{item['repr']}"
    )