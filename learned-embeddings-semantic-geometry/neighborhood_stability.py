import os
import re
import torch
import tiktoken
import torch.nn.functional as F
import matplotlib.pyplot as plt

TOKENS = [
    " information",
    " data",
    " research",
    " science",
    " health",
    " medical",
    " students",
    " education"
]

TOPK = 10
enc = tiktoken.get_encoding("gpt2")
token_ids = {}

for tok in TOKENS:
    tid = enc.encode(tok)[0]
    token_ids[tok] = tid
    print(tid,repr(tok))

checkpoint_files = []

for fname in os.listdir("."):
    m = re.match(r"gpt2_step_(\d+)\.pt",fname)

    if m:
        checkpoint_files.append((int(m.group(1)),fname))

checkpoint_files.sort()

print(f"\nFound {len(checkpoint_files)} checkpoints")

ref_ckpt = torch.load("gpt2_step_11500.pt",map_location="cpu")

if "model_state_dict" in ref_ckpt:
    ref_state = ref_ckpt["model_state_dict"]
else:
    ref_state = ref_ckpt

ref_E = ref_state["_orig_mod.transformer.wte.weight"][:50257]
ref_E = F.normalize(ref_E,dim=1)
final_neighbors = {}

for tok in TOKENS:
    tid = token_ids[tok]
    sims = ref_E @ ref_E[tid]
    idx = torch.topk(sims,TOPK + 1).indices.tolist()
    idx = [x for x in idx if x != tid][:TOPK]
    final_neighbors[tok] = set(idx)

steps = []

results = {tok: [] for tok in TOKENS}

for step, ckpt_file in checkpoint_files:

    print(f"Loading {step}")
    ckpt = torch.load(ckpt_file,map_location="cpu")
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    E = state[ "_orig_mod.transformer.wte.weight"][:50257]
    E = F.normalize(E,dim=1)

    steps.append(step)
    for tok in TOKENS:
        tid = token_ids[tok]
        sims = E @ E[tid]
        idx = torch.topk(sims,TOPK + 1).indices.tolist()

        idx = [x for x in idx if x != tid][:TOPK]
        current = set(idx)
        final = final_neighbors[tok]

        jaccard = len(current & final) / len(current | final)
        results[tok].append(jaccard)

plt.figure(figsize=(12,7))

for tok in TOKENS:

    plt.plot(steps,results[tok],marker="o",linewidth=2,label=tok.strip())

plt.xlabel("Training Step")
plt.ylabel("Jaccard Similarity")
plt.title("Neighborhood Stability")
plt.grid(True)
plt.legend(fontsize=8)

plt.tight_layout()

plt.savefig(
    "neighborhood_stability.png",
    dpi=300
)

plt.show()