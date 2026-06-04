import os
import re
import torch
import tiktoken
import matplotlib.pyplot as plt

TOKENS = [

    " information",
    " data",
    " research",
    " science",
    " health",
    " medical",
    " students",
    " education",

    "ingly",
    "ably",
    "ively",
    "ously",

    " the",
    " and",
    " of",
    " in",
]

enc = tiktoken.get_encoding("gpt2")
token_ids = {}
print("\nTracking Tokens:\n")

for tok in TOKENS:
    ids = enc.encode(tok)
    if len(ids) != 1:
        raise ValueError( f"{repr(tok)} -> {ids}" )
    token_ids[tok] = ids[0]
    print(token_ids[tok],repr(tok))


checkpoint_files = []

for fname in os.listdir("."):
    m = re.match(r"gpt2_step_(\d+)\.pt",fname)
    if m:
        checkpoint_files.append(
            (
                int(m.group(1)),
                fname
            )
        )

checkpoint_files.sort()
print(f"\nFound {len(checkpoint_files)} checkpoints")
ref_checkpoint = torch.load("gpt2_step_250.pt", map_location="cpu")
ref_state_dict = ref_checkpoint[ "model_state_dict"]
ref_E = ref_state_dict[ "_orig_mod.transformer.wte.weight"][:50257]


results = { tok: [] for tok in TOKENS}

steps = []

for step, ckpt_file in checkpoint_files:

    print(f"Loading {step}")

    checkpoint = torch.load(ckpt_file,map_location="cpu")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
     state_dict = checkpoint

    E = state_dict["_orig_mod.transformer.wte.weight"][:50257]

    steps.append(step)

    for tok in TOKENS:
        tid = token_ids[tok]
        movement = torch.norm(E[tid] - ref_E[tid]).item()
        results[tok].append(movement)


plt.figure(figsize=(14,8))

for tok in TOKENS:
    plt.plot(steps,results[tok],marker="o",linewidth=2,label=tok)

plt.xlabel("Training Step")
plt.ylabel("Distance from Step-250 Embedding")
plt.title("Embedding Movement During Training")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("embedding_movement_during_training.png",dpi=300)
plt.show()
