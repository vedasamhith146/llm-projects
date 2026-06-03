import os
import re

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

VOCAB_SIZE = 50257

# ==================================================
# FIND CHECKPOINTS
# ==================================================

checkpoint_files = []

for fname in os.listdir("."):

    m = re.match(
        r"gpt2_step_(\d+)\.pt",
        fname
    )

    if m:

        checkpoint_files.append(
            (
                int(m.group(1)),
                fname
            )
        )

checkpoint_files.sort()

print(f"Found {len(checkpoint_files)} checkpoints")

# ==================================================
# LOAD REFERENCE EMBEDDINGS (STEP 250)
# ==================================================

ref_ckpt = torch.load(
    "gpt2_step_250.pt",
    map_location="cpu"
)

ref_state = ref_ckpt["model_state_dict"]

ref_E = ref_state[
    "_orig_mod.transformer.wte.weight"
][:VOCAB_SIZE]

ref_E = F.normalize(
    ref_E,
    dim=1
)

# ==================================================
# STORAGE
# ==================================================

steps = []
global_similarity = []

# ==================================================
# MAIN LOOP
# ==================================================

for step, ckpt_file in checkpoint_files:

    print(f"Loading {step}")

    ckpt = torch.load(
        ckpt_file,
        map_location="cpu"
    )

    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    E = state[
        "_orig_mod.transformer.wte.weight"
    ][:VOCAB_SIZE]

    E = F.normalize(
        E,
        dim=1
    )

    # ------------------------------------------
    # cosine similarity token-by-token
    # ------------------------------------------

    token_cosines = (
        ref_E * E
    ).sum(dim=1)

    mean_cosine = (
        token_cosines.mean().item()
    )

    steps.append(step)
    global_similarity.append(mean_cosine)

    print(
        f"step={step:<6d}"
        f" mean_cos={mean_cosine:.4f}"
    )

# ==================================================
# PLOT
# ==================================================

plt.figure(figsize=(10,6))

plt.plot(
    steps,
    global_similarity,
    marker="o",
    linewidth=3
)

plt.xlabel("Training Step")
plt.ylabel("Mean Cosine vs Step 250")
plt.title(
    "Global Embedding Space Evolution"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "global_space_evolution.png",
    dpi=300
)

plt.show()

# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n")
print("="*60)
print("SUMMARY")
print("="*60)

for step, val in zip(
    steps,
    global_similarity
):
    print(
        f"{step:6d} : {val:.4f}"
    )