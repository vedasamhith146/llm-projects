import os
import re
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

VOCAB_SIZE = 50257

checkpoint_files = []

for fname in os.listdir("."):
    m = re.match(r"gpt2_step_(\d+)\.pt", fname)
    if m:
        checkpoint_files.append((int(m.group(1)),fname))

checkpoint_files.sort()

print(f"\nFound {len(checkpoint_files)} checkpoints")

ref_checkpoint = torch.load("gpt2_step_250.pt",map_location="cpu")
ref_state_dict = ref_checkpoint["model_state_dict"]
ref_E = ref_state_dict["_orig_mod.transformer.wte.weight"][:VOCAB_SIZE]

ref_E = F.normalize(ref_E, dim=1)

steps = []
global_similarity = []

for step, ckpt_file in checkpoint_files:
    print(f"Loading {step}")
    checkpoint = torch.load(ckpt_file,map_location="cpu")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    E = state_dict["_orig_mod.transformer.wte.weight"][:VOCAB_SIZE]
    E = F.normalize(E, dim=1)
    mean_cosine = (ref_E * E).sum(dim=1).mean().item()

    steps.append(step)
    global_similarity.append(mean_cosine)

plt.figure(figsize=(12,6))

plt.plot(steps,global_similarity,marker="o",linewidth=2)

plt.xlabel("Training Step")
plt.ylabel("Mean Cosine vs Step-250")
plt.title("Global Embedding Space Evolution")
plt.grid(True)
plt.tight_layout()

plt.savefig("global_embedding_space_evolution.png",dpi=300)
plt.show()

print("\nResults:\n")

for step, sim in zip(steps, global_similarity):
    print(step, round(sim, 4))