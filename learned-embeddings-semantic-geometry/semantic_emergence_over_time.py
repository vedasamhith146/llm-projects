import os
import re
import torch
import tiktoken
import torch.nn.functional as F
import matplotlib.pyplot as plt

PAIRS = [
    (" information"," data"),
    (" research"," science"),
    (" health"," medical"),
    (" students"," education"),
    ("ingly","ably"),
    ("ively","ously"),
    (" the"," and"),
    (" of"," in"),
]

enc = tiktoken.get_encoding("gpt2")

pair_ids = []

for a,b in PAIRS:
    a_id = enc.encode(a)[0]
    b_id = enc.encode(b)[0]

    pair_ids.append(
        (a,b,a_id,b_id)
    )

    print(a,"->",a_id)
    print(b,"->",b_id)

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

steps = []

results = {
    f"{a.strip()} ↔ {b.strip()}": []
    for a,b,_,_ in pair_ids
}

for step, ckpt_file in checkpoint_files:

    print(f"Loading {step}")

    checkpoint = torch.load(
        ckpt_file,
        map_location="cpu"
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    E = state_dict[
        "_orig_mod.transformer.wte.weight"
    ][:50257]

    E = F.normalize(E,dim=1)

    steps.append(step)

    for a,b,a_id,b_id in pair_ids:

        sim = torch.dot(
            E[a_id],
            E[b_id]
        ).item()

        results[
            f"{a.strip()} ↔ {b.strip()}"
        ].append(sim)

plt.figure(figsize=(12,7))

for name,values in results.items():

    plt.plot(
        steps,
        values,
        marker="o",
        linewidth=2,
        label=name
    )

plt.xlabel("Training Step")
plt.ylabel("Cosine Similarity")
plt.title("Semantic Emergence Over Time")
plt.grid(True)
plt.legend(fontsize=8)

plt.tight_layout()

plt.savefig(
    "semantic_emergence_over_time.png",
    dpi=300
)

plt.show()

print("\nFinal Similarities\n")

for name,values in results.items():
    print(
        f"{name:30s} {values[-1]:.4f}"
    )