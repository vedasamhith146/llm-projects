import torch
from moe import MOElayer,din,noe,topk
import matplotlib.pyplot as plt

model=MOElayer(din,noe)

print("Running simulation")
for batch in range(50):
    dummy_input=torch.randn(32*64,din)
    _=model(dummy_input,topk)

all_choices=torch.cat(model.expert_history).numpy()

plt.figure(figsize=(10, 6))
plt.hist(all_choices, bins=range(noe + 1), rwidth=0.8, color='skyblue', edgecolor='black', align='left')
plt.title(f"Expert Utilization (Top-{topk} Routing)")
plt.xlabel("Expert ID")
plt.ylabel("Number of times selected")
plt.xticks(range(noe))
plt.grid(axis='y', alpha=0.3)


expected_usage = len(all_choices) / noe
plt.axhline(y=expected_usage, color='r', linestyle='--', label='Perfect Balance')
plt.legend()

plt.show()
    
