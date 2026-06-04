import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
freqs = np.load("token_frequencies_60shards.npy")

checkpoint = torch.load("gpt2_step_11500.pt", map_location="cpu")
E = checkpoint["_orig_mod.transformer.wte.weight"]

norms = torch.norm(E,dim=1).numpy()

mask = freqs > 0

freqs = freqs[mask]
norms = norms[mask]
log_freqs = np.log10(freqs)

corr, pval = pearsonr(log_freqs,norms)

print("Pearson correlation:", corr)
print("P-value:", pval)
plt.figure(figsize=(10, 6))

plt.scatter(log_freqs,norms,alpha=0.3,s=5)

plt.xlabel("log10(Frequency)")
plt.ylabel("Embedding Norm")

plt.title(f"Frequency vs Embedding Norm\n"
f"Pearson r = {corr:.3f}"
)
plt.grid(True)
plt.tight_layout()
plt.show()