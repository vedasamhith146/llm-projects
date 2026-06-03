import matplotlib.pyplot as plt
import torch

losses = torch.load("losses.pt")

plt.plot(losses)
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()