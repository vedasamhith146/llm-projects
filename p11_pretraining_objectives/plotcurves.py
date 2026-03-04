import json
import matplotlib.pyplot as plt

def smooth_curve(points, factor=0.9):
    """
    Applies an exponential moving average to smooth out the noisy loss curve.
    factor: 0.0 means no smoothing, 0.99 means heavy smoothing.
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# 1. Load the data from your JSON files
try:
    with open('masked_loss.json', 'r') as f:
        masked_loss = json.load(f)
    with open('causal_loss.json', 'r') as f:
        causal_loss = json.load(f)
    with open('prefix_loss.json', 'r') as f:
        prefix_loss = json.load(f)
except FileNotFoundError as e:
    print(f"Error: Could not find one of the JSON files. Make sure they are in the same folder as this script!\n{e}")
    exit()

# 2. Set up the plot
plt.figure(figsize=(12, 7))

# Plot the noisy raw data in the background (faintly)
plt.plot(masked_loss, alpha=0.15, color='blue')
plt.plot(causal_loss, alpha=0.15, color='red')
plt.plot(prefix_loss, alpha=0.15, color='green')

# Plot the smoothed trend lines in the foreground (boldly)
plt.plot(smooth_curve(masked_loss, 0.95), label='Masked LM (Autoencoding)', color='blue', linewidth=2)
plt.plot(smooth_curve(causal_loss, 0.95), label='Causal LM (Autoregressive)', color='red', linewidth=2)
plt.plot(smooth_curve(prefix_loss, 0.95), label='Prefix LM (Hybrid)', color='green', linewidth=2)

# 3. Add labels and aesthetics
plt.title('Pretraining Objectives: Loss Comparison on TinyShakespeare', fontsize=14, fontweight='bold')
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Cross-Entropy Loss', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Save the plot as an image and show it on screen
plt.savefig('loss_comparison_curves.png', dpi=300, bbox_inches='tight')
print("Graph saved as 'loss_comparison_curves.png'")
plt.show()