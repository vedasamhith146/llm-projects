import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# RAW ACTIVATION DATA: No Residuals, No LayerNorm
# ============================================================
ACTIVATION_DATA = [
    {'h.0': 4.673561306844931e-06, 'h.1': 2.4538415441781523e-17, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 0.5320766568183899},
    {'h.0': 6.7431719799060374e-06, 'h.1': 3.1705760087263646e-17, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 5.630645751953125},
    {'h.0': 1.0718131306930445e-05, 'h.1': 4.657934431241639e-17, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 10.987664222717285},
    {'h.0': 1.6946329196798615e-05, 'h.1': 4.4420230860085577e-17, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 16.587554931640625},
    {'h.0': 2.506040982552804e-05, 'h.1': 6.906797972651191e-17, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 22.35691261291504},
    {'h.0': 3.7123016227269545e-05, 'h.1': 1.414318308252393e-16, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 28.22104835510254},
    {'h.0': 5.488094757311046e-05, 'h.1': 3.070188443144643e-16, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 34.11540222167969},
    {'h.0': 6.868952186778188e-05, 'h.1': 5.348914064935196e-16, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 40.00303649902344},
    {'h.0': 8.255169814219698e-05, 'h.1': 7.272864171933022e-16, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 45.86589431762695},
    {'h.0': 0.0001082868839148432, 'h.1': 1.1850922668329641e-15, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 51.688907623291016},
    {'h.0': 0.0001211771959788166, 'h.1': 1.3537370560268874e-15, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 57.460636138916016},
    {'h.0': 0.0001385087671224028, 'h.1': 2.0335516175704407e-15, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 63.18097686767578},
    {'h.0': 0.00016994554607663304, 'h.1': 2.4559076560598525e-15, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 68.84835815429688},
    {'h.0': 0.00019925435481127352, 'h.1': 3.855938344906858e-15, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 74.47659301757812},
    {'h.0': 0.00022062471543904394, 'h.1': 4.1557537034005165e-15, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 80.0435791015625},
    {'h.0': 0.0002662566548679024, 'h.1': 6.740002097562446e-15, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 85.54023742675781},
    {'h.0': 0.0002663756313268095, 'h.1': 6.585765444680437e-15, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 90.96754455566406},
    {'h.0': 0.0002898468228522688, 'h.1': 8.864610603995671e-15, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 96.3197021484375},
    {'h.0': 0.00034640359808690846, 'h.1': 1.0578734238695254e-14, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 101.5766372680664},
    {'h.0': 0.00036360486410558224, 'h.1': 1.1956729788935912e-14, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 106.72900390625}
]

# ============================================================
# RAW GRADIENT DATA: No Residuals, No LayerNorm (CORRECTED)
# ============================================================
GRADIENT_DATA = [
    {'h.0': 0.0, 'h.1': 0.0, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 0.08235554229312547},
    {'h.0': 0.0, 'h.1': 0.0, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 0.15596655528130024},
    {'h.0': 0.0, 'h.1': 0.0, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 0.26944875145667324},
    {'h.0': 0.0, 'h.1': 0.0, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 0.40038478195765104},
    {'h.0': 0.0, 'h.1': 0.0, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 0.5269383320241356},
    {'h.0': 0.0, 'h.1': 0.0, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 0.6564644054574699},
    {'h.0': 0.0, 'h.1': 0.0, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 0.7928278755618676},
    {'h.0': 0.0, 'h.1': 0.0, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 0.9130455620146603},
    {'h.0': 0.0, 'h.1': 0.0, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 1.0482585930575559},
    {'h.0': 0.0, 'h.1': 0.0, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 1.1643006765957151},
    {'h.0': 0.0, 'h.1': 0.0, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 1.270117709344378},
    {'h.0': 0.0, 'h.1': 0.0, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 1.36702513684349},
    {'h.0': 0.0, 'h.1': 0.0, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 1.441276068869374},
    {'h.0': 0.0, 'h.1': 0.0, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 1.4967399775187187},
    {'h.0': 0.0, 'h.1': 0.0, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 1.5016077485347479},
    {'h.0': 0.0, 'h.1': 0.0, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 1.5256698294260853},
    {'h.0': 0.0, 'h.1': 0.0, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 1.489812908843568},
    {'h.0': 0.0, 'h.1': 0.0, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 1.461140011709891},
    {'h.0': 0.0, 'h.1': 0.0, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 1.3491150852087517},
    {'h.0': 0.0, 'h.1': 0.0, 'h.2': 0.0, 'h.3': 0.0, 'h.4': 0.0, 'h.5': 0.0, 'h.6': 0.0, 'h.7': 0.0, 'h.8': 0.0, 'h.9': 0.0, 'h.10': 0.0, 'h.11': 1.2443306147554793}
]

# Convert to arrays
act_arr = np.array([[d[f'h.{i}'] for i in range(12)] for d in ACTIVATION_DATA])
grad_arr = np.array([[d[f'h.{i}'] for i in range(12)] for d in GRADIENT_DATA])
steps = np.arange(20) * 10
layer_labels = [f'h.{i}' for i in range(12)]

# ============================================================
# FIGURE 1: 4-PANEL ACTIVATION ANALYSIS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Panel 1: Heatmap
ax1 = axes[0, 0]
viz_arr = act_arr.copy()
viz_arr[viz_arr == 0] = 1e-20
im1 = ax1.imshow(np.log10(viz_arr + 1e-20), aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
ax1.set_xticks(range(12))
ax1.set_xticklabels(layer_labels, rotation=45)
ax1.set_yticks(range(0, 20, 2))
ax1.set_yticklabels([f'{s}' for s in steps[::2]])
ax1.set_xlabel('Layer', fontsize=12)
ax1.set_ylabel('Training Step', fontsize=12)
ax1.set_title('Activation Norms: No Residuals, No LayerNorm\n(log10 scale, green = higher)', fontsize=13, fontweight='bold')
plt.colorbar(im1, ax=ax1, label='log10(Activation Norm)')

# Panel 2: Trajectory
ax2 = axes[0, 1]
ax2.plot(steps, act_arr[:, 0], 'o-', color='crimson', linewidth=2.5, markersize=6, label='h.0 (embedding residual)')
ax2.plot(steps, act_arr[:, 1], 's--', color='gray', linewidth=1.5, markersize=4, label='h.1 (dead layer)')
ax2.plot(steps, act_arr[:, 11], 'D-', color='darkgreen', linewidth=2.5, markersize=6, label='h.11 (final layer, explodes)')
ax2.axhspan(1e-16, 1e-4, alpha=0.15, color='gray', label='Dead zone (h.1-h.10)')
ax2.set_xlabel('Training Step', fontsize=12)
ax2.set_ylabel('Activation Norm', fontsize=12)
ax2.set_title('Layer Trajectory: Only h.0 and h.11 Survive\n(No Residuals, No LayerNorm)', fontsize=13, fontweight='bold')
ax2.set_yscale('log')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.annotate('h.0: tiny but growing\n(embedding residual)', xy=(190, act_arr[-1,0]), 
             xytext=(120, 1e-3), fontsize=9, color='crimson',
             arrowprops=dict(arrowstyle='->', color='crimson', lw=1.5))
ax2.annotate('h.11: EXPLODES\n(uncontrolled growth)', xy=(190, act_arr[-1,11]), 
             xytext=(120, 30), fontsize=9, color='darkgreen',
             arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))
ax2.annotate('h.1-h.10: ZERO\n(no gradient flow)', xy=(100, 1e-15), 
             xytext=(30, 1e-12), fontsize=9, color='gray',
             arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Panel 3: Mean per layer
ax3 = axes[1, 0]
mean_across_steps = act_arr.mean(axis=0)
colors = ['crimson' if i == 0 else 'darkgreen' if i == 11 else 'gray' for i in range(12)]
bars = ax3.bar(range(12), mean_across_steps, color=colors, edgecolor='black', linewidth=0.5)
ax3.set_xticks(range(12))
ax3.set_xticklabels(layer_labels)
ax3.set_xlabel('Layer', fontsize=12)
ax3.set_ylabel('Mean Activation Norm\n(across all 20 steps)', fontsize=12)
ax3.set_title('Mean Activation per Layer\n(Averaged over 200 training steps)', fontsize=13, fontweight='bold')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, mean_across_steps)):
    if val > 1e-10:
        label = f'{val:.1f}' if val > 1 else f'{val:.2e}'
        ax3.text(bar.get_x() + bar.get_width()/2, val * 1.5, label, 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='crimson', edgecolor='black', label='h.0 (embedding residual)'),
    Patch(facecolor='darkgreen', edgecolor='black', label='h.11 (final layer, explodes)'),
    Patch(facecolor='gray', edgecolor='black', label='h.1-h.10 (dead, ~0)')
]
ax3.legend(handles=legend_elements, loc='upper left', fontsize=9)

# Panel 4: Growth ratio
ax4 = axes[1, 1]
ratio = act_arr[:, 11] / (act_arr[:, 0] + 1e-20)
ax4.plot(steps, ratio, 'o-', color='purple', linewidth=3, markersize=7, label='h.11 / h.0 ratio')
ax4.fill_between(steps, ratio, alpha=0.2, color='purple')
ax4.set_xlabel('Training Step', fontsize=12)
ax4.set_ylabel('Growth Ratio (h.11 / h.0)', fontsize=12)
ax4.set_title('The Explosion Problem\n(Final vs Embedding Layer Growth)', fontsize=13, fontweight='bold')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=1e6, color='red', linestyle='--', alpha=0.7, label='1M ratio')
ax4.legend(fontsize=11)
ax4.annotate(f'Final ratio: {ratio[-1]:.1e}x', xy=(190, ratio[-1]), 
             xytext=(120, ratio[-1]*3), fontsize=10, fontweight='bold', color='purple',
             arrowprops=dict(arrowstyle='->', color='purple', lw=2))
textstr = """Why this happens (No Residuals, No Norm):

- h.0: Tiny residual from embedding
  (not passed through any layer)

- h.1-h.10: ZERO
  No residual = no signal propagation
  Gradients die -> weights don't update
  -> dead layers

- h.11: EXPLODES
  Final layer gets raw gradient from loss
  Uncontrolled growth without norm
  -> 293,000x larger than h.0

This is a DEAD model with one
exploding layer."""
props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='black')
ax4.text(0.98, 0.02, textstr, transform=ax4.transAxes, fontsize=9,
         verticalalignment='bottom', horizontalalignment='right', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('no_residual_no_norm_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# FIGURE 2: GRADIENT NORMS + TRAJECTORY
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gradient bar chart
ax1 = axes[0]
mean_grad = grad_arr.mean(axis=0)
colors_grad = ['crimson' if i == 0 else 'darkgreen' if i == 11 else 'gray' for i in range(12)]
bars = ax1.bar(range(12), mean_grad, color=colors_grad, edgecolor='black', linewidth=0.8, width=0.7)
ax1.set_xticks(range(12))
ax1.set_xticklabels(layer_labels, fontsize=11)
ax1.set_xlabel('Layer', fontsize=13)
ax1.set_ylabel('Mean Gradient Norm\n(across 200 training steps)', fontsize=13)
ax1.set_title('Gradient Norms per Layer\n(No Residuals, No LayerNorm)', fontsize=14, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, mean_grad)):
    if val > 1e-10:
        label = f'{val:.2f}' if val > 0.1 else f'{val:.3f}'
        ax1.text(bar.get_x() + bar.get_width()/2, val * 1.5, label, 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
legend_elements = [
    Patch(facecolor='crimson', edgecolor='black', label='h.0: zero gradient (dead)'),
    Patch(facecolor='darkgreen', edgecolor='black', label='h.11: receives gradient (only layer learning)'),
    Patch(facecolor='gray', edgecolor='black', label='h.1-h.10: ZERO gradients (dead)')
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

# Gradient trajectory
ax2 = axes[1]
ax2.plot(steps, grad_arr[:, 11], 'D-', color='darkgreen', linewidth=2.5, markersize=7, label='h.11 gradient norm')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='h.0-h.10: exactly 0')
ax2.fill_between(steps, 0, grad_arr[:, 11], alpha=0.2, color='darkgreen')
ax2.set_xlabel('Training Step', fontsize=13)
ax2.set_ylabel('Gradient Norm', fontsize=13)
ax2.set_title('Gradient Flow Over Time\n(Only h.11 Receives Gradients)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.annotate(f'Step 0: {grad_arr[0,11]:.3f}', xy=(0, grad_arr[0,11]), xytext=(30, 0.3), 
             fontsize=10, arrowprops=dict(arrowstyle='->', color='darkgreen'))
ax2.annotate(f'Step 190: {grad_arr[-1,11]:.3f}', xy=(190, grad_arr[-1,11]), xytext=(140, 1.3), 
             fontsize=10, arrowprops=dict(arrowstyle='->', color='darkgreen'))
props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='black')
textstr = """CRITICAL OBSERVATION:

Gradient flow confirms the diagnosis:

- h.0-h.10: EXACTLY 0.0 at every step
  -> No backprop signal reaches them
  -> Weights frozen at initialization
  -> These layers are COMPLETELY DEAD

- h.11: ONLY layer with non-zero gradients
  -> Receives direct gradient from loss
  -> Grows from 0.08 -> 1.24 over 200 steps
  -> Then PLATEAUS (saturates around step 150)

This is NOT a 12-layer model.
It is a 1-layer model with 11 dead layers.

Without residuals:
  Forward pass dies after h.0
  Backward pass dies before h.11
  Only h.11 learns anything at all."""
ax2.text(0.98, 0.98, textstr, transform=ax2.transAxes, fontsize=9.5,
         verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('gradient_norm_and_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# FIGURE 3: SIDE-BY-SIDE ACTIVATIONS vs GRADIENTS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Activations heatmap
ax1 = axes[0, 0]
viz_arr = act_arr.copy()
viz_arr[viz_arr == 0] = 1e-20
im1 = ax1.imshow(np.log10(viz_arr + 1e-20), aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
ax1.set_xticks(range(12))
ax1.set_xticklabels(layer_labels, rotation=45)
ax1.set_yticks(range(0, 20, 2))
ax1.set_yticklabels([f'{s}' for s in steps[::2]])
ax1.set_xlabel('Layer', fontsize=11)
ax1.set_ylabel('Training Step', fontsize=11)
ax1.set_title('ACTIVATIONS: No Residuals, No LayerNorm\n(log10 scale)', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=ax1, label='log10(Activation Norm)')

# Activation trajectory
ax2 = axes[0, 1]
ax2.plot(steps, act_arr[:, 0], 'o-', color='crimson', linewidth=2, markersize=5, label='h.0')
ax2.plot(steps, act_arr[:, 11], 'D-', color='darkgreen', linewidth=2, markersize=5, label='h.11')
ax2.set_xlabel('Training Step', fontsize=11)
ax2.set_ylabel('Activation Norm', fontsize=11)
ax2.set_title('Activation Trajectory: h.0 vs h.11', fontsize=12, fontweight='bold')
ax2.set_yscale('log')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Gradients heatmap
ax3 = axes[1, 0]
viz_grad = grad_arr.copy()
viz_grad[viz_grad == 0] = 1e-20
im3 = ax3.imshow(np.log10(viz_grad + 1e-20), aspect='auto', cmap='plasma', interpolation='nearest')
ax3.set_xticks(range(12))
ax3.set_xticklabels(layer_labels, rotation=45)
ax3.set_yticks(range(0, 20, 2))
ax3.set_yticklabels([f'{s}' for s in steps[::2]])
ax3.set_xlabel('Layer', fontsize=11)
ax3.set_ylabel('Training Step', fontsize=11)
ax3.set_title('GRADIENTS: No Residuals, No LayerNorm\n(log10 scale)', fontsize=12, fontweight='bold')
plt.colorbar(im3, ax=ax3, label='log10(Gradient Norm)')

# Gradient trajectory
ax4 = axes[1, 1]
ax4.plot(steps, grad_arr[:, 11], 'D-', color='darkgreen', linewidth=2.5, markersize=6, label='h.11 gradient')
ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax4.fill_between(steps, 0, grad_arr[:, 11], alpha=0.2, color='darkgreen')
ax4.set_xlabel('Training Step', fontsize=11)
ax4.set_ylabel('Gradient Norm', fontsize=11)
ax4.set_title('Gradient Trajectory: Only h.11 Learns', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('act_vs_grad_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("All 3 figures saved!")