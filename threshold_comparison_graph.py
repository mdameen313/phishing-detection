# threshold_comparison_graph.py
"""
Generate comparison graph for threshold 0.70 vs 0.65
"""

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Data from your test results
# ============================================================

# Metrics
metrics = ['Accuracy', 'Phishing Recall', 'Legitimate Recall']

# Values for threshold 0.70 (before)
before_values = [72.8, 59.2, 78.3]

# Values for threshold 0.65 (after)
after_values = [77.6, 70.2, 90.6]

# Improvement
improvement = [after - before for after, before in zip(after_values, before_values)]

# ============================================================
# Create the graph
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---------- Subplot 1: Bar Chart Comparison ----------
ax1 = axes[0]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, before_values, width, label='Threshold 0.70 (Before)', 
                color='#d32f2f', edgecolor='#b71c1c', linewidth=1.5)
bars2 = ax1.bar(x + width/2, after_values, width, label='Threshold 0.65 (After)', 
                color='#4caf50', edgecolor='#2e7d32', linewidth=1.5)

# Add value labels on bars
for bar, value in zip(bars1, before_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

for bar, value in zip(bars2, after_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add improvement arrows
for i, (before, after, imp) in enumerate(zip(before_values, after_values, improvement)):
    if imp > 0:
        y_pos = max(before, after) + 4
        ax1.annotate(f'+{imp:.1f}%', (i, y_pos), ha='center', fontsize=9, 
                     color='#2e7d32', fontweight='bold')
        ax1.annotate('', xy=(i, max(before, after) + 2), xytext=(i, max(before, after) + 3.5),
                     arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=1.5))

ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax1.set_title('Threshold 0.70 vs 0.65 - Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=11)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=70, color='gray', linestyle='--', alpha=0.5)

# ---------- Subplot 2: Improvement Chart ----------
ax2 = axes[1]

colors = ['#4caf50' if imp > 0 else '#d32f2f' for imp in improvement]
bars = ax2.bar(metrics, improvement, color=colors, edgecolor='black', linewidth=1)

# Add value labels
for bar, value in zip(bars, improvement):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'+{value:.1f}%' if value > 0 else f'{value:.1f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
ax2.set_title('Improvement After Changing Threshold to 0.65', fontsize=14, fontweight='bold')
ax2.set_ylim(-5, 20)
ax2.grid(True, alpha=0.3, axis='y')

# Add summary text box
summary_text = f"""
Summary:
• Accuracy:  +{improvement[0]:.1f}%
• Phishing Recall: +{improvement[1]:.1f}%
• Legitimate Recall: +{improvement[2]:.1f}%
"""
ax2.text(0.98, 0.02, summary_text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#ccc'))

plt.tight_layout()
plt.savefig('threshold_comparison_graph.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.show()

print("✅ Graph saved as 'threshold_comparison_graph.png'")