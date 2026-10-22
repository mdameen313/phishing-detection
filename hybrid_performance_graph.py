# hybrid_performance_graph.py
"""
Hybrid Model Performance Graph
Shows accuracy improvement: RF → Hybrid → Hybrid + LLM
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================
OUTPUT_FOLDER = 'validation_graphs'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# ============================================================
# DATA - Model Performance
# ============================================================
# Three configurations
configs = ['Random Forest\nOnly', 'Hybrid Model\n(CBA + RF)', 'Hybrid + LLM\n(Full System)']

# Metrics for each configuration
accuracy  = [82.4, 85.6, 92.3]
precision = [83.1, 85.2, 88.7]
recall    = [88.7, 90.1, 94.2]
f1        = [85.8, 87.6, 91.4]

# ============================================================
# GRAPH 1: Bar Chart - All Metrics Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(configs))
width = 0.2

bars1 = ax.bar(x - 1.5*width, accuracy,  width, label='Accuracy',  color='#1a73e8', edgecolor='black')
bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='#4caf50', edgecolor='black')
bars3 = ax.bar(x + 0.5*width, recall,    width, label='Recall',    color='#ff9800', edgecolor='black')
bars4 = ax.bar(x + 1.5*width, f1,        width, label='F1 Score',  color='#d32f2f', edgecolor='black')

# Add value labels
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Model Configuration', fontsize=13, fontweight='bold')
ax.set_title('Hybrid Model Performance Comparison\n(Accuracy Improvement through Layers)', 
             fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=11)
ax.legend(loc='lower right', fontsize=11)
ax.set_ylim(70, 100)
ax.grid(True, alpha=0.3, axis='y')

# Add improvement arrows
# F1 Score improvements
ax.annotate('', xy=(1, 87.6), xytext=(0, 85.8),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(0.5, 86.7, '+1.8%', fontsize=10, color='green', fontweight='bold')

ax.annotate('', xy=(2, 91.4), xytext=(1, 87.6),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(1.5, 89.5, '+3.8%', fontsize=10, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_FOLDER}/hybrid_performance_bar.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {OUTPUT_FOLDER}/hybrid_performance_bar.png")


# ============================================================
# GRAPH 2: F1 Score Improvement (Line/Step Chart)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

# Plot lines for each metric
ax.plot(configs, accuracy,  'o-', color='#1a73e8', linewidth=3, markersize=14, label='Accuracy')
ax.plot(configs, precision, 's-', color='#4caf50', linewidth=3, markersize=14, label='Precision')
ax.plot(configs, recall,    '^-', color='#ff9800', linewidth=3, markersize=14, label='Recall')
ax.plot(configs, f1,        'd-', color='#d32f2f', linewidth=3, markersize=14, label='F1 Score')

# Add value labels
for i, (a, p, r, f) in enumerate(zip(accuracy, precision, recall, f1)):
    ax.text(i, a+0.7, f'{a}%', ha='center', fontsize=10, color='#1a73e8', fontweight='bold')
    ax.text(i, p-1.5, f'{p}%', ha='center', fontsize=10, color='#4caf50', fontweight='bold')
    ax.text(i, r+0.7, f'{r}%', ha='center', fontsize=10, color='#ff9800', fontweight='bold')
    ax.text(i, f-1.5, f'{f}%', ha='center', fontsize=10, color='#d32f2f', fontweight='bold')

ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Configuration', fontsize=13, fontweight='bold')
ax.set_title('Performance Improvement through Hybrid Layers\n(RF → Hybrid → Hybrid + LLM)', 
             fontsize=15, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.set_ylim(78, 98)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_FOLDER}/hybrid_performance_line.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {OUTPUT_FOLDER}/hybrid_performance_line.png")


# ============================================================
# GRAPH 3: Accuracy Improvement Highlight
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

# Focus on F1 score improvement
colors = ['#ff9800', '#4caf50', '#4caf50']
bars = ax.bar(configs, f1, color=colors, edgecolor='black', linewidth=2, width=0.6)

# Add value labels
for bar, val in zip(bars, f1):
    ax.text(bar.get_x() + bar.get_width()/2., val + 0.5,
            f'{val}%', ha='center', va='bottom', fontsize=16, fontweight='bold')

# Add improvement annotations
ax.annotate('', xy=(1, 87.6), xytext=(0, 85.8),
            arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
ax.text(0.5, 86.7, '+1.8%', fontsize=12, color='green', fontweight='bold', ha='center')

ax.annotate('', xy=(2, 91.4), xytext=(1, 87.6),
            arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
ax.text(1.5, 89.5, '+3.8%', fontsize=12, color='green', fontweight='bold', ha='center')

# Total improvement
ax.annotate('', xy=(2, 91.4), xytext=(0, 85.8),
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=3, 
                          connectionstyle="arc3,rad=-0.3"))
ax.text(1.0, 84, 'Total Improvement: +5.6%', fontsize=13, color='darkgreen', 
        fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='#e8f5e9', edgecolor='green', linewidth=2))

ax.set_ylabel('F1 Score (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Configuration', fontsize=13, fontweight='bold')
ax.set_title('F1 Score Improvement through Hybrid Layers', fontsize=15, fontweight='bold')
ax.set_ylim(80, 95)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_FOLDER}/hybrid_f1_improvement.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {OUTPUT_FOLDER}/hybrid_f1_improvement.png")


# ============================================================
# GRAPH 4: Combined - Model Comparison + Hybrid Improvement
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: Model Comparison (other models)
ax1 = axes[0]
models = ['Random Forest\n(Ours)', 'XGBoost', 'Gradient\nBoosting', 'Neural\nNetwork', 
          'KNN', 'SVM', 'Decision\nTree', 'Logistic\nRegression', 'AdaBoost']
f1_scores = [86.66, 86.62, 85.57, 85.15, 83.67, 82.14, 82.08, 77.99, 77.79]
colors = ['#4caf50'] + ['#1a73e8']*8

bars = ax1.barh(models, f1_scores, color=colors, edgecolor='black')
ax1.set_xlabel('F1 Score (%)', fontsize=12, fontweight='bold')
ax1.set_title('A. Model Comparison\n(9 Models Tested)', fontsize=13, fontweight='bold')
ax1.invert_yaxis()
ax1.set_xlim(70, 90)
for bar, val in zip(bars, f1_scores):
    ax1.text(val + 0.3, bar.get_y() + bar.get_height()/2, f'{val}%', 
             va='center', fontsize=9)

# Right: Hybrid Improvement
ax2 = axes[1]
bars = ax2.bar(configs, f1, color=['#ff9800', '#4caf50', '#2e7d32'], 
               edgecolor='black', linewidth=2, width=0.5)
ax2.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
ax2.set_title('B. Hybrid Model Improvement\n(RF → Hybrid → Hybrid + LLM)', 
              fontsize=13, fontweight='bold')
ax2.set_ylim(80, 95)
ax2.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, f1):
    ax2.text(bar.get_x() + bar.get_width()/2., val + 0.3,
             f'{val}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

# Add improvement arrows
ax2.annotate('', xy=(1, 87.6), xytext=(0, 85.8),
             arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax2.text(0.5, 86.7, '+1.8%', fontsize=10, color='green', fontweight='bold', ha='center')

ax2.annotate('', xy=(2, 91.4), xytext=(1, 87.6),
             arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax2.text(1.5, 89.5, '+3.8%', fontsize=10, color='green', fontweight='bold', ha='center')

plt.suptitle('Complete Model Performance Analysis', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_FOLDER}/complete_performance_analysis.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {OUTPUT_FOLDER}/complete_performance_analysis.png")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("✅ ALL HYBRID PERFORMANCE GRAPHS CREATED!")
print("="*70)
print(f"\n📁 Files saved in: {OUTPUT_FOLDER}/")
print("\n📊 New graphs:")
print(f"   ├── hybrid_performance_bar.png        - Bar chart")
print(f"   ├── hybrid_performance_line.png       - Line chart")
print(f"   ├── hybrid_f1_improvement.png         - F1 improvement highlight")
print(f"   └── complete_performance_analysis.png - Combined graph")

print("\n" + "="*70)
print("KEY NUMBERS TO REMEMBER")
print("="*70)
print(f"""
Random Forest Only:
   Accuracy: {accuracy[0]}%  |  F1: {f1[0]}%

Hybrid Model (CBA + RF):
   Accuracy: {accuracy[1]}%  |  F1: {f1[1]}%  (Improvement: +{f1[1]-f1[0]:.1f}%)

Hybrid + LLM (Full):
   Accuracy: {accuracy[2]}%  |  F1: {f1[2]}%  (Improvement: +{f1[2]-f1[1]:.1f}%)

TOTAL IMPROVEMENT: +{f1[2]-f1[0]:.1f}% in F1 Score
""")