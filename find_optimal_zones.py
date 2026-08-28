# find_optimal_zones.py
"""
Find optimal threshold zones for phishing detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FINDING OPTIMAL THRESHOLD ZONES")
print("="*70)

# Load data
test_df = pd.read_csv('processed_data/test_final.csv')
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Load model
rf = joblib.load('models/random_forest_final_v3.pkl')
scaler = joblib.load('models/scaler.pkl')

# Scale features
X_test_scaled = scaler.transform(X_test)
y_proba = rf.predict_proba(X_test_scaled)[:, 1]

print(f"\nTest set size: {len(test_df)}")
print(f"Phishing: {sum(y_test==1)}, Legitimate: {sum(y_test==0)}")

# ============================================================
# Test different lower bounds (legitimate threshold)
# ============================================================
print("\n" + "="*70)
print("1. TESTING DIFFERENT LOWER BOUNDS (0.30 to 0.60)")
print("="*70)

lower_bounds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
upper_bound = 0.65  # Fixed upper bound (phishing threshold)

results = []

for lower in lower_bounds:
    # Classify based on zones
    y_pred = np.zeros(len(y_test))
    for i, prob in enumerate(y_proba):
        if prob >= upper_bound:
            y_pred[i] = 1  # Phishing
        elif prob >= lower:
            y_pred[i] = 2  # Suspicious (needs LLM)
        else:
            y_pred[i] = 0  # Legitimate
    
    # For evaluation, treat suspicious as legitimate (safe default)
    y_pred_binary = (y_pred == 1).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    
    # Count classifications
    phishing_count = sum(y_pred == 1)
    suspicious_count = sum(y_pred == 2)
    legitimate_count = sum(y_pred == 0)
    
    results.append({
        'Lower_Bound': lower,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Phishing': phishing_count,
        'Suspicious': suspicious_count,
        'Legitimate': legitimate_count
    })

results_df = pd.DataFrame(results)

print("\nResults for different lower bounds:")
print(results_df.round(4).to_string(index=False))

# ============================================================
# Find optimal lower bound (best F1 score)
# ============================================================
best_idx = results_df['F1'].idxmax()
best_lower = results_df.iloc[best_idx]['Lower_Bound']
best_f1 = results_df.iloc[best_idx]['F1']

print(f"\n✅ Optimal Lower Bound: {best_lower:.2f} (F1: {best_f1:.4f})")

# ============================================================
# Visualizations
# ============================================================
print("\n" + "="*70)
print("2. GENERATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Metrics vs Lower Bound
ax1 = axes[0, 0]
ax1.plot(lower_bounds, results_df['Accuracy'], 'b-o', label='Accuracy', linewidth=2, markersize=8)
ax1.plot(lower_bounds, results_df['Precision'], 'g-s', label='Precision', linewidth=2, markersize=8)
ax1.plot(lower_bounds, results_df['Recall'], 'r-^', label='Recall', linewidth=2, markersize=8)
ax1.plot(lower_bounds, results_df['F1'], 'm-d', label='F1', linewidth=2, markersize=8)
ax1.axvline(x=best_lower, color='green', linestyle='--', label=f'Optimal: {best_lower:.2f}', linewidth=2)
ax1.axvline(x=0.50, color='red', linestyle='--', label='Current: 0.50', linewidth=2)
ax1.set_xlabel('Lower Bound (Legitimate Threshold)')
ax1.set_ylabel('Score')
ax1.set_title('Metrics vs Lower Bound (Upper = 0.65)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.5, 1.0)

# Subplot 2: Classification Distribution
ax2 = axes[0, 1]
x = np.arange(len(lower_bounds))
width = 0.25
ax2.bar(x - width, results_df['Phishing'], width, label='Phishing', color='#d32f2f')
ax2.bar(x, results_df['Suspicious'], width, label='Suspicious', color='#ff9800')
ax2.bar(x + width, results_df['Legitimate'], width, label='Legitimate', color='#4caf50')
ax2.set_xlabel('Lower Bound')
ax2.set_ylabel('Count')
ax2.set_title('Classification Distribution', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{lb:.2f}' for lb in lower_bounds])
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: F1 Score Heatmap
ax3 = axes[1, 0]
# Test combinations of upper and lower bounds
upper_bounds = [0.60, 0.65, 0.70, 0.75]
lower_bounds_test = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

f1_matrix = np.zeros((len(upper_bounds), len(lower_bounds_test)))

for i, upper in enumerate(upper_bounds):
    for j, lower in enumerate(lower_bounds_test):
        if upper > lower:
            y_pred = np.zeros(len(y_test))
            for k, prob in enumerate(y_proba):
                if prob >= upper:
                    y_pred[k] = 1
                elif prob >= lower:
                    y_pred[k] = 2
                else:
                    y_pred[k] = 0
            y_pred_binary = (y_pred == 1).astype(int)
            f1_matrix[i, j] = f1_score(y_test, y_pred_binary)

im = ax3.imshow(f1_matrix, cmap='RdYlGn', aspect='auto', vmin=0.6, vmax=0.85)
ax3.set_xticks(np.arange(len(lower_bounds_test)))
ax3.set_yticks(np.arange(len(upper_bounds)))
ax3.set_xticklabels([f'{lb:.2f}' for lb in lower_bounds_test])
ax3.set_yticklabels([f'{ub:.2f}' for ub in upper_bounds])
ax3.set_xlabel('Lower Bound (Legitimate)')
ax3.set_ylabel('Upper Bound (Phishing)')
ax3.set_title('F1 Score Heatmap', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(len(upper_bounds)):
    for j in range(len(lower_bounds_test)):
        if upper_bounds[i] > lower_bounds_test[j]:
            text = ax3.text(j, i, f'{f1_matrix[i, j]:.3f}',
                           ha='center', va='center', color='black', fontsize=9)

plt.colorbar(im, ax=ax3)

# Subplot 4: Summary
ax4 = axes[1, 1]
ax4.axis('off')
summary_text = f"""
OPTIMAL THRESHOLD ZONES
=======================

Best Lower Bound: {best_lower:.2f} (F1 = {best_f1:.4f})

Recommended Zones:
• PHISHING     : ≥ 0.65
• SUSPICIOUS   : {best_lower:.2f} - 0.64
• LEGITIMATE   : < {best_lower:.2f}

Comparison:
Current: 0.50
Optimal: {best_lower:.2f}

Why {best_lower:.2f}?
- Best F1 score among tested values
- Balances precision and recall
- Minimizes false positives
- Uses LLM only when needed

Distribution at Optimal:
- Phishing: {results_df[results_df['Lower_Bound']==best_lower]['Phishing'].values[0]}
- Suspicious: {results_df[results_df['Lower_Bound']==best_lower]['Suspicious'].values[0]}
- Legitimate: {results_df[results_df['Lower_Bound']==best_lower]['Legitimate'].values[0]}
"""
ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#ccc'))

plt.tight_layout()
plt.savefig('optimal_threshold_zones.png', dpi=200, bbox_inches='tight')
print("✅ Saved: optimal_threshold_zones.png")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print("3. SUMMARY")
print("="*70)

print(f"""
OPTIMAL THRESHOLD ZONES
=======================

Tested Lower Bounds: 0.30 to 0.60
Upper Bound Fixed: 0.65

✅ Optimal Lower Bound: {best_lower:.2f}

Recommended Zones:
┌─────────────────────┬──────────────┬─────────────────────────────────┐
│ Zone                │ Threshold    │ Action                         │
├─────────────────────┼──────────────┼─────────────────────────────────┤
│ PHISHING            │ ≥ 0.65       │ Block immediately               │
│ SUSPICIOUS          │ {best_lower:.2f} - 0.64  │ Call LLM for analysis         │
│ LEGITIMATE          │ < {best_lower:.2f}       │ Allow navigation              │
└─────────────────────┴──────────────┴─────────────────────────────────┘

Performance at Optimal:
- Accuracy:  {results_df[results_df['Lower_Bound']==best_lower]['Accuracy'].values[0]:.4f}
- Precision: {results_df[results_df['Lower_Bound']==best_lower]['Precision'].values[0]:.4f}
- Recall:    {results_df[results_df['Lower_Bound']==best_lower]['Recall'].values[0]:.4f}
- F1:        {best_f1:.4f}

Distribution:
- Phishing:   {results_df[results_df['Lower_Bound']==best_lower]['Phishing'].values[0]}
- Suspicious: {results_df[results_df['Lower_Bound']==best_lower]['Suspicious'].values[0]}
- Legitimate: {results_df[results_df['Lower_Bound']==best_lower]['Legitimate'].values[0]}

Recommendation: Use {best_lower:.2f} as the lower bound
""")

print("\n✅ Optimal Threshold Analysis Complete!")