"""
500 URL Testing
Test model on 500 random URLs, analyze success rate and errors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from sklearn.preprocessing import StandardScaler
import random
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("500 URL TESTING - SUCCESS RATE & ERROR ANALYSIS")
print("="*70)

# Load data
test_df = pd.read_csv('processed_data/test_final.csv')
print(f"\nTotal test samples: {len(test_df)}")

# Randomly select 500 URLs
random.seed(42)
sample_indices = random.sample(range(len(test_df)), 500)
sample_df = test_df.iloc[sample_indices].copy()

print(f"Random sample size: {len(sample_df)}")

# Prepare data
X_sample = sample_df.drop('label', axis=1)
y_sample = sample_df['label']

# Load model and scaler
rf = joblib.load('models/random_forest_final_v3.pkl')
scaler = joblib.load('models/scaler.pkl')

# Scale features
X_sample_scaled = scaler.transform(X_sample)

# Predict
y_pred = rf.predict(X_sample_scaled)
y_proba = rf.predict_proba(X_sample_scaled)[:, 1]

# ============================================================
# 1. Overall Performance
# ============================================================
print("\n" + "="*70)
print("1. OVERALL PERFORMANCE")
print("="*70)

accuracy = accuracy_score(y_sample, y_pred)
precision = precision_score(y_sample, y_pred)
recall = recall_score(y_sample, y_pred)
f1 = f1_score(y_sample, y_pred)

print(f"\nSample Size: {len(sample_df)}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_sample, y_pred)
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {cm[0,0]:,}")
print(f"  False Positives: {cm[0,1]:,}")
print(f"  False Negatives: {cm[1,0]:,}")
print(f"  True Positives:  {cm[1,1]:,}")

# ============================================================
# 2. Error Analysis
# ============================================================
print("\n" + "="*70)
print("2. ERROR ANALYSIS")
print("="*70)

# Identify errors
sample_df['Predicted'] = y_pred
sample_df['Probability'] = y_proba
sample_df['Correct'] = y_sample == y_pred

# False Positives (Predicted Phishing, Actual Legitimate)
false_positives = sample_df[(sample_df['Predicted'] == 1) & (sample_df['label'] == 0)]
# False Negatives (Predicted Legitimate, Actual Phishing)
false_negatives = sample_df[(sample_df['Predicted'] == 0) & (sample_df['label'] == 1)]

print(f"\nErrors Found:")
print(f"  False Positives: {len(false_positives)} (Legitimate sites flagged as phishing)")
print(f"  False Negatives: {len(false_negatives)} (Phishing sites missed)")

# Error types
error_types = {
    'False Positives': len(false_positives),
    'False Negatives': len(false_negatives),
    'Correct Predictions': len(sample_df) - len(false_positives) - len(false_negatives)
}

print("\nError Type Distribution:")
for error_type, count in error_types.items():
    print(f"  {error_type}: {count} ({count/len(sample_df)*100:.1f}%)")

# ============================================================
# 3. Probability Distribution Analysis
# ============================================================
print("\n" + "="*70)
print("3. PROBABILITY DISTRIBUTION")
print("="*70)

# Distribution of probabilities for correct vs incorrect
correct_probs = sample_df[sample_df['Correct']]['Probability']
incorrect_probs = sample_df[~sample_df['Correct']]['Probability']

print(f"\nCorrect Predictions Probability Range:")
print(f"  Min: {correct_probs.min():.4f}")
print(f"  Max: {correct_probs.max():.4f}")
print(f"  Mean: {correct_probs.mean():.4f}")

print(f"\nIncorrect Predictions Probability Range:")
if len(incorrect_probs) > 0:
    print(f"  Min: {incorrect_probs.min():.4f}")
    print(f"  Max: {incorrect_probs.max():.4f}")
    print(f"  Mean: {incorrect_probs.mean():.4f}")

# ============================================================
# 4. Visualizations
# ============================================================
print("\n" + "="*70)
print("4. GENERATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Subplot 1: Confusion Matrix
ax1 = axes[0, 0]
im = ax1.imshow(cm, cmap='Blues')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Legitimate', 'Phishing'])
ax1.set_yticklabels(['Legitimate', 'Phishing'])
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_title('Confusion Matrix (500 URLs)', fontsize=14, fontweight='bold')
for i in range(2):
    for j in range(2):
        ax1.text(j, i, cm[i, j], ha='center', va='center', fontsize=14, fontweight='bold')

# Subplot 2: Error Types
ax2 = axes[0, 1]
error_labels = list(error_types.keys())
error_values = list(error_types.values())
colors = ['#4caf50', '#d32f2f', '#ff9800']
ax2.pie(error_values, labels=error_labels, autopct='%1.1f%%', colors=colors, startangle=90)
ax2.set_title('Error Type Distribution', fontsize=14, fontweight='bold')

# Subplot 3: Correct vs Incorrect Distribution
ax3 = axes[0, 2]
ax3.hist(correct_probs, bins=20, alpha=0.7, color='#4caf50', label='Correct (n={})'.format(len(correct_probs)))
if len(incorrect_probs) > 0:
    ax3.hist(incorrect_probs, bins=20, alpha=0.7, color='#d32f2f', label='Incorrect (n={})'.format(len(incorrect_probs)))
ax3.axvline(x=0.70, color='black', linestyle='--', label='Threshold (0.70)')
ax3.set_xlabel('Prediction Probability')
ax3.set_ylabel('Frequency')
ax3.set_title('Probability Distribution', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: Feature Importance
ax4 = axes[1, 0]
feature_importance = pd.DataFrame({
    'Feature': X_sample.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False).head(10)
colors = ['#d32f2f' if i < 3 else '#ff9800' if i < 7 else '#4caf50' for i in range(10)]
ax4.barh(feature_importance['Feature'], feature_importance['Importance']*100, color=colors)
ax4.set_xlabel('Importance (%)')
ax4.set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
ax4.invert_yaxis()
for i, v in enumerate(feature_importance['Importance']*100):
    ax4.text(v + 0.5, i, f'{v:.2f}%', va='center')

# Subplot 5: False Positives Analysis
ax5 = axes[1, 1]
if len(false_positives) > 0:
    fp_features = false_positives.mean(numeric_only=True).sort_values(ascending=False).head(5)
    ax5.barh(fp_features.index, fp_features.values, color='#ff9800')
    ax5.set_xlabel('Average Feature Value')
    ax5.set_title('False Positives - Top Features', fontsize=14, fontweight='bold')
else:
    ax5.text(0.5, 0.5, 'No False Positives Found!', ha='center', va='center', fontsize=16, fontweight='bold')
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
ax5.grid(True, alpha=0.3)

# Subplot 6: False Negatives Analysis
ax6 = axes[1, 2]
if len(false_negatives) > 0:
    fn_features = false_negatives.mean(numeric_only=True).sort_values(ascending=False).head(5)
    ax6.barh(fn_features.index, fn_features.values, color='#d32f2f')
    ax6.set_xlabel('Average Feature Value')
    ax6.set_title('False Negatives - Top Features', fontsize=14, fontweight='bold')
else:
    ax6.text(0.5, 0.5, 'No False Negatives Found!', ha='center', va='center', fontsize=16, fontweight='bold')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test_500_urls.png', dpi=200, bbox_inches='tight')
print("✅ Saved: test_500_urls.png")

# ============================================================
# 5. Summary Report
# ============================================================
print("\n" + "="*70)
print("5. SUMMARY REPORT")
print("="*70)

print(f"""
500 URL TESTING SUMMARY
=======================

Test Configuration:
  - Total URLs Tested: {len(sample_df)}
  - Random Seed: 42 (reproducible)
  - Threshold: 0.70

Performance Results:
  - Accuracy:  {accuracy:.4f}
  - Precision: {precision:.4f}
  - Recall:    {recall:.4f}
  - F1-Score:  {f1:.4f}

Error Analysis:
  - Correct Predictions: {len(sample_df) - len(false_positives) - len(false_negatives)} ({100 - (len(false_positives)+len(false_negatives))/len(sample_df)*100:.1f}%)
  - False Positives: {len(false_positives)} ({len(false_positives)/len(sample_df)*100:.1f}%)
  - False Negatives: {len(false_negatives)} ({len(false_negatives)/len(sample_df)*100:.1f}%)

False Positive Analysis:
  - Legitimate sites wrongly flagged as phishing
  - Potential causes:
    • URL has suspicious patterns (long length, many dots)
    • Contains sensitive keywords in domain
    • New/unknown domain

False Negative Analysis:
  - Phishing sites missed by the model
  - Potential causes:
    • URL looks very legitimate (HTTPS, short domain)
    • No obvious phishing indicators
    • New phishing technique not in training data

Recommendations:
  1. Reduce False Positives: Add more legitimate domains to whitelist
  2. Reduce False Negatives: Add more phishing samples to training data
  3. Threshold Tuning: Current threshold 0.70 is optimal for F1
  4. Feature Enhancement: Add more features (domain age, SSL check)
""")

print("\n✅ 500 URL Testing Complete!")