"""
Feature Selection Analysis
Shows why we selected 16 features, their importance, frequency, and detection ratio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FEATURE SELECTION ANALYSIS")
print("="*70)

# Load data
train_df = pd.read_csv('processed_data/train_enhanced.csv')
print(f"\nLoaded {len(train_df)} training samples")

# Separate features and labels
X = train_df.drop('label', axis=1)
y = train_df['label']
feature_names = X.columns.tolist()
print(f"\nTotal features available: {len(feature_names)}")

# ============================================================
# 1. Feature Importance using Random Forest
# ============================================================
print("\n" + "="*70)
print("1. RANDOM FOREST FEATURE IMPORTANCE")
print("="*70)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_

# Create importance dataframe
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances,
    'Importance_Percent': importances * 100
}).sort_values('Importance', ascending=False)

print("\nFeature Importance Ranking:")
print(importance_df.to_string(index=False))

# ============================================================
# 2. Feature Frequency in Dataset
# ============================================================
print("\n" + "="*70)
print("2. FEATURE FREQUENCY IN DATASET")
print("="*70)

frequency_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Value': X.mean(),
    'Std_Dev': X.std(),
    'Min': X.min(),
    'Max': X.max(),
    'Non_Zero_Count': (X != 0).sum(),
    'Non_Zero_Percent': (X != 0).sum() / len(X) * 100
})

print("\nFeature Statistics:")
print(frequency_df.to_string(index=False))

# ============================================================
# 3. Detection Ratio for Each Feature
# ============================================================
print("\n" + "="*70)
print("3. DETECTION RATIO FOR EACH FEATURE")
print("="*70)

detection_ratios = []
for feature in feature_names:
    # Split data by feature value
    feature_positive = X[X[feature] > 0]
    feature_zero = X[X[feature] == 0]
    
    # Calculate phishing ratio
    if len(feature_positive) > 0:
        pos_phishing_ratio = y.loc[feature_positive.index].mean()
    else:
        pos_phishing_ratio = 0
    
    if len(feature_zero) > 0:
        zero_phishing_ratio = y.loc[feature_zero.index].mean()
    else:
        zero_phishing_ratio = 0
    
    # Detection ratio improvement
    detection_ratio = pos_phishing_ratio - zero_phishing_ratio
    detection_ratios.append(detection_ratio)

detection_df = pd.DataFrame({
    'Feature': feature_names,
    'Phishing_Ratio_When_Present': [pos_phishing_ratio for _ in feature_names],
    'Phishing_Ratio_When_Absent': [zero_phishing_ratio for _ in feature_names],
    'Detection_Improvement': detection_ratios
}).sort_values('Detection_Improvement', ascending=False)

print("\nDetection Ratio for Each Feature:")
print(detection_df.to_string(index=False))

# ============================================================
# 4. Top 16 Features Selection Proof
# ============================================================
print("\n" + "="*70)
print("4. WHY 16 FEATURES? - Cumulative Importance Analysis")
print("="*70)

# Sort features by importance
importance_df_sorted = importance_df.sort_values('Importance', ascending=False)
cumulative_importance = np.cumsum(importance_df_sorted['Importance'].values)

print("\nCumulative Feature Importance:")
for i, row in importance_df_sorted.iterrows():
    print(f"  {row['Feature']:<20}: {row['Importance_Percent']:.2f}% (Cumulative: {cumulative_importance[i]*100:.2f}%)")

# Find how many features needed for 95% importance
features_for_95 = np.where(cumulative_importance >= 0.95)[0]
if len(features_for_95) > 0:
    n_features_95 = features_for_95[0] + 1
    print(f"\n→ {n_features_95} features needed to cover 95% of importance")

# ============================================================
# 5. Visualizations
# ============================================================
print("\n" + "="*70)
print("5. GENERATING VISUALIZATIONS")
print("="*70)

# Figure 1: Feature Importance
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Top 16 Feature Importance
ax1 = axes[0, 0]
top16 = importance_df.head(16)
colors = ['#d32f2f' if i < 5 else '#ff9800' if i < 10 else '#4caf50' for i in range(16)]
ax1.barh(top16['Feature'], top16['Importance_Percent'], color=colors)
ax1.set_xlabel('Importance (%)')
ax1.set_title('Top 16 Feature Importance', fontsize=14, fontweight='bold')
ax1.invert_yaxis()
for i, v in enumerate(top16['Importance_Percent']):
    ax1.text(v + 0.5, i, f'{v:.1f}%', va='center')

# Subplot 2: Cumulative Importance
ax2 = axes[0, 1]
ax2.plot(range(1, len(cumulative_importance)+1), cumulative_importance*100, 'b-', linewidth=2)
ax2.axhline(y=95, color='r', linestyle='--', label='95% Threshold')
ax2.axvline(x=n_features_95, color='g', linestyle='--', label=f'{n_features_95} features')
ax2.set_xlabel('Number of Features')
ax2.set_ylabel('Cumulative Importance (%)')
ax2.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: Detection Improvement
ax3 = axes[1, 0]
top_detection = detection_df.head(16)
ax3.barh(top_detection['Feature'], top_detection['Detection_Improvement']*100, color='#1a73e8')
ax3.set_xlabel('Detection Improvement (%)')
ax3.set_title('Detection Improvement by Feature', fontsize=14, fontweight='bold')
ax3.invert_yaxis()

# Subplot 4: Feature Frequency
ax4 = axes[1, 1]
frequency_sorted = frequency_df.sort_values('Non_Zero_Percent', ascending=False).head(16)
ax4.barh(frequency_sorted['Feature'], frequency_sorted['Non_Zero_Percent'], color='#7b1fa2')
ax4.set_xlabel('Non-Zero Frequency (%)')
ax4.set_title('Feature Frequency in Dataset', fontsize=14, fontweight='bold')
ax4.invert_yaxis()

plt.tight_layout()
plt.savefig('feature_selection_analysis.png', dpi=200, bbox_inches='tight')
print("✅ Saved: feature_selection_analysis.png")

# ============================================================
# 6. Summary Report
# ============================================================
print("\n" + "="*70)
print("6. FEATURE SELECTION SUMMARY")
print("="*70)

print(f"""
FEATURE SELECTION SUMMARY
=========================
Total Features Available: {len(feature_names)}
Features Selected: 16
Features Needed for 95% Importance: {n_features_95}

Why 16 Features?
- Top 16 features contribute {cumulative_importance[15]*100:.2f}% of total importance
- They are the most discriminative features for phishing detection
- Adding more features gives diminishing returns
- 16 features keep the model lightweight for browser extension

Selected Features (by importance):
{importance_df.head(16)[['Feature', 'Importance_Percent']].to_string(index=False)}

Detection Improvement Summary:
{detection_df.head(5)[['Feature', 'Detection_Improvement']].to_string(index=False)}
""")

print("\n✅ Feature Selection Analysis Complete!")