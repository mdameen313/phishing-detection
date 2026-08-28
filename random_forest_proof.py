"""
Random Forest Proof
Complete model validation with metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("RANDOM FOREST - COMPLETE VALIDATION")
print("="*70)

# Load data
train_df = pd.read_csv('processed_data/train_enhanced.csv')
val_df = pd.read_csv('processed_data/val_final.csv')
test_df = pd.read_csv('processed_data/test_final.csv')

print(f"\nTraining: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")

# Prepare data
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_val = val_df.drop('label', axis=1)
y_val = val_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 1. Train Random Forest
# ============================================================
print("\n" + "="*70)
print("1. TRAINING RANDOM FOREST")
print("="*70)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)

print("\nModel Parameters:")
print(f"  Trees: {rf.n_estimators}")
print(f"  Max Depth: {rf.max_depth}")
print(f"  Features: {X_train.shape[1]}")

# ============================================================
# 2. Cross-Validation
# ============================================================
print("\n" + "="*70)
print("2. CROSS-VALIDATION")
print("="*70)

cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='f1')
print(f"\n5-Fold Cross-Validation F1 Scores: {cv_scores}")
print(f"Mean CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# 3. Test Set Evaluation
# ============================================================
print("\n" + "="*70)
print("3. TEST SET EVALUATION")
print("="*70)

y_pred = rf.predict(X_test_scaled)
y_proba = rf.predict_proba(X_test_scaled)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\nTest Set Metrics:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {cm[0,0]:,}")
print(f"  False Positives: {cm[0,1]:,}")
print(f"  False Negatives: {cm[1,0]:,}")
print(f"  True Positives:  {cm[1,1]:,}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

# ============================================================
# 4. Feature Importance
# ============================================================
print("\n" + "="*70)
print("4. FEATURE IMPORTANCE")
print("="*70)

feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
for i, row in feature_importance.iterrows():
    print(f"  {row['Feature']:<20}: {row['Importance']*100:.2f}%")

# ============================================================
# 5. Visualizations
# ============================================================
print("\n" + "="*70)
print("5. GENERATING VISUALIZATIONS")
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
ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
for i in range(2):
    for j in range(2):
        ax1.text(j, i, cm[i, j], ha='center', va='center', fontsize=14, fontweight='bold')

# Subplot 2: ROC Curve
ax2 = axes[0, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
ax2.plot([0, 1], [0, 1], 'r--', label='Random')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: Precision-Recall Curve
ax3 = axes[0, 2]
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
ax3.plot(recall_vals, precision_vals, 'g-', linewidth=2)
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Subplot 4: Feature Importance
ax4 = axes[1, 0]
top10 = feature_importance.head(10)
colors = ['#d32f2f' if i < 3 else '#ff9800' if i < 7 else '#4caf50' for i in range(10)]
ax4.barh(top10['Feature'], top10['Importance']*100, color=colors)
ax4.set_xlabel('Importance (%)')
ax4.set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
ax4.invert_yaxis()
for i, v in enumerate(top10['Importance']*100):
    ax4.text(v + 0.5, i, f'{v:.2f}%', va='center')

# Subplot 5: CV Scores
ax5 = axes[1, 1]
ax5.bar(range(1, 6), cv_scores, color='#1a73e8')
ax5.axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
ax5.fill_between([0.5, 5.5], cv_scores.mean() - cv_scores.std(), cv_scores.mean() + cv_scores.std(), 
                 alpha=0.2, color='red', label=f'±1 Std: {cv_scores.std():.4f}')
ax5.set_xlabel('Fold')
ax5.set_ylabel('F1 Score')
ax5.set_title('Cross-Validation Scores', fontsize=14, fontweight='bold')
ax5.set_ylim(0.7, 1.0)
ax5.legend()
ax5.grid(True, alpha=0.3)

# Subplot 6: Metrics Comparison
ax6 = axes[1, 2]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
values = [accuracy, precision, recall, f1, roc_auc]
colors = ['#4caf50', '#1a73e8', '#ff9800', '#7b1fa2', '#d32f2f']
bars = ax6.bar(metrics, values, color=colors)
ax6.set_ylim(0.5, 1.0)
ax6.set_ylabel('Score')
ax6.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
for bar, val in zip(bars, values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.4f}', 
             ha='center', fontsize=10, fontweight='bold')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('random_forest_proof.png', dpi=200, bbox_inches='tight')
print("✅ Saved: random_forest_proof.png")

# ============================================================
# 6. Summary Report
# ============================================================
print("\n" + "="*70)
print("6. RANDOM FOREST SUMMARY")
print("="*70)

print(f"""
RANDOM FOREST SUMMARY
====================
Model Configuration:
  - Trees: {rf.n_estimators}
  - Max Depth: {rf.max_depth}
  - Min Samples Split: {rf.min_samples_split}
  - Min Samples Leaf: {rf.min_samples_leaf}
  - Class Weight: Balanced

Cross-Validation:
  - 5-Fold F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}

Test Set Performance:
  - Accuracy:  {accuracy:.4f}
  - Precision: {precision:.4f}
  - Recall:    {recall:.4f}
  - F1-Score:  {f1:.4f}
  - ROC-AUC:   {roc_auc:.4f}

Confusion Matrix:
  - True Negatives:  {cm[0,0]:,}
  - False Positives: {cm[0,1]:,}
  - False Negatives: {cm[1,0]:,}
  - True Positives:  {cm[1,1]:,}

Top 5 Features:
{feature_importance.head(5)[['Feature', 'Importance']].to_string(index=False)}
""")

print("\n✅ Random Forest Proof Complete!")