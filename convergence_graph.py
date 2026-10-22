# convergence_graph.py
"""
Convergence Graph - All Models Meeting at a Point
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost not installed. Skipping XGBoost.")

print("="*70)
print("CONVERGENCE GRAPH - ALL MODELS MEETING AT A POINT")
print("="*70)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_csv('processed_data/train_enhanced.csv')
test_df = pd.read_csv('processed_data/test_final.csv')

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ============================================================
# 2. DEFINE MODELS
# ============================================================
models = {}

# Random Forest (Your Model)
models['Random Forest (Yours)'] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# XGBoost (if available)
if HAS_XGBOOST:
    models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')

# Gradient Boosting
models['Gradient Boosting'] = GradientBoostingClassifier(n_estimators=100, random_state=42)

# SVM
models['SVM'] = SVC(kernel='rbf', probability=True, random_state=42)

# Neural Network
models['Neural Network'] = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)

print(f"\n📊 Models to train: {list(models.keys())}")

# ============================================================
# 3. GENERATE CONVERGENCE DATA
# ============================================================
print("\n📊 Training models on different data sizes...")

# Training percentages
train_sizes = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
convergence_data = {}

for name, model in models.items():
    print(f"   Training: {name}")
    scores = []
    
    for size in train_sizes:
        n_samples = int(len(X_train) * size)
        X_sub = X_train_scaled[:n_samples]
        y_sub = y_train[:n_samples]
        
        model.fit(X_sub, y_sub)
        y_pred = model.predict(X_test_scaled)
        f1 = f1_score(y_test, y_pred) * 100
        scores.append(f1)
    
    convergence_data[name] = scores

# ============================================================
# 4. FIND CONVERGENCE POINT
# ============================================================
# Where all models become close to each other
all_scores = np.array(list(convergence_data.values()))
avg_scores = np.mean(all_scores, axis=0)

# Find where variance between models is minimal
variance = np.var(all_scores, axis=0)
min_var_idx = np.argmin(variance)
convergence_point = train_sizes[min_var_idx] * 100
convergence_value = avg_scores[min_var_idx]

print(f"\n📌 Convergence Point: {convergence_point:.0f}% training data")
print(f"   F1 Score at Convergence: {convergence_value:.1f}%")

# ============================================================
# 5. PLOT CONVERGENCE GRAPH
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))

# Colors for models
colors = {
    'Random Forest (Yours)': '#4caf50',
    'XGBoost': '#1a73e8',
    'Gradient Boosting': '#ff9800',
    'SVM': '#9e9e9e',
    'Neural Network': '#7b1fa2'
}

# Plot each model
for name, scores in convergence_data.items():
    ax.plot([s*100 for s in train_sizes], scores, 
            label=name, color=colors.get(name, '#333'),
            linewidth=2.5, marker='o', markersize=6)

# Mark convergence point
ax.axvline(x=convergence_point, color='red', linestyle='--', linewidth=2.5)
ax.axhline(y=convergence_value, color='red', linestyle='--', linewidth=2.5)

# Add convergence point marker
ax.plot(convergence_point, convergence_value, 'ro', markersize=12, zorder=10)
ax.text(convergence_point + 2, convergence_value + 1.5, 
        f'Convergence Point\n{convergence_point:.0f}% Data\nF1: {convergence_value:.1f}%',
        fontsize=10, fontweight='bold', color='red',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9))

# Shade convergence zone
ax.axvspan(convergence_point - 5, convergence_point + 5, alpha=0.15, color='red', label='Convergence Zone')

# Labels and title
ax.set_xlabel('Training Data (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('F1 Score (%)', fontsize=14, fontweight='bold')
ax.set_title('CONVERGENCE ANALYSIS - All Models Meeting at a Point', 
             fontsize=16, fontweight='bold', color='#1a1a2e')

# Legend
ax.legend(loc='lower right', fontsize=10)

# Grid
ax.grid(True, alpha=0.3)

# Set limits
ax.set_xlim(0, 105)
ax.set_ylim(45, 95)

# Add convergence annotation
ax.text(105, 48, 
        f'Convergence Point: {convergence_point:.0f}%\n'
        f'F1 Score: {convergence_value:.1f}%\n'
        f'Models: {len(models)}',
        fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#ccc'))

# ============================================================
# 6. CREATE OUTPUT FOLDER AND SAVE
# ============================================================
import os
output_folder = 'validation_graphs'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

plt.tight_layout()
plt.savefig(f'{output_folder}/convergence_graph.png', dpi=200, bbox_inches='tight')
print(f"\n✅ Convergence graph saved to {output_folder}/convergence_graph.png")

# ============================================================
# 7. CONVERGENCE SUMMARY TABLE
# ============================================================
print("\n" + "="*70)
print("CONVERGENCE SUMMARY")
print("="*70)

for i, size in enumerate(train_sizes):
    scores_at_size = [data[i] for data in convergence_data.values()]
    diff = max(scores_at_size) - min(scores_at_size)
    print(f"  {size*100:5.0f}% Data:  F1 Range = {min(scores_at_size):.1f}% - {max(scores_at_size):.1f}%  (Diff: {diff:.1f}%)")

print(f"\n📌 Convergence Point: {convergence_point:.0f}% training data")
print(f"   All models reach approximately {convergence_value:.1f}% F1 score")
print(f"   At this point, the difference between models is minimal")

# Save convergence results
conv_df = pd.DataFrame({
    'Model': list(convergence_data.keys()),
    'Convergence_F1': [scores[min_var_idx] for scores in convergence_data.values()],
    'Final_F1': [scores[-1] for scores in convergence_data.values()],
    'Improvement': [scores[-1] - scores[min_var_idx] for scores in convergence_data.values()]
})
conv_df.to_csv(f'{output_folder}/convergence_results.csv', index=False)
print(f"\n✅ Convergence results saved to {output_folder}/convergence_results.csv")

# ============================================================
# 8. SHOW ON SCREEN
# ============================================================
plt.show()

print("\n✅ Analysis Complete!")
print("="*70)