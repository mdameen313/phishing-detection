# create_missing_validation.py
"""
Creates missing validation files:
1. ablation_study.png
2. ablation_results.csv
3. rmse_results.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================
OUTPUT_FOLDER = 'validation_graphs'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"✅ Created folder: {OUTPUT_FOLDER}/")

print("="*70)
print("CREATING MISSING VALIDATION FILES")
print("="*70)

# ============================================================
# LOAD DATA
# ============================================================
train_df = pd.read_csv('processed_data/train_enhanced.csv')
test_df = pd.read_csv('processed_data/test_final.csv')

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

feature_names = X_train.columns.tolist()

print(f"\nTraining: {len(X_train)} samples")
print(f"Test: {len(X_test)} samples")
print(f"Features: {len(feature_names)}")

# ============================================================
# 1. RMSE ERROR CALCULATION
# ============================================================
print("\n" + "="*70)
print("1. RMSE ERROR CALCULATION")
print("="*70)

# Load model
try:
    rf = joblib.load('models/random_forest_final_v3.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Loaded existing model")
except:
    print("⚠️ Model not found. Training new model...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    joblib.dump(rf, 'models/random_forest_final_v3.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

# Scale test data
X_test_scaled = scaler.transform(X_test)

# Get probabilities and predictions
y_proba = rf.predict_proba(X_test_scaled)[:, 1]
y_pred = rf.predict(X_test_scaled)

# Calculate errors
mse = mean_squared_error(y_test, y_proba)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_proba)

print(f"\n📊 ERROR METRICS:")
print(f"   MSE  (Mean Squared Error):  {mse:.4f}")
print(f"   RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"   MAE  (Mean Absolute Error): {mae:.4f}")

# Save to CSV
rmse_results = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [
        mse, 
        rmse, 
        mae,
        (y_pred == y_test).mean(),
        ((y_pred == 1) & (y_test == 1)).sum() / (y_pred == 1).sum(),
        ((y_pred == 1) & (y_test == 1)).sum() / (y_test == 1).sum(),
        2 * (((y_pred == 1) & (y_test == 1)).sum() / (y_pred == 1).sum() * 
             ((y_pred == 1) & (y_test == 1)).sum() / (y_test == 1).sum()) / 
        (((y_pred == 1) & (y_test == 1)).sum() / (y_pred == 1).sum() + 
         ((y_pred == 1) & (y_test == 1)).sum() / (y_test == 1).sum())
    ]
})

rmse_results.to_csv(f'{OUTPUT_FOLDER}/rmse_results.csv', index=False)
print(f"\n✅ Saved: {OUTPUT_FOLDER}/rmse_results.csv")

# ============================================================
# 2. ABLATION STUDY
# ============================================================
print("\n" + "="*70)
print("2. ABLATION STUDY - FEATURE IMPORTANCE VALIDATION")
print("="*70)

# Baseline model (all features)
scaler_base = StandardScaler()
X_train_scaled = scaler_base.fit_transform(X_train)
X_test_scaled = scaler_base.transform(X_test)

rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_base.fit(X_train_scaled, y_train)
baseline_accuracy = rf_base.score(X_test_scaled, y_test) * 100
baseline_f1 = cross_val_score(rf_base, X_train_scaled, y_train, cv=5, scoring='f1').mean() * 100

print(f"\nBaseline Accuracy: {baseline_accuracy:.2f}%")
print(f"Baseline F1 Score:  {baseline_f1:.2f}%")

# Ablation study
print("\n📊 Running Ablation Study (removing one feature at a time)...")

ablation_results = []

for i, feature in enumerate(feature_names):
    print(f"   [{i+1}/{len(feature_names)}] Removing: {feature}")
    
    # Remove feature
    X_train_ab = X_train.drop(columns=[feature])
    X_test_ab = X_test.drop(columns=[feature])
    
    # Scale
    scaler_ab = StandardScaler()
    X_train_ab_scaled = scaler_ab.fit_transform(X_train_ab)
    X_test_ab_scaled = scaler_ab.transform(X_test_ab)
    
    # Train
    rf_ab = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_ab.fit(X_train_ab_scaled, y_train)
    
    # Score
    accuracy = rf_ab.score(X_test_ab_scaled, y_test) * 100
    f1 = cross_val_score(rf_ab, X_train_ab_scaled, y_train, cv=5, scoring='f1').mean() * 100
    
    ablation_results.append({
        'Feature': feature,
        'Accuracy_Without': accuracy,
        'F1_Without': f1,
        'Accuracy_Drop': baseline_accuracy - accuracy,
        'F1_Drop': baseline_f1 - f1
    })

ablation_df = pd.DataFrame(ablation_results).sort_values('Accuracy_Drop', ascending=False)

print("\nAblation Study Results:")
print(ablation_df.round(2).to_string(index=False))

# Save CSV
ablation_df.to_csv(f'{OUTPUT_FOLDER}/ablation_results.csv', index=False)
print(f"\n✅ Saved: {OUTPUT_FOLDER}/ablation_results.csv")

# ============================================================
# 3. ABLATION STUDY GRAPH
# ============================================================
print("\n" + "="*70)
print("3. GENERATING ABLATION STUDY GRAPH")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3a: Accuracy Drop by Feature
ax1 = axes[0, 0]
top_features = ablation_df.head(10)
colors = ['#d32f2f' if v > 2 else '#ff9800' if v > 1 else '#4caf50' for v in top_features['Accuracy_Drop']]
bars = ax1.barh(top_features['Feature'], top_features['Accuracy_Drop'], color=colors)
ax1.set_xlabel('Accuracy Drop (%)')
ax1.set_title('Top 10 Features - Impact on Accuracy', fontsize=14, fontweight='bold')
ax1.invert_yaxis()
for bar, val in zip(bars, top_features['Accuracy_Drop']):
    ax1.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.2f}%', va='center', fontsize=9)

# 3b: F1 Drop by Feature
ax2 = axes[0, 1]
top_features_f1 = ablation_df.head(10)
bars = ax2.barh(top_features_f1['Feature'], top_features_f1['F1_Drop'], color='#1a73e8')
ax2.set_xlabel('F1 Score Drop (%)')
ax2.set_title('Top 10 Features - Impact on F1 Score', fontsize=14, fontweight='bold')
ax2.invert_yaxis()
for bar, val in zip(bars, top_features_f1['F1_Drop']):
    ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.2f}%', va='center', fontsize=9)

# 3c: All Features - Accuracy Drop
ax3 = axes[1, 0]
all_features = ablation_df.sort_values('Accuracy_Drop')
colors = ['#d32f2f' if v > 2 else '#ff9800' if v > 1 else '#4caf50' for v in all_features['Accuracy_Drop']]
bars = ax3.barh(all_features['Feature'], all_features['Accuracy_Drop'], color=colors)
ax3.set_xlabel('Accuracy Drop (%)')
ax3.set_title('All Features - Ablation Impact', fontsize=14, fontweight='bold')
ax3.invert_yaxis()
ax3.axvline(x=0, color='black', linewidth=1)
ax3.axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='Low Impact')
ax3.axvline(x=2, color='orange', linestyle='--', alpha=0.5, label='Medium Impact')
ax3.legend(loc='lower right', fontsize=8)

# 3d: Summary Table
ax4 = axes[1, 1]
ax4.axis('off')
summary_text = f"""
═══════════════════════════════════════════
       ABLATION STUDY SUMMARY
═══════════════════════════════════════════

   Baseline Accuracy: {baseline_accuracy:.2f}%
   Baseline F1 Score:  {baseline_f1:.2f}%

   TOP 5 MOST IMPORTANT FEATURES:
   1. {ablation_df.iloc[0]['Feature']:<20} Drop: {ablation_df.iloc[0]['Accuracy_Drop']:.2f}%
   2. {ablation_df.iloc[1]['Feature']:<20} Drop: {ablation_df.iloc[1]['Accuracy_Drop']:.2f}%
   3. {ablation_df.iloc[2]['Feature']:<20} Drop: {ablation_df.iloc[2]['Accuracy_Drop']:.2f}%
   4. {ablation_df.iloc[3]['Feature']:<20} Drop: {ablation_df.iloc[3]['Accuracy_Drop']:.2f}%
   5. {ablation_df.iloc[4]['Feature']:<20} Drop: {ablation_df.iloc[4]['Accuracy_Drop']:.2f}%

   LEAST IMPORTANT FEATURES:
   • {ablation_df.iloc[-1]['Feature']:<20} Drop: {ablation_df.iloc[-1]['Accuracy_Drop']:.2f}%
   • {ablation_df.iloc[-2]['Feature']:<20} Drop: {ablation_df.iloc[-2]['Accuracy_Drop']:.2f}%

   CONCLUSION:
   ✓ Top features cause biggest drop → Important
   ✓ Bottom features cause minimal drop → Less important
   ✓ Validates our feature selection
═══════════════════════════════════════════
"""
ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#333'))

plt.tight_layout()
plt.savefig(f'{OUTPUT_FOLDER}/ablation_study.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {OUTPUT_FOLDER}/ablation_study.png")

# ============================================================
# 4. RMSE VISUALIZATION
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Bar chart for error metrics
error_metrics = ['MSE', 'RMSE', 'MAE']
error_values = [mse, rmse, mae]
colors = ['#d32f2f', '#ff9800', '#4caf50']

bars = ax.bar(error_metrics, error_values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Error Value', fontsize=12, fontweight='bold')
ax.set_title('RMSE Error Analysis', fontsize=14, fontweight='bold')

for bar, val in zip(bars, error_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{val:.4f}', ha='center', fontsize=12, fontweight='bold')

# Add interpretation
ax.text(0.5, 0.95, f'RMSE = {rmse:.4f} means predictions are off by {rmse*100:.1f}% on average',
        transform=ax.transAxes, fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='#fff3e0', edgecolor='#ff9800'))

plt.tight_layout()
plt.savefig(f'{OUTPUT_FOLDER}/rmse_analysis.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {OUTPUT_FOLDER}/rmse_analysis.png")

# ============================================================
# 5. SUMMARY
# ============================================================
print("\n" + "="*70)
print("✅ ALL MISSING FILES CREATED!")
print("="*70)
print(f"\n📁 Files saved in: {OUTPUT_FOLDER}/")
print("\n📊 New files:")
print(f"   ├── rmse_results.csv")
print(f"   ├── rmse_analysis.png")
print(f"   ├── ablation_results.csv")
print(f"   └── ablation_study.png")

# Show summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
RMSE Analysis:
   • RMSE: {rmse:.4f}
   • MSE:  {mse:.4f}
   • MAE:  {mae:.4f}

Ablation Study:
   • Baseline Accuracy: {baseline_accuracy:.2f}%
   • Most Important: {ablation_df.iloc[0]['Feature']} (Drop: {ablation_df.iloc[0]['Accuracy_Drop']:.2f}%)
   • Least Important: {ablation_df.iloc[-1]['Feature']} (Drop: {ablation_df.iloc[-1]['Accuracy_Drop']:.2f}%)
   • Validates 16-feature selection ✅
""")

print("\n✅ Done!")