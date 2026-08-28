# test_500_validation.py
"""
Test on 500 random URLs to validate threshold change
"""

import sys
import os
import pandas as pd
import numpy as np
import random
import re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hybrid_model_fixed_v2 import HybridPhishingDetector

print("="*70)
print("VALIDATION TEST: 500 RANDOM URLs (Threshold = 0.65)")
print("="*70)

# ============================================================
# Load test data
# ============================================================
test_df = pd.read_csv('processed_data/test_final.csv')
print(f"\nLoaded {len(test_df)} test samples")

# ============================================================
# Randomly select 500 samples
# ============================================================
random.seed(42)
sample_indices = random.sample(range(len(test_df)), 500)
sample_df = test_df.iloc[sample_indices].copy()

print(f"Selected {len(sample_df)} random URLs")

# ============================================================
# Initialize detector
# ============================================================
detector = HybridPhishingDetector()

# ============================================================
# Get feature columns (exclude label)
# ============================================================
feature_cols = [col for col in sample_df.columns if col != 'label']

# ============================================================
# Test each URL
# ============================================================
print("\nTesting URLs...")
print("-"*70)

correct = 0
phishing_correct = 0
legit_correct = 0
phishing_total = 0
legit_total = 0

# Store results for analysis
results_list = []

for idx, row in sample_df.iterrows():
    # Get actual label
    actual_label = row['label']
    
    # Get features
    features = row[feature_cols].to_dict()
    
    # We need to reconstruct URL for prediction - we'll use the detector
    # Since we have features, we'll create a dummy URL and pass features directly
    
    # For testing, we'll use the detector's predict method with features
    # But we need a URL string. Let's create a placeholder
    # Since we don't have raw URLs, we'll use the predict_with_features method
    
    # Instead, let's load URLs from the processed data if available
    # We'll create a dummy URL based on features
    url_placeholder = f"https://test-{idx}.com"
    
    # For actual testing, we need to use the detector's predict method
    # But it expects a URL string. Let's modify the approach.
    
# ============================================================
# Better approach: Use the detector directly with features
# ============================================================
print("\n" + "="*70)
print("Testing with actual URLs from dataset...")
print("="*70)

# Since we don't have raw URLs in test_final.csv,
# let's use the existing test_urls.csv if it exists,
# or use a different approach.

# Check if test_urls.csv exists
urls_file = 'processed_data/test_urls.csv'
urls_file_alt = 'processed_data/test_urls.csv'

try:
    # Try to load URLs
    urls_df = pd.read_csv('processed_data/test_urls.csv')
    urls = urls_df['url'].tolist()
    print(f"Loaded {len(urls)} URLs from test_urls.csv")
except:
    # If not, create a list from the test data
    print("test_urls.csv not found. Using sample_df with features...")
    
    # Since we can't get raw URLs, we'll use the features directly
    # We'll test by creating a custom function
    
    urls = []
    for idx, row in sample_df.iterrows():
        # Create a URL based on features (for display only)
        url = f"https://test-url-{idx}.com"
        urls.append(url)

# If we have URLs, test them
if urls:
    print(f"\nTesting {len(urls)} URLs...")
    print("-"*70)
    
    # Use the same sample indices for URLs
    # We'll use the features from sample_df directly
    
    # For each sample, we need to:
    # 1. Get the actual label
    # 2. Extract features
    # 3. Predict using the features
    
    # Since the detector expects a URL, we'll use the features to predict
    # Create a modified test that uses features directly
    
    # Reset results
    correct = 0
    phishing_correct = 0
    legit_correct = 0
    phishing_total = 0
    legit_total = 0
    
    for idx, row in sample_df.iterrows():
        actual_label = row['label']
        
        # Get features as dict
        features_dict = row[feature_cols].to_dict()
        
        # Create a DataFrame from features
        features_df = pd.DataFrame([features_dict])
        
        # Scale features
        features_scaled = detector.scaler.transform(features_df)
        features_scaled = pd.DataFrame(features_scaled, columns=feature_cols)
        
        # Predict using Random Forest directly
        prob = detector.rf_model.predict_proba(features_scaled)[0][1]
        pred = 1 if prob >= 0.65 else 0  # Using new threshold 0.65
        verdict = 'phishing' if pred == 1 else 'legitimate'
        
        # Determine if correct
        is_correct = (verdict == 'phishing' and actual_label == 1) or \
                     (verdict == 'legitimate' and actual_label == 0)
        
        if is_correct:
            correct += 1
        
        if actual_label == 1:
            phishing_total += 1
            if verdict == 'phishing':
                phishing_correct += 1
        else:
            legit_total += 1
            if verdict == 'legitimate':
                legit_correct += 1

print("\n" + "="*70)
print("📊 RESULTS")
print("="*70)

# Calculate metrics
accuracy = correct / len(sample_df) * 100 if len(sample_df) > 0 else 0
phishing_accuracy = phishing_correct / phishing_total * 100 if phishing_total > 0 else 0
legit_accuracy = legit_correct / legit_total * 100 if legit_total > 0 else 0

print(f"\nTotal URLs Tested: {len(sample_df)}")
print(f"Correct Predictions: {correct}/{len(sample_df)} ({accuracy:.1f}%)")

print(f"\n📊 Detailed Breakdown:")
print(f"  Phishing URLs: {phishing_total}")
print(f"    ✅ Correctly Detected: {phishing_correct}/{phishing_total} ({phishing_accuracy:.1f}%)")
print(f"    ❌ Missed: {phishing_total - phishing_correct}")

print(f"\n  Legitimate URLs: {legit_total}")
print(f"    ✅ Correctly Identified: {legit_correct}/{legit_total} ({legit_accuracy:.1f}%)")
print(f"    ❌ False Positives: {legit_total - legit_correct}")

# Compare with previous threshold
print("\n" + "="*70)
print("📈 COMPARISON WITH PREVIOUS THRESHOLD (0.70)")
print("="*70)

print(f"""
| Metric           | Threshold 0.70 | Threshold 0.65 | Change |
|------------------|----------------|----------------|--------|
| Accuracy         | ~72.8%         | {accuracy:.1f}%        | +{accuracy - 72.8:.1f}% |
| Phishing Recall  | ~59.2%         | {phishing_accuracy:.1f}%        | +{phishing_accuracy - 59.2:.1f}% |
| Legitimate Recall| ~78.3%         | {legit_accuracy:.1f}%        | +{legit_accuracy - 78.3:.1f}% |
""")

print("\n✅ Validation Complete!")