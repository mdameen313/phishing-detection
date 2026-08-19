"""
Complete Data Preprocessing Pipeline - FINAL VERSION
Creates train/val/test splits from balanced data
"""

import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from urllib.parse import urlparse
import tldextract
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

# ===================== CONFIGURATION =====================
CONFIG = {
    'dataset_2_path': 'malicious_urls.csv',
    'dataset_3_path': 'phishing_site_urls.csv',
    'total_samples': 30000,              # Total balanced samples
    'train_ratio': 0.7,                  # 70% training
    'val_ratio': 0.15,                   # 15% validation
    'test_ratio': 0.15,                  # 15% test
    'random_state': 42,
    'output_dir': './processed_data/'
}

# ===================== DATA LOADING =====================
def load_datasets():
    print("\n=== LOADING DATASETS ===")
    
    df_malicious = pd.read_csv(CONFIG['dataset_2_path'])
    print(f"Dataset #2: {len(df_malicious):,} URLs")
    print(f"  - Columns: {df_malicious.columns.tolist()}")
    
    label_col_2 = None
    for col in df_malicious.columns:
        if col.lower() in ['type', 'label', 'category', 'class']:
            label_col_2 = col
            break
    
    if label_col_2:
        print(f"  - Label column: '{label_col_2}'")
        print(f"  - Types: {df_malicious[label_col_2].value_counts().to_dict()}")
    
    df_phishing = pd.read_csv(CONFIG['dataset_3_path'])
    print(f"\nDataset #3: {len(df_phishing):,} URLs")
    print(f"  - Columns: {df_phishing.columns.tolist()}")
    
    label_col_3 = None
    for col in df_phishing.columns:
        if col.lower() in ['type', 'label', 'category', 'class', 'status']:
            label_col_3 = col
            break
    
    if label_col_3:
        print(f"  - Label column: '{label_col_3}'")
        print(f"  - Types: {df_phishing[label_col_3].value_counts().to_dict()}")
    
    return df_malicious, df_phishing, label_col_2, label_col_3

# ===================== LABEL STANDARDIZATION =====================
def standardize_labels(df_malicious, df_phishing, label_col_2, label_col_3):
    print("\n=== STANDARDIZING LABELS ===")
    
    if label_col_2:
        label_mapping_2 = {}
        for label in df_malicious[label_col_2].unique():
            label_str = str(label).lower().strip()
            if label_str in ['benign', 'good', 'safe', 'legitimate', '0']:
                label_mapping_2[label] = 0
            elif label_str in ['phishing', 'bad', 'malicious', 'defacement', 'malware', 'spam', '1']:
                label_mapping_2[label] = 1
            else:
                label_mapping_2[label] = np.nan
        
        df_malicious['label'] = df_malicious[label_col_2].map(label_mapping_2)
    
    if label_col_3:
        label_mapping_3 = {}
        for label in df_phishing[label_col_3].unique():
            label_str = str(label).lower().strip()
            if label_str in ['benign', 'good', 'safe', 'legitimate', '0']:
                label_mapping_3[label] = 0
            elif label_str in ['phishing', 'bad', 'malicious', 'defacement', 'malware', 'spam', '1']:
                label_mapping_3[label] = 1
            else:
                label_mapping_3[label] = np.nan
        
        df_phishing['label'] = df_phishing[label_col_3].map(label_mapping_3)
    
    df_malicious = df_malicious.dropna(subset=['label'])
    df_phishing = df_phishing.dropna(subset=['label'])
    df_malicious['label'] = df_malicious['label'].astype(int)
    df_phishing['label'] = df_phishing['label'].astype(int)
    
    print(f"Dataset #2: {len(df_malicious):,} URLs (Phishing: {sum(df_malicious['label']==1):,}, Legit: {sum(df_malicious['label']==0):,})")
    print(f"Dataset #3: {len(df_phishing):,} URLs (Phishing: {sum(df_phishing['label']==1):,}, Legit: {sum(df_phishing['label']==0):,})")
    
    return df_malicious, df_phishing

# ===================== URL CLEANING =====================
def clean_url(url):
    url = str(url).strip().lower()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    url = url.rstrip('/')
    return url

def clean_and_deduplicate(df_malicious, df_phishing):
    print("\n=== CLEANING URLS ===")
    
    url_col_2 = 'url' if 'url' in df_malicious.columns else df_malicious.columns[0]
    url_col_3 = 'URL' if 'URL' in df_phishing.columns else df_phishing.columns[0]
    
    df_malicious['url_clean'] = df_malicious[url_col_2].apply(clean_url)
    df_phishing['url_clean'] = df_phishing[url_col_3].apply(clean_url)
    
    before_malicious = len(df_malicious)
    before_phishing = len(df_phishing)
    df_malicious = df_malicious.drop_duplicates(subset=['url_clean'])
    df_phishing = df_phishing.drop_duplicates(subset=['url_clean'])
    
    print(f"Dataset #2: {before_malicious:,} → {len(df_malicious):,} (removed {before_malicious - len(df_malicious):,} duplicates)")
    print(f"Dataset #3: {before_phishing:,} → {len(df_phishing):,} (removed {before_phishing - len(df_phishing):,} duplicates)")
    
    urls_ds2 = set(df_malicious['url_clean'])
    before_remove = len(df_phishing)
    df_phishing = df_phishing[~df_phishing['url_clean'].isin(urls_ds2)]
    print(f"Removed {before_remove - len(df_phishing):,} URLs from Dataset #3 that overlap with Dataset #2")
    
    return df_malicious, df_phishing

# ===================== DATA BALANCING =====================
def balance_dataset(df, target_samples=15000, random_state=42):
    print(f"\n=== BALANCING DATASET ===")
    
    df_phishing = df[df['label'] == 1]
    df_legit = df[df['label'] == 0]
    
    print(f"Before balancing:")
    print(f"  Phishing: {len(df_phishing):,}")
    print(f"  Legitimate: {len(df_legit):,}")
    
    # Balance both classes
    if len(df_phishing) > target_samples:
        df_phishing = resample(df_phishing, replace=False, n_samples=target_samples, random_state=random_state)
    if len(df_legit) > target_samples:
        df_legit = resample(df_legit, replace=False, n_samples=target_samples, random_state=random_state)
    
    df_balanced = pd.concat([df_phishing, df_legit]).sample(frac=1, random_state=random_state)
    
    print(f"After balancing:")
    print(f"  Phishing: {len(df_balanced[df_balanced['label']==1]):,}")
    print(f"  Legitimate: {len(df_balanced[df_balanced['label']==0]):,}")
    
    return df_balanced

# ===================== FEATURE EXTRACTION =====================
def extract_features(url):
    url = str(url)
    
    try:
        parsed = urlparse(url if '://' in url else f'http://{url}')
        domain_info = tldextract.extract(url)
    except:
        return {k: 0 for k in ['url_length', 'domain_length', 'subdomain_length', 'path_length',
                               'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
                               'num_digits', 'num_special_chars', 'num_sensitive_words',
                               'has_@', 'has_https', 'has_ip', 'has_suspicious_tld', 'is_shortened']}
    
    num_dots = url.count('.')
    num_hyphens = url.count('-')
    num_underscores = url.count('_')
    num_slashes = url.count('/')
    num_digits = sum(c.isdigit() for c in url)
    num_special = sum(not c.isalnum() and c not in ['/', '.', ':', '-', '_'] for c in url)
    
    sensitive_words = ['login', 'verify', 'secure', 'update', 'bank', 'signin', 
                      'account', 'confirm', 'password', 'credit', 'paypal', 
                      'amazon', 'netflix', 'apple', 'google', 'microsoft']
    num_sensitive = sum(1 for word in sensitive_words if word in url.lower())
    
    features = {
        'url_length': len(url),
        'domain_length': len(domain_info.domain) if domain_info.domain else 0,
        'subdomain_length': len(domain_info.subdomain) if domain_info.subdomain else 0,
        'path_length': len(parsed.path),
        'num_dots': num_dots,
        'num_hyphens': num_hyphens,
        'num_underscores': num_underscores,
        'num_slashes': num_slashes,
        'num_digits': num_digits,
        'num_special_chars': num_special,
        'num_sensitive_words': num_sensitive,
        'has_@': 1 if '@' in url else 0,
        'has_https': 1 if parsed.scheme == 'https' else 0,
        'has_ip': 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0,
        'has_suspicious_tld': 1 if domain_info.suffix in ['tk', 'ml', 'ga', 'cf', 'xyz', 'top', 'club', 'work', 'info', 'biz', 'click'] else 0,
        'is_shortened': 1 if any(s in url for s in ['bit.ly', 'tinyurl', 'goo.gl', 'shorturl', 'is.gd', 'ow.ly']) else 0,
    }
    return features

def extract_features_batch(urls, desc="Extracting features"):
    features_list = []
    
    for url in tqdm(urls, desc=desc):
        try:
            features_list.append(extract_features(url))
        except Exception as e:
            features_list.append({k: 0 for k in extract_features('').keys()})
    
    return pd.DataFrame(features_list)

# ===================== MAIN PIPELINE =====================
def main():
    print("="*60)
    print("PHISHING DATASET PREPROCESSING PIPELINE (FINAL VERSION)")
    print("="*60)
    
    try:
        # 1. Load datasets
        print("\n[STEP 1] Loading datasets...")
        df_malicious, df_phishing, label_col_2, label_col_3 = load_datasets()
        print("✅ Step 1 complete")
        
        # 2. Standardize labels
        print("\n[STEP 2] Standardizing labels...")
        df_malicious, df_phishing = standardize_labels(df_malicious, df_phishing, label_col_2, label_col_3)
        print("✅ Step 2 complete")
        
        # 3. Clean and deduplicate
        print("\n[STEP 3] Cleaning URLs...")
        df_malicious, df_phishing = clean_and_deduplicate(df_malicious, df_phishing)
        print("✅ Step 3 complete")
        
        # 4. Combine datasets (use all cleaned data)
        print("\n[STEP 4] Combining datasets...")
        # Take balanced samples from Dataset #2
        df_combined = balance_dataset(df_malicious, CONFIG['total_samples'] // 2)
        
        # Add some samples from Dataset #3 if available
        if len(df_phishing) > 0:
            # Take a balanced sample from Dataset #3 too
            df_phishing_balanced = balance_dataset(df_phishing, CONFIG['total_samples'] // 4)
            df_combined = pd.concat([df_combined, df_phishing_balanced])
        
        print(f"Combined dataset: {len(df_combined):,} URLs")
        print(f"  Phishing: {sum(df_combined['label']==1):,}")
        print(f"  Legitimate: {sum(df_combined['label']==0):,}")
        print("✅ Step 4 complete")
        
        # 5. Split into train/val/test
        print("\n[STEP 5] Splitting into train/val/test...")
        
        X = df_combined['url_clean']
        y = df_combined['label']
        
        # First split: train vs (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(CONFIG['val_ratio'] + CONFIG['test_ratio']),
            random_state=CONFIG['random_state'],
            stratify=y
        )
        
        # Second split: val vs test
        val_test_ratio = CONFIG['test_ratio'] / (CONFIG['val_ratio'] + CONFIG['test_ratio'])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=val_test_ratio,
            random_state=CONFIG['random_state'],
            stratify=y_temp
        )
        
        print(f"Training set: {len(X_train):,} URLs")
        print(f"  Phishing: {sum(y_train==1):,}, Legit: {sum(y_train==0):,}")
        print(f"Validation set: {len(X_val):,} URLs")
        print(f"  Phishing: {sum(y_val==1):,}, Legit: {sum(y_val==0):,}")
        print(f"Test set: {len(X_test):,} URLs")
        print(f"  Phishing: {sum(y_test==1):,}, Legit: {sum(y_test==0):,}")
        print("✅ Step 5 complete")
        
        # 6. Extract features
        print("\n[STEP 6] Extracting features...")
        
        X_train_features = extract_features_batch(X_train, "Training set")
        X_val_features = extract_features_batch(X_val, "Validation set")
        X_test_features = extract_features_batch(X_test, "Test set")
        
        # Add labels
        X_train_features['label'] = y_train.values
        X_val_features['label'] = y_val.values
        X_test_features['label'] = y_test.values
        print("✅ Step 6 complete")
        
        # 7. Save processed data
        print("\n[STEP 7] Saving processed data...")
        
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        
        # Save CSV files
        X_train_features.to_csv(f"{CONFIG['output_dir']}train_dataset.csv", index=False)
        X_val_features.to_csv(f"{CONFIG['output_dir']}val_dataset.csv", index=False)
        X_test_features.to_csv(f"{CONFIG['output_dir']}test_dataset.csv", index=False)
        
        # Save pickle files
        joblib.dump(X_train_features, f"{CONFIG['output_dir']}train_features.pkl")
        joblib.dump(X_val_features, f"{CONFIG['output_dir']}val_features.pkl")
        joblib.dump(X_test_features, f"{CONFIG['output_dir']}test_features.pkl")
        
        # Save raw URLs
        pd.Series(X_train).to_csv(f"{CONFIG['output_dir']}train_urls.csv", index=False, header=['url'])
        pd.Series(X_val).to_csv(f"{CONFIG['output_dir']}val_urls.csv", index=False, header=['url'])
        pd.Series(X_test).to_csv(f"{CONFIG['output_dir']}test_urls.csv", index=False, header=['url'])
        
        # Save summary
        summary = {
            'Training set': {'URLs': len(X_train), 'Phishing': sum(y_train==1), 'Legit': sum(y_train==0)},
            'Validation set': {'URLs': len(X_val), 'Phishing': sum(y_val==1), 'Legit': sum(y_val==0)},
            'Test set': {'URLs': len(X_test), 'Phishing': sum(y_test==1), 'Legit': sum(y_test==0)}
        }
        summary_df = pd.DataFrame(summary).T
        summary_df.to_csv(f"{CONFIG['output_dir']}dataset_summary.csv")
        print("✅ Step 7 complete")
        
        # 8. Verify files
        print("\n[STEP 8] Verifying files...")
        files = os.listdir(CONFIG['output_dir'])
        print(f"Files in {CONFIG['output_dir']}:")
        for f in files:
            size = os.path.getsize(f"{CONFIG['output_dir']}{f}")
            print(f"  - {f} ({size:,} bytes)")
        
        print("\n" + "="*60)
        print("✅ PROCESSING COMPLETE!")
        print("="*60)
        print(f"\nFiles saved in: {CONFIG['output_dir']}")
        print("\n📊 Dataset Summary:")
        print(summary_df)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()