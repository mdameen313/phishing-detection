# test_full_hybrid_groq_fixed.py
"""
Hybrid Test with GROQ + REAL URLs - BALANCED
"""

import pandas as pd
import numpy as np
import joblib
import re
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
import tldextract
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# FIXED: Use correct model name
GROQ_MODEL = "openai/gpt-oss-120b"  # ✅ Working model

TEST_SIZE = 500
PHISHING_THRESHOLD = 0.65
SUSPICIOUS_THRESHOLD = 0.45

OUTPUT_FOLDER = 'validation_graphs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("="*70)
print("HYBRID TEST - GROQ + REAL BALANCED URLs")
print("="*70)
print(f"Model:              {GROQ_MODEL}")
print(f"Test Size:          {TEST_SIZE} URLs")
print(f"Phishing Threshold: ≥ {PHISHING_THRESHOLD}")
print(f"Suspicious Range:   {SUSPICIOUS_THRESHOLD} – 0.64")
print("="*70)

# ============================================================
# STEP 1: LOAD URLS
# ============================================================
print("\n📂 Loading URLs from original datasets...")

def find_file(filenames):
    for name in filenames:
        for path in [name, f'../{name}', f'processed_data/{name}', f'../../{name}']:
            if os.path.exists(path):
                return path
    return None

mal_path = find_file(['malicious_urls.csv'])
phish_path = find_file(['phishing_site_urls.csv'])

if not mal_path or not phish_path:
    print("❌ Dataset files not found!")
    exit()

malicious_df = pd.read_csv(mal_path)
phishing_df = pd.read_csv(phish_path)
print(f"✅ malicious_urls.csv: {len(malicious_df)} rows")
print(f"✅ phishing_site_urls.csv: {len(phishing_df)} rows")

# ============================================================
# STEP 2: STANDARDIZE LABELS
# ============================================================
def standardize_labels(df, url_col, label_col, phishing_labels, legit_labels):
    df = df.copy()
    df['label_std'] = df[label_col].astype(str).str.lower().str.strip().map(
        lambda x: 1 if x in phishing_labels else (0 if x in legit_labels else None)
    )
    df = df.dropna(subset=['label_std'])
    df = df.rename(columns={url_col: 'url'})
    df['label'] = df['label_std'].astype(int)
    return df[['url', 'label']]

mal_url_col = 'url' if 'url' in malicious_df.columns else malicious_df.columns[0]
mal_lbl_col = 'label' if 'label' in malicious_df.columns else malicious_df.columns[1]
phish_url_col = 'url' if 'url' in phishing_df.columns else phishing_df.columns[0]
phish_lbl_col = 'Label' if 'Label' in phishing_df.columns else (
                'label' if 'label' in phishing_df.columns else phishing_df.columns[1])

malicious_df = standardize_labels(
    malicious_df, mal_url_col, mal_lbl_col,
    phishing_labels=['phishing', 'malware', 'defacement', 'spam', 'bad'],
    legit_labels=['benign', 'good']
)
phishing_df = standardize_labels(
    phishing_df, phish_url_col, phish_lbl_col,
    phishing_labels=['bad', 'phishing'],
    legit_labels=['good', 'benign']
)

print(f"\n  malicious: {len(malicious_df)} rows")
print(f"    Phishing: {sum(malicious_df['label']==1)}")
print(f"    Benign: {sum(malicious_df['label']==0)}")
print(f"\n  phishing_site_urls: {len(phishing_df)} rows")
print(f"    Phishing: {sum(phishing_df['label']==1)}")
print(f"    Legitimate: {sum(phishing_df['label']==0)}")

# ============================================================
# STEP 3: CLEAN URLs
# ============================================================
def clean_url(url):
    url = str(url).strip().lower()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    url = url.rstrip('/')
    return url

malicious_df['url_clean'] = malicious_df['url'].apply(clean_url)
phishing_df['url_clean'] = phishing_df['url'].apply(clean_url)

malicious_df = malicious_df.drop_duplicates(subset=['url_clean'])
phishing_df = phishing_df.drop_duplicates(subset=['url_clean'])

# ============================================================
# STEP 4: FIXED BALANCED SAMPLING (50/50)
# ============================================================
print("\n" + "="*70)
print("BALANCED SAMPLING (250 Phishing + 250 Legitimate)")
print("="*70)

np.random.seed(42)
HALF_SIZE = TEST_SIZE // 2  # 250 each

# Collect ALL phishing URLs from both datasets
all_phishing = pd.concat([
    malicious_df[malicious_df['label'] == 1][['url_clean']].assign(label=1),
    phishing_df[phishing_df['label'] == 1][['url_clean']].assign(label=1)
]).drop_duplicates(subset=['url_clean'])

# Collect ALL legitimate URLs from both datasets
all_legit = pd.concat([
    malicious_df[malicious_df['label'] == 0][['url_clean']].assign(label=0),
    phishing_df[phishing_df['label'] == 0][['url_clean']].assign(label=0)
]).drop_duplicates(subset=['url_clean'])

print(f"\n  Total phishing URLs available: {len(all_phishing)}")
print(f"  Total legitimate URLs available: {len(all_legit)}")

# Sample exactly HALF_SIZE from each
phish_sample = all_phishing.sample(n=HALF_SIZE, random_state=42)
legit_sample = all_legit.sample(n=HALF_SIZE, random_state=42)

# Combine and shuffle
test_urls_df = pd.concat([phish_sample, legit_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n✅ Balanced test set: {len(test_urls_df)} URLs")
print(f"  Phishing: {sum(test_urls_df['label']==1)}")
print(f"  Legitimate: {sum(test_urls_df['label']==0)}")

print(f"\n📌 Sample URLs:")
for i in range(8):
    row = test_urls_df.iloc[i]
    label = "PHISHING" if row['label'] == 1 else "LEGIT"
    print(f"  [{label:9}] {row['url_clean'][:60]}")

# ============================================================
# STEP 5: EXTRACT FEATURES
# ============================================================
print("\n🔧 Extracting features...")

def extract_features(url):
    url = str(url)
    has_https = 1 if url.startswith('https://') else 0
    
    clean = re.sub(r'^https?://', '', url)
    clean = re.sub(r'^www\.', '', clean)
    clean = clean.rstrip('/')
    
    try:
        parsed = urlparse(url if '://' in url else f'http://{url}')
        domain_info = tldextract.extract(clean)
    except:
        domain_info = tldextract.extract('')
        parsed = urlparse('')
    
    sensitive_words = ['login', 'verify', 'secure', 'update', 'confirm',
                       'password', 'credit', 'signin', 'account', 'validate',
                       'authenticate', 'verification', 'security', 'captcha']
    
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top',
                       '.click', '.link', '.download', '.review', '.stream',
                       '.loan', '.work', '.men', '.party', '.science',
                       '.racing', '.bid', '.win', '.faith', '.accountant',
                       '.date', '.trade', '.webcam', '.zip', '.mov',
                       '.rest', '.sbs', '.online', '.live', '.site',
                       '.tech', '.store', '.shop', '.space', '.pro']
    
    shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 'shorturl', 'is.gd',
                  'ow.ly', 'buff.ly', 'rebrand.ly', 'tiny.cc', 't.co', 'lnkd.in']
    
    features = {
        'url_length': len(clean),
        'domain_length': len(domain_info.domain) if domain_info.domain else 0,
        'subdomain_length': len(domain_info.subdomain) if domain_info.subdomain else 0,
        'path_length': len(parsed.path),
        'num_dots': clean.count('.'),
        'num_hyphens': clean.count('-'),
        'num_underscores': clean.count('_'),
        'num_slashes': clean.count('/'),
        'num_digits': sum(c.isdigit() for c in clean),
        'num_special_chars': sum(not c.isalnum() and c not in ['/', '.', ':', '-', '_'] for c in clean),
        'num_sensitive_words': sum(1 for w in sensitive_words if w in clean.lower()),
        'has_@': 1 if '@' in clean else 0,
        'has_https': has_https,
        'has_ip': 1 if re.search(r'\d+\.\d+\.\d+\.\d+', clean) else 0,
        'has_suspicious_tld': 1 if any(clean.endswith(t) for t in suspicious_tlds) else 0,
        'is_shortened': 1 if any(s in clean for s in shorteners) else 0,
    }
    return features

features_list = [extract_features(url) for url in test_urls_df['url_clean']]
X_test = pd.DataFrame(features_list)
y_test = test_urls_df['label'].values
urls_list = test_urls_df['url_clean'].tolist()

print(f"✅ Extracted: {X_test.shape}")

# ============================================================
# STEP 6: LOAD MODEL
# ============================================================
print("\n📦 Loading model...")
rf = joblib.load('models/random_forest_final_v3.pkl')
scaler = joblib.load('models/scaler.pkl')

feature_cols = ['url_length', 'domain_length', 'subdomain_length', 'path_length',
                'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
                'num_digits', 'num_special_chars', 'num_sensitive_words',
                'has_@', 'has_https', 'has_ip', 'has_suspicious_tld', 'is_shortened']

X_test = X_test[feature_cols]
X_test_scaled = scaler.transform(X_test)
y_proba = rf.predict_proba(X_test_scaled)[:, 1]
print("✅ Predictions ready")

# ============================================================
# STEP 7: CBA RULES
# ============================================================
def apply_cba_rules(url):
    url_lower = url.lower()
    triggered = []
    
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.click']
    if any(url_lower.endswith(tld) for tld in suspicious_tlds):
        triggered.append(('Suspicious TLD', 0.85))
    
    if re.search(r'\d+\.\d+\.\d+\.\d+', url):
        triggered.append(('IP Address', 0.95))
    
    if '@' in url:
        triggered.append(('@ Symbol', 0.90))
    
    shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 'shorturl', 'is.gd', 'ow.ly']
    if any(s in url_lower for s in shorteners):
        triggered.append(('Shortened URL', 0.70))
    
    sensitive = ['login', 'verify', 'secure', 'update', 'confirm', 'signin']
    count = sum(1 for w in sensitive if w in url_lower)
    if count >= 2:
        triggered.append(('Multiple Sensitive Words', 0.75))
    
    if triggered:
        conf = sum(c for _, c in triggered) / len(triggered)
        return True, min(conf, 0.95)
    return False, 0

# ============================================================
# STEP 8: GROQ LLM FUNCTION
# ============================================================
def call_groq_llm(url, prob):
    prompt = f"""You are a cybersecurity expert analyzing a URL for phishing.

URL: {url}

ML confidence: {prob:.1%} (SUSPICIOUS range 0.45-0.64)
The ML model is uncertain about this URL.

Analyze and respond:
1. Is this PHISHING? (YES/NO)
2. Confidence (0-100%)
3. Brief explanation

Look for:
- Suspicious TLDs (.xyz, .tk, .ml, .top)
- Brand impersonation
- Random domains
- Sensitive keywords

Format exactly:
Verdict: YES or NO
Confidence: XX%
Explanation: [brief]"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,  # ✅ Fixed model
            temperature=0.1,
            max_tokens=150,
        )
        
        text = response.choices[0].message.content
        
        verdict = 'phishing' if 'Verdict: YES' in text or 'Verdict:YES' in text else 'legitimate'
        conf_match = re.search(r'Confidence:\s*(\d+)%', text)
        confidence = int(conf_match.group(1)) / 100 if conf_match else 0.5
        
        return verdict, confidence
    except Exception as e:
        print(f"      ⚠️ Groq error: {str(e)[:80]}")
        return 'legitimate', 0.5

# ============================================================
# STEP 9: RUN CASCADE
# ============================================================
print("\n" + "="*70)
print("🔄 RUNNING CASCADE")
print("="*70)

final_predictions = []
cascade_paths = []
suspicious_indices = []
llm_calls = 0

# Layer 1: CBA
cba_indices = set()
for i in range(len(X_test)):
    is_phishing, conf = apply_cba_rules(urls_list[i])
    if is_phishing and conf >= 0.70:
        cba_indices.add(i)
        final_predictions.append(1)
        cascade_paths.append('CBA')

print(f"✅ CBA decided: {len(cba_indices)}")

# Layer 2: RF
for i in range(len(X_test)):
    if i in cba_indices:
        continue
    prob = y_proba[i]
    if prob >= PHISHING_THRESHOLD:
        final_predictions.append(1)
        cascade_paths.append('RF-Phishing')
    elif prob >= SUSPICIOUS_THRESHOLD:
        suspicious_indices.append(i)
        cascade_paths.append('LLM-Pending')
    else:
        final_predictions.append(0)
        cascade_paths.append('RF-Legitimate')

print(f"✅ RF-Phishing: {cascade_paths.count('RF-Phishing')}")
print(f"✅ RF-Legitimate: {cascade_paths.count('RF-Legitimate')}")
print(f"⚠️ LLM calls: {len(suspicious_indices)}")

# Layer 3: Groq LLM
print(f"\n🧠 Calling Groq LLM ({GROQ_MODEL})")
print(f"   ⚡ Fast model - ~1-2 sec per call")
print(f"   ⏱️  Estimated: {len(suspicious_indices)*2/60:.1f} minutes\n")

for count, idx in enumerate(suspicious_indices, 1):
    url = urls_list[idx]
    prob = y_proba[idx]
    
    print(f"   [{count}/{len(suspicious_indices)}] {url[:55]}...")
    
    verdict, confidence = call_groq_llm(url, prob)
    final_predictions.append(1 if verdict == 'phishing' else 0)
    cascade_paths[cascade_paths.index('LLM-Pending')] = f'LLM-{verdict.capitalize()}'
    llm_calls += 1
    
    print(f"      → {verdict.upper()} ({confidence:.0%})")
    
    if count < len(suspicious_indices):
        time.sleep(2)

final_predictions = np.array(final_predictions)

# ============================================================
# STEP 10: RESULTS
# ============================================================
cm = confusion_matrix(y_test, final_predictions)
accuracy = accuracy_score(y_test, final_predictions)
precision = precision_score(y_test, final_predictions, zero_division=0)
recall = recall_score(y_test, final_predictions, zero_division=0)
f1 = f1_score(y_test, final_predictions, zero_division=0)

y_pred_rf = (y_proba >= PHISHING_THRESHOLD).astype(int)
cm_rf = confusion_matrix(y_test, y_pred_rf)
acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, zero_division=0)
rec_rf = recall_score(y_test, y_pred_rf, zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)

print("\n" + "="*70)
print("📊 FINAL RESULTS")
print("="*70)
print(f"\n{'Metric':<15} {'RF Only':<12} {'Full Cascade':<15} {'Change':<10}")
print("-"*55)
print(f"{'Accuracy':<15} {acc_rf*100:.2f}%{'':<7} {accuracy*100:.2f}%{'':<10} {(accuracy-acc_rf)*100:+.2f}%")
print(f"{'Precision':<15} {prec_rf*100:.2f}%{'':<7} {precision*100:.2f}%{'':<10} {(precision-prec_rf)*100:+.2f}%")
print(f"{'Recall':<15} {rec_rf*100:.2f}%{'':<7} {recall*100:.2f}%{'':<10} {(recall-rec_rf)*100:+.2f}%")
print(f"{'F1 Score':<15} {f1_rf*100:.2f}%{'':<7} {f1*100:.2f}%{'':<10} {(f1-f1_rf)*100:+.2f}%")

print(f"\n📊 Confusion Matrix (Full Cascade):")
print(f"   TN: {cm[0,0]:3d}  |  FP: {cm[0,1]:3d}")
print(f"   FN: {cm[1,0]:3d}  |  TP: {cm[1,1]:3d}")

print(f"\n📊 Cascade Distribution:")
print(f"   CBA:            {cascade_paths.count('CBA')}")
print(f"   RF-Phishing:    {cascade_paths.count('RF-Phishing')}")
print(f"   RF-Legitimate:  {cascade_paths.count('RF-Legitimate')}")
print(f"   LLM decided:    {llm_calls}")

# ============================================================
# STEP 11: GRAPHS
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'],
            cbar=False, annot_kws={'size': 16, 'weight': 'bold'},
            linewidths=2, linecolor='white', ax=axes[0])
axes[0].set_title(f'A. Random Forest Only\nAcc: {acc_rf*100:.2f}%  |  F1: {f1_rf*100:.2f}%',
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('Predicted', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Actual', fontsize=11, fontweight='bold')

sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'],
            cbar=False, annot_kws={'size': 16, 'weight': 'bold'},
            linewidths=2, linecolor='white', ax=axes[1])
axes[1].set_title(f'B. Full Cascade (CBA + RF + Groq)\nAcc: {accuracy*100:.2f}%  |  F1: {f1*100:.2f}%',
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('Predicted', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Actual', fontsize=11, fontweight='bold')

plt.suptitle('Confusion Matrix: RF-Only vs Full Cascade',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_FOLDER}/confusion_matrix_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"\n✅ Saved: {OUTPUT_FOLDER}/confusion_matrix_comparison.png")

# Cascade distribution graph
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cba_count = cascade_paths.count('CBA')
rf_phish = cascade_paths.count('RF-Phishing')
rf_legit = cascade_paths.count('RF-Legitimate')
llm_phish = cascade_paths.count('LLM-Phishing')
llm_legit = cascade_paths.count('LLM-Legitimate')

labels = []
sizes = []
colors = []
if cba_count > 0:
    labels.append('CBA'); sizes.append(cba_count); colors.append('#4caf50')
if rf_phish > 0:
    labels.append('RF-Phishing'); sizes.append(rf_phish); colors.append('#d32f2f')
if rf_legit > 0:
    labels.append('RF-Legitimate'); sizes.append(rf_legit); colors.append('#2e7d32')
if llm_phish > 0:
    labels.append('LLM-Phishing'); sizes.append(llm_phish); colors.append('#ff9800')
if llm_legit > 0:
    labels.append('LLM-Legitimate'); sizes.append(llm_legit); colors.append('#f57c00')

axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90,
            textprops={'fontsize': 10, 'weight': 'bold'})
axes[0].set_title('Cascade Decision Distribution', fontsize=13, fontweight='bold')

layers = ['CBA\n(Rules)', 'Random Forest', 'Groq LLM\n(Fallback)']
counts = [cba_count, rf_phish + rf_legit, llm_calls]
bars = axes[1].bar(layers, counts, color=['#4caf50', '#1a73e8', '#ff9800'],
                   edgecolor='black', linewidth=1.5)
axes[1].set_ylabel('Number of URLs', fontsize=12, fontweight='bold')
axes[1].set_title('URLs Handled by Each Layer', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, counts):
    pct = val / len(X_test) * 100
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{val}\n({pct:.1f}%)', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_FOLDER}/cascade_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {OUTPUT_FOLDER}/cascade_distribution.png")

# Save results
results_df = pd.DataFrame({
    'URL': urls_list,
    'Actual': y_test,
    'RF_Probability': y_proba,
    'Cascade_Path': cascade_paths,
    'Final_Verdict': final_predictions,
    'Correct': (final_predictions == y_test).astype(int)
})
results_df.to_csv(f'{OUTPUT_FOLDER}/full_cascade_real_results.csv', index=False)

print(f"✅ Saved: {OUTPUT_FOLDER}/full_cascade_real_results.csv")
print("\n" + "="*70)
print("✅ COMPLETE")
print("="*70)