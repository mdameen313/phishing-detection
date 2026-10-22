# test_hybrid_with_content_analysis_FIXED.py
"""
Hybrid Test with Content Analysis - FIXED VERSION
Handles fetch failures gracefully
"""

import pandas as pd
import numpy as np
import joblib
import re
import time
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from groq import Groq
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
GROQ_API_KEY = "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"
GROQ_MODEL = "openai/gpt-oss-120b"
client = Groq(api_key=GROQ_API_KEY)

TEST_SIZE = 200
PHISHING_THRESHOLD = 0.65
SUSPICIOUS_THRESHOLD = 0.45
FETCH_TIMEOUT = 3  # Shorter timeout

OUTPUT_FOLDER = 'validation_graphs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OFFICIAL_DOMAINS = [
    'google.com', 'facebook.com', 'amazon.com', 'amazon.in', 'microsoft.com',
    'apple.com', 'paypal.com', 'netflix.com', 'instagram.com', 'twitter.com',
    'linkedin.com', 'youtube.com', 'whatsapp.com', 'binance.com', 'coinbase.com',
    'hdfcbank.com', 'sbi.co.in', 'icicibank.com', 'axisbank.com', 'kotak.com',
]

BRAND_KEYWORDS = [
    'paypal', 'amazon', 'google', 'microsoft', 'apple', 'facebook',
    'instagram', 'netflix', 'flipkart', 'hdfc', 'sbi', 'icici',
    'axis', 'kotak', 'binance', 'coinbase', 'steam',
]

URGENCY_KEYWORDS = [
    'urgent', 'immediate', 'verify now', 'act now', 'limited time',
    'account locked', 'account suspended', 'immediate action',
]

print("="*70)
print("HYBRID TEST - FIXED VERSION")
print("="*70)

# ============================================================
# LOAD URLS
# ============================================================
def find_file(filenames):
    for name in filenames:
        for path in [name, f'../{name}', f'processed_data/{name}', f'../../{name}']:
            if os.path.exists(path):
                return path
    return None

malicious_df = pd.read_csv(find_file(['malicious_urls.csv']))
phishing_df = pd.read_csv(find_file(['phishing_site_urls.csv']))

def standardize(df, url_col, label_col, phishing_labels, legit_labels):
    df = df.copy()
    df['label_std'] = df[label_col].astype(str).str.lower().str.strip().map(
        lambda x: 1 if x in phishing_labels else (0 if x in legit_labels else None)
    )
    df = df.dropna(subset=['label_std']).rename(columns={url_col: 'url'})
    df['label'] = df['label_std'].astype(int)
    return df[['url', 'label']]

malicious_df = standardize(malicious_df, 'url', 'type',
                            ['phishing', 'malware', 'defacement', 'spam'],
                            ['benign', 'good'])
phishing_df = standardize(phishing_df, 'URL', 'Label', ['bad'], ['good'])

def clean_url(url):
    url = str(url).strip().lower()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    return url.rstrip('/')

malicious_df['url_clean'] = malicious_df['url'].apply(clean_url)
phishing_df['url_clean'] = phishing_df['url'].apply(clean_url)
malicious_df = malicious_df.drop_duplicates(subset=['url_clean'])
phishing_df = phishing_df.drop_duplicates(subset=['url_clean'])

HALF = TEST_SIZE // 2
all_phishing = pd.concat([
    malicious_df[malicious_df['label']==1][['url_clean']].assign(label=1),
    phishing_df[phishing_df['label']==1][['url_clean']].assign(label=1)
]).drop_duplicates(subset=['url_clean'])

all_legit = pd.concat([
    malicious_df[malicious_df['label']==0][['url_clean']].assign(label=0),
    phishing_df[phishing_df['label']==0][['url_clean']].assign(label=0)
]).drop_duplicates(subset=['url_clean'])

np.random.seed(42)
test_urls_df = pd.concat([
    all_phishing.sample(n=HALF, random_state=42),
    all_legit.sample(n=HALF, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Test: {len(test_urls_df)} URLs")

# ============================================================
# EXTRACT URL FEATURES
# ============================================================
import tldextract

def extract_url_features(url):
    url = str(url)
    has_https = 1 if url.startswith('https://') else 0
    clean = re.sub(r'^https?://', '', url)
    clean = re.sub(r'^www\.', '', clean).rstrip('/')
    
    try:
        parsed = urlparse(url if '://' in url else f'http://{url}')
        domain_info = tldextract.extract(clean)
    except:
        domain_info = tldextract.extract('')
        parsed = urlparse('')
    
    sensitive = ['login', 'verify', 'secure', 'update', 'confirm', 'password', 'signin']
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top']
    shorteners = ['bit.ly', 'tinyurl', 'goo.gl']
    
    return {
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
        'num_sensitive_words': sum(1 for w in sensitive if w in clean.lower()),
        'has_@': 1 if '@' in clean else 0,
        'has_https': has_https,
        'has_ip': 1 if re.search(r'\d+\.\d+\.\d+\.\d+', clean) else 0,
        'has_suspicious_tld': 1 if any(clean.endswith(t) for t in suspicious_tlds) else 0,
        'is_shortened': 1 if any(s in clean for s in shorteners) else 0,
    }

# ============================================================
# CONTENT ANALYSIS - ROBUST VERSION
# ============================================================
def analyze_page_content(url):
    """
    Fetch and analyze page content with proper error handling
    Returns (score, signals, status)
    """
    signals = []
    score = 0.0
    
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    # ============================================================
    # TRY FETCHING
    # ============================================================
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(
            url, 
            timeout=FETCH_TIMEOUT, 
            headers=headers, 
            allow_redirects=True,
            verify=False  # Skip SSL verification to avoid errors
        )
        html = response.text
        final_url = response.url
    except requests.exceptions.Timeout:
        return -1, ["fetch_timeout"], "timeout"
    except requests.exceptions.ConnectionError:
        return -1, ["connection_failed"], "failed"
    except requests.exceptions.TooManyRedirects:
        return -1, ["too_many_redirects"], "failed"
    except Exception as e:
        return -1, [f"fetch_error: {str(e)[:50]}"], "failed"
    
    # ============================================================
    # PARSE HTML
    # ============================================================
    try:
        soup = BeautifulSoup(html, 'html.parser')
    except:
        return -1, ["parse_failed"], "failed"
    
    # Domain info
    parsed = urlparse(final_url)
    page_domain = parsed.netloc.lower().replace('www.', '')
    is_official = any(page_domain.endswith(od) for od in OFFICIAL_DOMAINS)
    
    if is_official:
        return 0.0, ["official_domain"], "success"
    
    # ============================================================
    # SIGNALS
    # ============================================================
    
    # 1. Title brand mismatch
    title = soup.title.string.lower() if soup.title and soup.title.string else ''
    for brand in BRAND_KEYWORDS:
        if brand in title:
            score += 0.20
            signals.append(f"title_brand:{brand}")
            break
    
    # 2. Password field
    if soup.find_all('input', {'type': 'password'}):
        score += 0.30
        signals.append("password_field")
    
    # 3. Credit card fields
    text_lower = html.lower()
    if 'card_number' in text_lower or 'cvv' in text_lower:
        score += 0.30
        signals.append("credit_card_field")
    
    # 4. Form action external
    for form in soup.find_all('form'):
        action = form.get('action', '')
        if action:
            action_url = urljoin(url, action)
            action_domain = urlparse(action_url).netloc.lower()
            if action_domain and action_domain != page_domain:
                score += 0.40
                signals.append("form_action_external")
                break
    
    # 5. Urgency keywords
    text_content = soup.get_text().lower()
    urgency = sum(1 for kw in URGENCY_KEYWORDS if kw in text_content)
    if urgency >= 2:
        score += 0.15
        signals.append(f"urgency:{urgency}")
    
    # 6. No copyright
    has_copy = '©' in text_content or 'copyright' in text_content
    if not has_copy:
        score += 0.05
        signals.append("no_copyright")
    
    # 7. External favicon
    favicon = soup.find('link', rel=re.compile(r'icon', re.I))
    if favicon and favicon.get('href'):
        fav_url = urljoin(url, favicon['href'])
        fav_domain = urlparse(fav_url).netloc.lower()
        if fav_domain and fav_domain != page_domain:
            score += 0.15
            signals.append("favicon_external")
    
    # 8. JS redirect
    for script in soup.find_all('script', src=False):
        if script.string and ('window.location' in script.string):
            score += 0.20
            signals.append("js_redirect")
            break
    
    # 9. Meta refresh
    if soup.find('meta', attrs={'http-equiv': re.compile(r'refresh', re.I)}):
        score += 0.20
        signals.append("meta_refresh")
    
    # 10. Many iframes
    if len(soup.find_all('iframe')) > 3:
        score += 0.10
        signals.append("many_iframes")
    
    score = min(score, 1.0)
    
    return score, signals, "success"

# ============================================================
# LOAD MODEL
# ============================================================
rf = joblib.load('models/random_forest_final_v3.pkl')
scaler = joblib.load('models/scaler.pkl')

feature_cols = ['url_length', 'domain_length', 'subdomain_length', 'path_length',
                'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
                'num_digits', 'num_special_chars', 'num_sensitive_words',
                'has_@', 'has_https', 'has_ip', 'has_suspicious_tld', 'is_shortened']

url_features = pd.DataFrame([extract_url_features(u) for u in test_urls_df['url_clean']])
url_features = url_features[feature_cols]
X_scaled = scaler.transform(url_features)
y_proba = rf.predict_proba(X_scaled)[:, 1]

y_test = test_urls_df['label'].values
urls_list = test_urls_df['url_clean'].tolist()

# ============================================================
# CBA RULES
# ============================================================
def apply_cba(url):
    triggered = []
    tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top']
    if any(url.endswith(t) for t in tlds):
        triggered.append(0.85)
    if re.search(r'\d+\.\d+\.\d+\.\d+', url):
        triggered.append(0.95)
    if '@' in url:
        triggered.append(0.90)
    return (True, sum(triggered)/len(triggered)) if triggered else (False, 0)

# ============================================================
# LLM FUNCTION
# ============================================================
def call_groq_llm(url, prob, signals):
    signals_text = ", ".join(signals[:5]) if signals else "none"
    prompt = f"""Analyze this URL for phishing:
URL: {url}
ML confidence: {prob:.1%}
Content signals: {signals_text}

Format:
Verdict: YES or NO
Confidence: XX%"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=100,
        )
        text = response.choices[0].message.content
        verdict = 'phishing' if 'Verdict: YES' in text else 'legitimate'
        conf_match = re.search(r'Confidence:\s*(\d+)%', text)
        confidence = int(conf_match.group(1)) / 100 if conf_match else 0.5
        return verdict, confidence
    except Exception as e:
        return 'legitimate', 0.5

# ============================================================
# RUN CASCADE
# ============================================================
print("\n" + "="*70)
print("🔄 RUNNING CASCADE")
print("="*70)

final_preds = []
paths = []
suspicious_indices = []
content_success = 0
content_failed = 0
llm_count = 0

# Layer 1: CBA
cba_indices = set()
for i in range(len(urls_list)):
    is_phish, conf = apply_cba(urls_list[i])
    if is_phish and conf >= 0.70:
        cba_indices.add(i)
        final_preds.append(1)
        paths.append('CBA')

print(f"✅ CBA: {len(cba_indices)}")

# Layer 2: RF
for i in range(len(urls_list)):
    if i in cba_indices:
        continue
    prob = y_proba[i]
    if prob >= PHISHING_THRESHOLD:
        final_preds.append(1)
        paths.append('RF-Phishing')
    elif prob >= SUSPICIOUS_THRESHOLD:
        suspicious_indices.append(i)
        paths.append('SUSPICIOUS')
    else:
        final_preds.append(0)
        paths.append('RF-Legitimate')

print(f"✅ RF-Phishing: {paths.count('RF-Phishing')}")
print(f"✅ RF-Legitimate: {paths.count('RF-Legitimate')}")
print(f"⚠️ Suspicious: {len(suspicious_indices)}")

# Layer 3: Content Analysis
print(f"\n🌐 Layer 3: Content analysis for {len(suspicious_indices)} URLs")
print(f"   Timeout: {FETCH_TIMEOUT}s per URL")
print(f"   Estimated: {len(suspicious_indices) * 2 / 60:.1f} minutes")

content_uncertain = []

for count, idx in enumerate(suspicious_indices, 1):
    url = urls_list[idx]
    
    try:
        score, signals, status = analyze_page_content(url)
        
        if status == 'success':
            content_success += 1
            print(f"   [{count}/{len(suspicious_indices)}] ✅ {url[:45]} | Score: {score:.2f}")
            
            # Decision
            if score >= 0.65:
                final_preds.append(1)
                paths[paths.index('SUSPICIOUS')] = 'Content-Phishing'
            elif score < 0.20:
                final_preds.append(0)
                paths[paths.index('SUSPICIOUS')] = 'Content-Legitimate'
            else:
                content_uncertain.append((idx, score, signals))
                paths[paths.index('SUSPICIOUS')] = 'Content-Uncertain'
        else:
            content_failed += 1
            print(f"   [{count}/{len(suspicious_indices)}] ❌ {url[:45]} | Failed: {status}")
            # Fetch failed - fall back to RF decision
            content_uncertain.append((idx, y_proba[idx], [f"fetch_{status}"]))
            paths[paths.index('SUSPICIOUS')] = 'Content-Failed'
    
    except KeyboardInterrupt:
        print("\n⚠️ User interrupted")
        break
    except Exception as e:
        content_failed += 1
        print(f"   [{count}/{len(suspicious_indices)}] ❌ {url[:45]} | Error: {str(e)[:30]}")
        content_uncertain.append((idx, y_proba[idx], ["error"]))
        paths[paths.index('SUSPICIOUS')] = 'Content-Failed'

print(f"\n✅ Content analysis done:")
print(f"   Success: {content_success}")
print(f"   Failed: {content_failed}")

# Layer 4: LLM for uncertain
if content_uncertain:
    print(f"\n🧠 Layer 4: LLM for {len(content_uncertain)} uncertain cases")
    
    for count, (idx, score, signals) in enumerate(content_uncertain, 1):
        url = urls_list[idx]
        prob = y_proba[idx]
        
        print(f"   [{count}/{len(content_uncertain)}] {url[:50]}...")
        
        verdict, confidence = call_groq_llm(url, prob, signals)
        final_preds.append(1 if verdict == 'phishing' else 0)
        llm_count += 1
        
        print(f"      → {verdict.upper()} ({confidence:.0%})")
        
        if count < len(content_uncertain):
            time.sleep(2)

# Fill any remaining
while len(final_preds) < len(urls_list):
    final_preds.append(0)

final_preds = np.array(final_preds[:len(urls_list)])

# ============================================================
# RESULTS
# ============================================================
cm = confusion_matrix(y_test, final_preds)
accuracy = accuracy_score(y_test, final_preds)
precision = precision_score(y_test, final_preds, zero_division=0)
recall = recall_score(y_test, final_preds, zero_division=0)
f1 = f1_score(y_test, final_preds, zero_division=0)

y_pred_rf = (y_proba >= PHISHING_THRESHOLD).astype(int)
cm_rf = confusion_matrix(y_test, y_pred_rf)
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)

print("\n" + "="*70)
print("📊 RESULTS")
print("="*70)
print(f"\n{'Metric':<15} {'RF Only':<12} {'Full Cascade':<15} {'Change':<10}")
print("-"*55)
print(f"{'Accuracy':<15} {acc_rf*100:.2f}%{'':<7} {accuracy*100:.2f}%{'':<10} {(accuracy-acc_rf)*100:+.2f}%")
print(f"{'F1 Score':<15} {f1_rf*100:.2f}%{'':<7} {f1*100:.2f}%{'':<10} {(f1-f1_rf)*100:+.2f}%")

print(f"\n📊 Cascade Distribution:")
print(f"   CBA:                {paths.count('CBA')}")
print(f"   RF-Phishing:        {paths.count('RF-Phishing')}")
print(f"   RF-Legitimate:      {paths.count('RF-Legitimate')}")
print(f"   Content-Phishing:   {paths.count('Content-Phishing')}")
print(f"   Content-Legitimate: {paths.count('Content-Legitimate')}")
print(f"   Content-Uncertain:  {paths.count('Content-Uncertain')}")
print(f"   Content-Failed:     {paths.count('Content-Failed')}")
print(f"   LLM:                {llm_count}")

print(f"\n📊 Confusion Matrix:")
print(f"   TN: {cm[0,0]}  |  FP: {cm[0,1]}")
print(f"   FN: {cm[1,0]}  |  TP: {cm[1,1]}")

# ============================================================
# GRAPHS
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'],
            cbar=False, annot_kws={'size': 16, 'weight': 'bold'},
            linewidths=2, linecolor='white', ax=axes[0])
axes[0].set_title(f'A. RF Only\nAcc: {acc_rf*100:.1f}%  |  F1: {f1_rf*100:.1f}%',
                  fontsize=13, fontweight='bold')

sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'],
            cbar=False, annot_kws={'size': 16, 'weight': 'bold'},
            linewidths=2, linecolor='white', ax=axes[1])
axes[1].set_title(f'B. Full Cascade\nAcc: {accuracy*100:.1f}%  |  F1: {f1*100:.1f}%',
                  fontsize=13, fontweight='bold')

plt.suptitle('Confusion Matrix: RF Only vs Full Cascade',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_FOLDER}/confusion_matrix_final.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"\n✅ Saved: {OUTPUT_FOLDER}/confusion_matrix_final.png")

print("\n✅ COMPLETE!")