# test_final_hybrid_with_live_urls.py
"""
FINAL Hybrid Test WITH Live PhishTank URLs
Combines: Live phishing + Legitimate + UCI samples
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

PHISHING_THRESHOLD = 0.65
SUSPICIOUS_THRESHOLD = 0.45
FETCH_TIMEOUT = 5

OUTPUT_FOLDER = 'validation_graphs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# LIVE PHISHTANK URLS (From your screenshots)
# ============================================================
PHISHTANK_LIVE_URLS = [
    "http://coinbase.979217.com",
    "https://chlyzi.com/login/smbc-card/index.html/entath/jsp",
    "http://allegro.ofer09128875.buzz",
    "http://allegrolokalnie.oferta198273129.cfd",
    "http://allegro.1230120h85.buzz",
    "https://www.wznlmy.com/?email=",
    "https://www.bjmg.com/rjrxch1c",
    "https://espaces-clients-prime.com/g7GgFaHztFepM/7yujen=Nefretic3@laposte",
    "https://retractations-service.com/g7GgFaHztFepM/7yujen=Nefretic3@laposte",
    "http://www.allegrolokalnie.7418541.cfd",
    "https://mtsakielgeuin.webflow.io",
    "https://bnmail-awa.vercel.app",
    "https://a-beautifulwoman.net/wp-includes/indx.html",
    "https://metamassklogin.webflow.io",
    "https://service-helpercenter9192.netlify.app",
    "https://oluwaburnazip-renushahrawat.replit.app",
    "https://security-server-landing-page--lme85959.replit.app",
    "https://security-server-landing-page--spencer-hunt1.replit.app",
    "https://jh1183.craftum.io",
    "https://www.ebay.item.co.uk.login-secures-wr8wsvzt.sutranajahotelkolak",
    "https://imapa.services/isignesp.php",
    "http://allegrolokalnie.oferta27631623.cfd",
    "https://retractations-cdiscount.com/g7GgFaHztFepM/7yujen=malourd45@laposte.net",
    "https://portal-zen.com/g7GgFaHztFepM/7yujen=malourd45@laposte.net",
    "http://allegro.131200h88.buzz",
    "http://allegro.tewo23923.buzz",
    "https://contact-mobile-43-r-53.us-west-1.amazonaws.com/uzi6",
    "https://track.pstmrk.it/3s/v-mallo2oo.com/",
    "https://v-mallo2oo.com/?token=y1pm20sx56wb&session_id=58595",
    "https://service-web-500.s3.dualstack.eu-west-2.amazonaws.com/9pmi",
    "http://allegro.12300120h9j8j.buzz",
    "https://allegrolokalnie.rezerwacja81258.click/logowanie?referrer=http",
    "https://allegrolokalnie.rezerwacja81258.click/nowa-elektryczna-hulajno",
    "https://atualizarnvtaxas.co/7xUe1cYp2Ql0vWCm",
    "https://sinarmasmarketing-product.pages.dev/",
    "https://allegrolokalnie.oferta324452.cfd",
    "http://allegrolokalnie.lokalna-ofeta78685134.sbs",
]

# Legitimate URLs to test (should pass)
LEGITIMATE_URLS = [
    "https://www.google.com",
    "https://www.facebook.com",
    "https://www.amazon.in",
    "https://www.hdfcbank.com",
    "https://www.sbi.co.in",
    "https://www.icicibank.com",
    "https://www.flipkart.com",
    "https://www.paypal.com",
    "https://github.com",
    "https://stackoverflow.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.netflix.com",
    "https://twitter.com",
    "https://www.linkedin.com",
    "https://www.instagram.com",
    "https://www.youtube.com",
    "https://www.wikipedia.org",
    "https://www.reddit.com",
    "https://www.quora.com",
]

OFFICIAL_DOMAINS = [
    'google.com', 'facebook.com', 'amazon.com', 'amazon.in', 'microsoft.com',
    'apple.com', 'paypal.com', 'netflix.com', 'instagram.com', 'twitter.com',
    'linkedin.com', 'youtube.com', 'whatsapp.com', 'binance.com', 'coinbase.com',
    'hdfcbank.com', 'sbi.co.in', 'icicibank.com', 'axisbank.com', 'kotak.com',
    'flipkart.com', 'github.com', 'stackoverflow.com', 'wikipedia.org',
    'reddit.com', 'quora.com',
]

URGENCY_KEYWORDS = [
    'urgent', 'immediate', 'verify now', 'act now', 'limited time',
    'account locked', 'account suspended',
]

print("="*70)
print("FINAL HYBRID TEST - LIVE PHISHTANK URLs")
print("="*70)
print(f"Live Phishing URLs: {len(PHISHTANK_LIVE_URLS)}")
print(f"Legitimate URLs: {len(LEGITIMATE_URLS)}")
print(f"Total Test Size: {len(PHISHTANK_LIVE_URLS) + len(LEGITIMATE_URLS)}")
print("="*70)

# ============================================================
# BUILD TEST SET
# ============================================================
test_urls = PHISHTANK_LIVE_URLS + LEGITIMATE_URLS
test_labels = [1] * len(PHISHTANK_LIVE_URLS) + [0] * len(LEGITIMATE_URLS)

# Shuffle
np.random.seed(42)
indices = np.random.permutation(len(test_urls))
test_urls = [test_urls[i] for i in indices]
test_labels = [test_labels[i] for i in indices]

# Clean URLs for feature extraction
def clean_url(url):
    url = str(url).strip().lower()
    # Keep original but clean for feature extraction
    return url

urls_list = [clean_url(u) for u in test_urls]
y_test = np.array(test_labels)

print(f"\nTest Set Built:")
print(f"  Phishing: {sum(y_test==1)}")
print(f"  Legitimate: {sum(y_test==0)}")

# ============================================================
# URL FEATURES
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
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', 
                       '.buzz', '.cfd', '.sbs', '.click', '.live', '.site']
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
# CONTENT ANALYSIS
# ============================================================
def analyze_page_content(url):
    signals = []
    score = 0.0
    
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, timeout=FETCH_TIMEOUT, headers=headers,
                               allow_redirects=True, verify=False)
        html = response.text
        final_url = response.url
    except:
        return -1, ["fetch_failed"], "failed"
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
    except:
        return -1, ["parse_failed"], "failed"
    
    parsed = urlparse(final_url)
    page_domain = parsed.netloc.lower().replace('www.', '')
    is_official = any(page_domain.endswith(od) for od in OFFICIAL_DOMAINS)
    
    if is_official:
        return 0.0, ["official_domain"], "success"
    
    # Signals
    if soup.find_all('input', {'type': 'password'}):
        score += 0.30
        signals.append("password_field")
    
    for form in soup.find_all('form'):
        action = form.get('action', '')
        if action:
            action_url = urljoin(url, action)
            action_domain = urlparse(action_url).netloc.lower()
            if action_domain and action_domain != page_domain:
                score += 0.40
                signals.append("form_external")
                break
    
    text_content = soup.get_text().lower()
    urgency = sum(1 for kw in URGENCY_KEYWORDS if kw in text_content)
    if urgency >= 2:
        score += 0.15
        signals.append(f"urgency:{urgency}")
    
    if not ('©' in text_content or 'copyright' in text_content):
        score += 0.05
        signals.append("no_copyright")
    
    for script in soup.find_all('script', src=False):
        if script.string and 'window.location' in script.string:
            score += 0.20
            signals.append("js_redirect")
            break
    
    return min(score, 1.0), signals, "success"

# ============================================================
# GROQ LLM (FIXED)
# ============================================================
def call_groq_llm(url, prob, signals):
    signals_text = ", ".join(signals[:3]) if signals else "none"
    
    prompt = f"""Analyze this URL for phishing:

URL: {url}
ML confidence: {prob:.1%}
Content signals: {signals_text}

Is this URL phishing? Respond EXACTLY:
VERDICT: PHISHING or LEGITIMATE
CONFIDENCE: XX%"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=500,  # FIXED
        )
        
        text = response.choices[0].message.content
        text_upper = text.upper()
        
        # Robust parsing
        if 'PHISHING' in text_upper and 'NOT PHISHING' not in text_upper:
            verdict = 'phishing'
        else:
            verdict = 'legitimate'
        
        conf_match = re.search(r'CONFIDENCE[:\s]*(\d+)', text_upper)
        confidence = int(conf_match.group(1)) / 100 if conf_match else 0.5
        
        return verdict, confidence
        
    except Exception as e:
        print(f"      ❌ Groq: {str(e)[:80]}")
        return 'legitimate', 0.5

# ============================================================
# LOAD MODEL
# ============================================================
print("\n📦 Loading Random Forest model...")
rf = joblib.load('models/random_forest_final_v3.pkl')
scaler = joblib.load('models/scaler.pkl')

feature_cols = ['url_length', 'domain_length', 'subdomain_length', 'path_length',
                'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
                'num_digits', 'num_special_chars', 'num_sensitive_words',
                'has_@', 'has_https', 'has_ip', 'has_suspicious_tld', 'is_shortened']

url_features = pd.DataFrame([extract_url_features(u) for u in urls_list])
url_features = url_features[feature_cols]
X_scaled = scaler.transform(url_features)
y_proba = rf.predict_proba(X_scaled)[:, 1]

print("✅ Model ready")

# ============================================================
# CBA RULES
# ============================================================
def apply_cba(url):
    triggered = []
    if any(url.endswith(t) for t in ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.buzz', '.cfd', '.sbs', '.click', '.live']):
        triggered.append(0.85)
    if re.search(r'\d+\.\d+\.\d+\.\d+', url):
        triggered.append(0.95)
    if '@' in url:
        triggered.append(0.90)
    return (True, sum(triggered)/len(triggered)) if triggered else (False, 0)

# ============================================================
# RUN CASCADE
# ============================================================
print("\n" + "="*70)
print("🔄 RUNNING CASCADE ON LIVE URLs")
print("="*70)

final_preds = []
paths = []
suspicious_indices = []
llm_count = 0
content_success = 0
content_failed = 0

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

# Layer 3: Content
print(f"\n🌐 Content analysis for {len(suspicious_indices)} URLs...")

content_uncertain = []

for count, idx in enumerate(suspicious_indices, 1):
    url = urls_list[idx]
    try:
        score, signals, status = analyze_page_content(url)
        
        if status == 'success':
            content_success += 1
            if score >= 0.65:
                final_preds.append(1)
                paths[paths.index('SUSPICIOUS')] = 'Content-Phishing'
                print(f"   [{count}/{len(suspicious_indices)}] 🚨 {url[:45]} | Score: {score:.2f}")
            elif score < 0.20:
                final_preds.append(0)
                paths[paths.index('SUSPICIOUS')] = 'Content-Legitimate'
                print(f"   [{count}/{len(suspicious_indices)}] ✅ {url[:45]} | Score: {score:.2f}")
            else:
                content_uncertain.append((idx, score, signals))
                paths[paths.index('SUSPICIOUS')] = 'Content-Uncertain'
                print(f"   [{count}/{len(suspicious_indices)}] ❓ {url[:45]} | Score: {score:.2f}")
        else:
            content_failed += 1
            content_uncertain.append((idx, y_proba[idx], [f"fetch_{status}"]))
            paths[paths.index('SUSPICIOUS')] = 'Content-Failed'
            print(f"   [{count}/{len(suspicious_indices)}] ❌ {url[:45]} | Failed")
    except Exception as e:
        content_failed += 1
        content_uncertain.append((idx, y_proba[idx], ["error"]))
        paths[paths.index('SUSPICIOUS')] = 'Content-Failed'

print(f"\n✅ Content: Success={content_success}, Failed={content_failed}")

# Layer 4: LLM
if content_uncertain:
    print(f"\n🧠 LLM for {len(content_uncertain)} cases...")
    
    for count, (idx, score, signals) in enumerate(content_uncertain, 1):
        url = urls_list[idx]
        prob = y_proba[idx]
        
        verdict, confidence = call_groq_llm(url, prob, signals)
        final_preds.append(1 if verdict == 'phishing' else 0)
        llm_count += 1
        
        emoji = "🚨" if verdict == 'phishing' else "✅"
        print(f"   [{count}/{len(content_uncertain)}] {emoji} {url[:40]} → {verdict.upper()} ({confidence:.0%})")
        
        if count < len(content_uncertain):
            time.sleep(2)

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
print(f"{'Precision':<15} {precision_score(y_test, y_pred_rf, zero_division=0)*100:.2f}%{'':<7} {precision*100:.2f}%")
print(f"{'Recall':<15} {recall_score(y_test, y_pred_rf, zero_division=0)*100:.2f}%{'':<7} {recall*100:.2f}%")
print(f"{'F1 Score':<15} {f1_rf*100:.2f}%{'':<7} {f1*100:.2f}%{'':<10} {(f1-f1_rf)*100:+.2f}%")

print(f"\n📊 Cascade Distribution:")
print(f"   CBA:                {paths.count('CBA')}")
print(f"   RF-Phishing:        {paths.count('RF-Phishing')}")
print(f"   RF-Legitimate:      {paths.count('RF-Legitimate')}")
print(f"   Content-Phishing:   {paths.count('Content-Phishing')}")
print(f"   Content-Legitimate: {paths.count('Content-Legitimate')}")
print(f"   Content-Failed:     {paths.count('Content-Failed')}")
print(f"   LLM decided:        {llm_count}")

print(f"\n📊 Confusion Matrix:")
print(f"   TN: {cm[0,0]}  |  FP: {cm[0,1]}")
print(f"   FN: {cm[1,0]}  |  TP: {cm[1,1]}")

# ============================================================
# GRAPHS
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'],
            cbar=False, annot_kws={'size': 16, 'weight': 'bold'},
            linewidths=2, linecolor='white', ax=axes[0])
axes[0].set_title(f'A. RF Only\nAcc: {acc_rf*100:.1f}% | F1: {f1_rf*100:.1f}%',
                  fontsize=13, fontweight='bold')

sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'],
            cbar=False, annot_kws={'size': 16, 'weight': 'bold'},
            linewidths=2, linecolor='white', ax=axes[1])
axes[1].set_title(f'B. Full Cascade (Live URLs)\nAcc: {accuracy*100:.1f}% | F1: {f1*100:.1f}%',
                  fontsize=13, fontweight='bold')

plt.suptitle('Confusion Matrix: Live PhishTank URLs',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_FOLDER}/confusion_matrix_live_urls.png', dpi=200, bbox_inches='tight')
plt.close()

# Save results to CSV
results_df = pd.DataFrame({
    'URL': urls_list,
    'Actual': y_test,
    'RF_Probability': y_proba,
    'Cascade_Path': paths,
    'Final_Verdict': final_preds,
    'Correct': (final_preds == y_test).astype(int)
})
results_df.to_csv(f'{OUTPUT_FOLDER}/live_urls_results.csv', index=False)

print(f"\n✅ Saved: {OUTPUT_FOLDER}/confusion_matrix_live_urls.png")
print(f"✅ Saved: {OUTPUT_FOLDER}/live_urls_results.csv")
print("\n✅ COMPLETE!")