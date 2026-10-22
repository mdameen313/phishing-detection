# test_content_cascade.py
"""
FINAL Hybrid Test - Best Performing Version
- RF threshold at 0.55
- Hard LLM prompt (Default toward Phishing if fetch failed)
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
import tldextract
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from groq import Groq
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
GROQ_API_KEY = "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"
GROQ_MODEL = "openai/gpt-oss-120b"
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

TEST_SIZE = 200
PHISHING_THRESHOLD = 0.65
SUSPICIOUS_THRESHOLD = 0.45
RF_BASELINE_THRESHOLD = 0.50
CONTENT_PHISHING_SCORE = 0.65
CONTENT_LEGIT_SCORE = 0.30
FETCH_TIMEOUT = 5

CLEAN_CONTENT_RF_ESCALATION = 0.55 

OUTPUT_FOLDER = 'validation_graphs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OFFICIAL_DOMAINS = [
    'google.com', 'facebook.com', 'amazon.com', 'amazon.in', 'microsoft.com',
    'apple.com', 'paypal.com', 'netflix.com', 'instagram.com', 'twitter.com',
    'linkedin.com', 'youtube.com', 'whatsapp.com', 'binance.com', 'coinbase.com',
    'hdfcbank.com', 'sbi.co.in', 'icicibank.com', 'axisbank.com', 'kotak.com',
    'flipkart.com', 'snapchat.com', 'tiktok.com', 'spotify.com', 'github.com',
    'metamask.io', 'trustwallet.com', 'steampowered.com', 'dropbox.com',
]

BRAND_KEYWORDS = [
    'paypal', 'amazon', 'google', 'microsoft', 'apple', 'facebook',
    'instagram', 'netflix', 'flipkart', 'hdfc', 'sbi', 'icici',
    'axis', 'kotak', 'binance', 'coinbase', 'metamask', 'steam',
    'spotify', 'twitter', 'linkedin', 'whatsapp', 'snapchat', 'tiktok',
    'co-operativebank', 'cooperativebank'
]

URGENCY_KEYWORDS = [
    'urgent', 'immediate', 'verify now', 'act now', 'limited time',
    'account locked', 'account suspended', 'immediate action',
    'verify your account', 'click here', 'act fast', 'expires soon',
    'last chance', 'final warning',
]

SUSPICIOUS_TLDS = {'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'click', 'link', 'download', 'review', 'stream'}

SUSPICIOUS_PATH_PATTERNS = [
    r'/wp-includes/.*\.(php|html|css)',
    r'/wp-content/.*/[0-9a-f]{6,}',
    r'\.(apk|exe|scr|bat|jar)(\?|$)',
    r'/(login|signin|verify|secure|update)[-_]?(account|now|info)?\.(php|html)',
    r'/[0-9a-f]{16,}',
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_suspicious_tld(url_clean):
    ext = tldextract.extract(url_clean)
    return ext.suffix.split('.')[-1] in SUSPICIOUS_TLDS if ext.suffix else False

def is_onion(url_clean):
    ext = tldextract.extract(url_clean)
    return ext.suffix == 'onion' or url_clean.endswith('.onion') or '.onion/' in url_clean

def brand_impersonation_in_hostname(url_clean):
    ext = tldextract.extract(url_clean)
    registered_domain = f"{ext.domain}.{ext.suffix}".lower()
    full_host = f"{ext.subdomain}.{ext.domain}.{ext.suffix}".lower()
    for brand in BRAND_KEYWORDS:
        if brand in full_host and brand not in registered_domain:
            return True
    return False

def url_path_risk_signals(url_clean):
    return [p for p in SUSPICIOUS_PATH_PATTERNS if re.search(p, url_clean, re.I)]

# ============================================================
# LOAD URLS
# ============================================================
def find_file(filenames):
    for name in filenames:
        for path in [name, f'../{name}', f'processed_data/{name}', f'../../{name}']:
            if os.path.exists(path):
                return path
    raise FileNotFoundError(f"Could not find any of: {filenames}")

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

malicious_df = standardize(malicious_df, 'url', 'type', ['phishing', 'malware', 'defacement', 'spam'], ['benign', 'good'])
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
    malicious_df[malicious_df['label'] == 1][['url_clean']].assign(label=1),
    phishing_df[phishing_df['label'] == 1][['url_clean']].assign(label=1)
]).drop_duplicates(subset=['url_clean'])

all_legit = pd.concat([
    malicious_df[malicious_df['label'] == 0][['url_clean']].assign(label=0),
    phishing_df[phishing_df['label'] == 0][['url_clean']].assign(label=0)
]).drop_duplicates(subset=['url_clean'])

conflict = set(all_phishing['url_clean']) & set(all_legit['url_clean'])
all_phishing = all_phishing[~all_phishing['url_clean'].isin(conflict)]
all_legit = all_legit[~all_legit['url_clean'].isin(conflict)]

np.random.seed(42)
test_urls_df = pd.concat([
    all_phishing.sample(n=HALF, random_state=42),
    all_legit.sample(n=HALF, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

# ============================================================
# EXTRACT URL FEATURES
# ============================================================
def extract_url_features(url):
    url = str(url)
    has_https = 1 if url.startswith('https://') else 0
    clean = re.sub(r'^https?://', '', url)
    clean = re.sub(r'^www\.', '', clean).rstrip('/')

    try:
        parsed = urlparse(url if '://' in url else f'http://{url}')
        domain_info = tldextract.extract(clean)
    except Exception:
        domain_info = tldextract.extract('')
        parsed = urlparse('')

    sensitive = ['login', 'verify', 'secure', 'update', 'confirm', 'password', 'credit', 'signin', 'account', 'validate']
    shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 'shorturl', 'is.gd', 'ow.ly']

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
        'has_suspicious_tld': 1 if get_suspicious_tld(clean) else 0,
        'is_shortened': 1 if any(s in clean for s in shorteners) else 0,
    }

# ============================================================
# CONTENT ANALYSIS
# ============================================================
def host_matches_official(host):
    host = host.lower().replace('www.', '')
    return any(host == od or host.endswith('.' + od) for od in OFFICIAL_DOMAINS)

def analyze_page_content(url):
    signals = []
    score = 0.0

    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, timeout=FETCH_TIMEOUT, headers=headers, allow_redirects=True)
        html = response.text
        final_url = response.url
    except Exception as e:
        return 0.0, [f"fetch_failed: {str(e)[:50]}"], "failed"

    if not html or len(html.strip()) == 0:
        return 0.0, ["empty_page"], "failed"

    try:
        soup = BeautifulSoup(html, 'html.parser')
    except Exception:
        return 0.0, ["parse_failed"], "failed"

    parsed = urlparse(final_url)
    page_domain = parsed.netloc.lower().replace('www.', '')
    is_official = host_matches_official(page_domain)

    title = soup.title.string.lower() if soup.title and soup.title.string else ''
    for brand in BRAND_KEYWORDS:
        if re.search(rf'\b{re.escape(brand)}\b', title) and brand not in page_domain and not is_official:
            score += 0.20
            signals.append(f"title_brand_mismatch: '{brand}'")
            break

    password_fields = soup.find_all('input', {'type': 'password'})
    if password_fields and not is_official:
        score += 0.30
        signals.append("password_field_on_wrong_domain")

    html_lower = html.lower()
    if any(k in html_lower for k in ['card_number', 'cardnumber', 'cvv']) and not is_official:
        score += 0.30
        signals.append("credit_card_field_on_wrong_domain")

    external_form_found = False
    for form in soup.find_all('form'):
        action = form.get('action', '')
        if action and not action.lower().startswith(('javascript:', '#', 'mailto:')):
            action_domain = urlparse(urljoin(final_url, action)).netloc.lower().replace('www.', '')
            if action_domain and action_domain != page_domain and not host_matches_official(action_domain):
                external_form_found = True
                break
    if external_form_found:
        score += 0.40
        signals.append("form_action_external_domain")

    text_content = soup.get_text(separator=' ').lower()
    urgency_count = sum(1 for kw in URGENCY_KEYWORDS if kw in text_content)
    if urgency_count >= 2:
        score += 0.15
        signals.append(f"urgency_keywords ({urgency_count})")

    favicon = soup.find('link', rel=re.compile(r'icon', re.I))
    if favicon and favicon.get('href'):
        favicon_domain = urlparse(urljoin(final_url, favicon['href'])).netloc.lower().replace('www.', '')
        if favicon_domain and favicon_domain != page_domain and not host_matches_official(favicon_domain):
            score += 0.15
            signals.append("favicon_external_domain")

    external_scripts = 0
    for script in soup.find_all('script', src=True):
        script_domain = urlparse(urljoin(final_url, script['src'])).netloc.lower().replace('www.', '')
        if script_domain and script_domain != page_domain:
            external_scripts += 1
    if external_scripts > 15 and not is_official:
        score += 0.05
        signals.append(f"many_external_scripts ({external_scripts})")

    for script in soup.find_all('script', src=False):
        if script.string:
            m = re.search(r'(?:window|document)\.location(?:\.href)?\s*=\s*[\'"]([^\'"]+)[\'"]', script.string)
            if m:
                target_domain = urlparse(urljoin(final_url, m.group(1))).netloc.lower().replace('www.', '')
                if target_domain and target_domain != page_domain and not host_matches_official(target_domain):
                    score += 0.20
                    signals.append(f"js_redirect_external: -> {target_domain}")
                    break

    if soup.find('meta', attrs={'http-equiv': re.compile(r'refresh', re.I)}):
        score += 0.20
        signals.append("meta_refresh_redirect")

    hidden_fields = soup.find_all('input', {'type': 'hidden'})
    if len(hidden_fields) > 8:
        score += 0.10
        signals.append(f"many_hidden_fields ({len(hidden_fields)})")

    iframes = soup.find_all('iframe')
    if len(iframes) > 6 and not is_official:
        score += 0.05
        signals.append(f"many_iframes ({len(iframes)})")

    return min(score, 1.0), signals, "success"

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

url_features = pd.DataFrame([extract_url_features(u) for u in test_urls_df['url_clean']])
url_features = url_features[feature_cols]
X_scaled = scaler.transform(url_features)
y_proba = rf.predict_proba(X_scaled)[:, 1]

y_test = test_urls_df['label'].values
urls_list = test_urls_df['url_clean'].tolist()
N = len(urls_list)

# ============================================================
# CBA RULES
# ============================================================
def apply_cba(url):
    triggered = []
    if get_suspicious_tld(url):
        triggered.append(0.85)
    if is_onion(url):
        triggered.append(0.95)
    if brand_impersonation_in_hostname(url):
        triggered.append(0.90)
    if re.search(r'\d+\.\d+\.\d+\.\d+', url):
        triggered.append(0.95)
    if '@' in url:
        triggered.append(0.90)
    shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 'shorturl', 'is.gd']
    if any(s in url for s in shorteners):
        triggered.append(0.70)
    return (True, sum(triggered) / len(triggered)) if triggered else (False, 0)

# ============================================================
# GROQ LLM (HARD PROMPT)
# ============================================================
def call_groq_llm(url, prob, content_signals, fetch_status="success"):
    if client is None:
        return ('phishing' if prob >= 0.50 else 'legitimate'), 0.5

    path_hits = url_path_risk_signals(url)
    onion_flag = is_onion(url)

    if fetch_status == "failed":
        evidence_note = (
            "The page content could NOT be fetched (dead link, blocked bot access, or offline). "
            "This URL already passed through an ML classifier that flagged it as suspicious. "
            "A failed fetch is NOT evidence of legitimacy. Only call this legitimate if the "
            "domain itself is clearly a long-established, well-known entity."
        )
    else:
        evidence_note = "Page content was fetched and analyzed."

    path_note = f"\nSuspicious URL-path patterns: {path_hits}" if path_hits else ""
    onion_note = "\nThis is a Tor .onion address." if onion_flag else ""
    signals_text = ", ".join(content_signals[:5]) if content_signals else "none"

    prompt = f"""Cybersecurity analysis of suspicious URL:

URL: {url}
ML confidence: {prob:.1%}
Content fetch status: {fetch_status}
{evidence_note}
Content signals: {signals_text}{path_note}{onion_note}

This URL was already flagged suspicious by an ML model — you are the final check.
Default toward PHISHING unless there is concrete evidence it is an established site.

Reply in exactly this format:
Verdict: YES or NO
Confidence: XX%"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=1000,
        )
        text = response.choices[0].message.content or ""
        verdict_match = re.search(r'Verdict:\s*\**\s*(YES|NO)', text, re.I)
        verdict = ('phishing' if verdict_match.group(1).upper() == 'YES' else 'legitimate') if verdict_match else ('phishing' if prob >= 0.50 else 'legitimate')
        conf_match = re.search(r'Confidence:\s*\**\s*(\d+)\s*%', text)
        confidence = int(conf_match.group(1)) / 100 if conf_match else 0.5
        return verdict, confidence
    except Exception:
        return ('phishing' if prob >= 0.50 else 'legitimate'), 0.5

# ============================================================
# RUN CASCADE
# ============================================================
print("\n🔄 RUNNING CASCADE...")

final_preds = np.full(N, -1, dtype=int)
paths = [None] * N
suspicious_indices = []
content_decided = 0
content_failed = 0
llm_count = 0
diagnostic_rows = []

# Layer 1: CBA
for i in range(N):
    is_phish, conf = apply_cba(urls_list[i])
    if is_phish and conf >= 0.70:
        final_preds[i] = 1
        paths[i] = 'CBA'

print(f"✅ CBA: {paths.count('CBA')}")

# Layer 2: RF
for i in range(N):
    if paths[i] is not None:
        continue
    prob = y_proba[i]
    if prob >= PHISHING_THRESHOLD:
        final_preds[i] = 1
        paths[i] = 'RF-Phishing'
    elif prob >= SUSPICIOUS_THRESHOLD:
        suspicious_indices.append(i)
        paths[i] = 'SUSPICIOUS'
    else:
        final_preds[i] = 0
        paths[i] = 'RF-Legitimate'

print(f"✅ RF-Phishing: {paths.count('RF-Phishing')}")
print(f"✅ RF-Legitimate: {paths.count('RF-Legitimate')}")
print(f"⚠️ Suspicious: {len(suspicious_indices)}")

# Layer 3: Content Analysis
content_uncertain = []
if suspicious_indices:
    print(f"\n🌐 Content analysis for {len(suspicious_indices)} URLs...")
    for count, idx in enumerate(suspicious_indices, 1):
        url = urls_list[idx]
        true_label = y_test[idx]
        
        try:
            score, signals, status = analyze_page_content(url)
            
            if status == "failed":
                content_failed += 1
                if y_proba[idx] >= 0.50:
                    final_preds[idx] = 1
                    paths[idx] = 'FetchFailed-Phishing'
                    content_decided += 1
                    print(f"   [{count}/{len(suspicious_indices)}] 🚨 Fetch Failed -> Phishing | {url[:40]}")
                else:
                    content_uncertain.append((idx, score, signals, status))
                    paths[idx] = 'Content-Failed'
                    print(f"   [{count}/{len(suspicious_indices)}] ❌ Fetch Failed -> LLM | {url[:40]}")
            
            elif score >= CONTENT_PHISHING_SCORE:
                final_preds[idx] = 1
                paths[idx] = 'Content-Phishing'
                content_decided += 1
                print(f"   [{count}/{len(suspicious_indices)}] 🚨 Content Phishing ({score:.2f}) | {url[:40]}")
            
            # 0.55 cutoff
            elif score < CONTENT_LEGIT_SCORE and y_proba[idx] < CLEAN_CONTENT_RF_ESCALATION:
                final_preds[idx] = 0
                paths[idx] = 'Content-Legitimate'
                content_decided += 1
                print(f"   [{count}/{len(suspicious_indices)}] ✅ Content Legit ({score:.2f}) & Low RF | {url[:40]}")
            
            else:
                content_uncertain.append((idx, score, signals, status))
                paths[idx] = 'Content-Uncertain'
                print(f"   [{count}/{len(suspicious_indices)}] ❓ Clean but High RF -> LLM | {url[:40]}")

            diagnostic_rows.append({
                'url': url, 'true_label': true_label, 'rf_prob': y_proba[idx],
                'content_score': score, 'status': status,
                'signals': '; '.join(signals), 'path': paths[idx]
            })

        except Exception as e:
            content_failed += 1
            content_uncertain.append((idx, 0.5, ["error"], "failed"))
            paths[idx] = 'Content-Failed'

# Layer 4: LLM
if content_uncertain:
    print(f"\n🧠 LLM for {len(content_uncertain)} cases...")
    for count, (idx, content_score, signals, status) in enumerate(content_uncertain, 1):
        url = urls_list[idx]
        prob = y_proba[idx]
        
        verdict, confidence = call_groq_llm(url, prob, signals, fetch_status=status)
        final_preds[idx] = 1 if verdict == 'phishing' else 0
        paths[idx] = f'LLM-{verdict.capitalize()}'
        llm_count += 1
        
        emoji = "🚨" if verdict == 'phishing' else "✅"
        print(f"   [{count}/{len(content_uncertain)}] {emoji} LLM: {verdict.upper()} | {url[:40]}")
        
        if count < len(content_uncertain):
            time.sleep(1.5)

# Export diagnostics
pd.DataFrame(diagnostic_rows).to_csv(f'{OUTPUT_FOLDER}/suspicious_url_diagnostics.csv', index=False)

# ============================================================
# RESULTS
# ============================================================
assert (final_preds >= 0).all(), "Error: Some URLs were not predicted!"

cm = confusion_matrix(y_test, final_preds, labels=[0, 1])
accuracy = accuracy_score(y_test, final_preds)
precision = precision_score(y_test, final_preds, zero_division=0)
recall = recall_score(y_test, final_preds, zero_division=0)
f1 = f1_score(y_test, final_preds, zero_division=0)

y_pred_rf = (y_proba >= RF_BASELINE_THRESHOLD).astype(int)
cm_rf = confusion_matrix(y_test, y_pred_rf, labels=[0, 1])
acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, zero_division=0)
rec_rf = recall_score(y_test, y_pred_rf, zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)

print("\n" + "=" * 70)
print("📊 FINAL RESULTS")
print("=" * 70)
print(f"\n{'Metric':<12}{'RF Only':>12}{'Full Cascade':>16}{'Change':>12}")
print("-" * 52)
for name, a, b in [('Accuracy', acc_rf, accuracy), ('Precision', prec_rf, precision),
                   ('Recall', rec_rf, recall), ('F1 Score', f1_rf, f1)]:
    print(f"{name:<12}{a*100:>11.2f}%{b*100:>15.2f}%{(b-a)*100:>+11.2f}%")

print(f"\n📊 Confusion Matrix (Full Cascade):")
print(f"   TN: {cm[0,0]}  |  FP: {cm[0,1]}")
print(f"   FN: {cm[1,0]}  |  TP: {cm[1,1]}")

# Graphs
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Reds', cbar=False, ax=axes[0], annot_kws={'size': 16})
axes[0].set_title(f'A. RF Only\nAcc: {acc_rf*100:.1f}% | F1: {f1_rf*100:.1f}%')
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[1], annot_kws={'size': 16})
axes[1].set_title(f'B. Full Cascade\nAcc: {accuracy*100:.1f}% | F1: {f1*100:.1f}%')
plt.savefig(f'{OUTPUT_FOLDER}/confusion_matrix_content_cascade.png', dpi=200, bbox_inches='tight')
plt.close()

print("\n✅ Saved: validation_graphs/confusion_matrix_content_cascade.png")
print("✅ Saved: validation_graphs/suspicious_url_diagnostics.csv")
print("\n✅ COMPLETE!")