# test_live_urls.py
import pandas as pd
import numpy as np
import joblib
import re
import time
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import tldextract
from groq import Groq
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
GROQ_API_KEY = "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"
GROQ_MODEL = "openai/gpt-oss-120b"
client = Groq(api_key=GROQ_API_KEY)

SUSPICIOUS_TLDS = {'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'click', 'link', 'download', 'review', 'stream', 'buzz', 'cfd', 'sbs', 'lat', 'vip'}
FREE_HOSTING = ['netlify.app', 'azurewebsites.net', 'webflow.io', 'replit.app', 'pages.dev', 'vercel.app', 'github.io', 'herokuapp.com', 'surge.sh']
SUSPICIOUS_PATH_PATTERNS = [
    r'/wp-includes/.*\.(php|html|css)',
    r'/wp-content/.*/[0-9a-f]{6,}',
    r'\.(apk|exe|scr|bat|jar)(\?|$)',
    r'/(login|signin|verify|secure|update)[-_]?(account|now|info)?\.(php|html)',
    r'/[0-9a-f]{16,}',
]
URGENCY_KEYWORDS = ['urgent', 'immediate', 'verify now', 'act now', 'limited time', 'account locked', 'account suspended', 'immediate action', 'verify your account', 'click here', 'act fast', 'expires soon', 'last chance', 'final warning']
BRAND_KEYWORDS = ['paypal', 'amazon', 'google', 'microsoft', 'apple', 'facebook', 'instagram', 'netflix', 'flipkart', 'hdfc', 'sbi', 'icici', 'axis', 'kotak', 'binance', 'coinbase', 'metamask', 'steam', 'spotify', 'twitter', 'linkedin', 'whatsapp', 'snapchat', 'tiktok', 'co-operativebank', 'cooperativebank']
OFFICIAL_DOMAINS = ['google.com', 'facebook.com', 'amazon.com', 'amazon.in', 'microsoft.com', 'apple.com', 'paypal.com', 'netflix.com', 'instagram.com', 'twitter.com', 'linkedin.com', 'youtube.com', 'whatsapp.com', 'binance.com', 'coinbase.com', 'hdfcbank.com', 'sbi.co.in', 'icicibank.com', 'axisbank.com', 'kotak.com', 'flipkart.com', 'snapchat.com', 'tiktok.com', 'spotify.com', 'github.com', 'metamask.io', 'trustwallet.com', 'steampowered.com', 'dropbox.com']

# ============================================================
# TEST URLS
# ============================================================
phishing_urls = [
    "https://droits-prime-contestations.com/",
    "https://droits-prime-contestations.com/blog/blog.php",
    "https://droits-prime-contestations.com/index.php",
    "https://famous-dango-d57402.netlify.app/",
    "https://allegrolokalnie.kup-teraz-516833-pl.help",
    "https://h81870982.pro/",
    "https://h81870982.pro/suivi",
    "https://random.cnwsfert.com/-login",
    "http://allegro.4787155.lat",
    "https://allegrolokalnie.4787155.lat",
    "http://www.skvurthaler.vip/",
    "https://www.republic-trade.team/",
    "http://republic-trade.team/",
    "https://glittering-moxie-783c0b.netlify.app/",
    "http://bjmsjcfsh.com",
    "https://stellar-ganache-88ceb2.netlify.app/",
    "https://abcappseitienst.azurewebsites.net/",
    "http://abcappseitienst.azurewebsites.net/",
    "https://web-6j9l-findomestic.cnwsfert.com/-login",
    "https://allegro.12345.lat"
]

legit_urls = [
    "https://www.google.com", "https://github.com", "https://www.hdfcbank.com",
    "https://stackoverflow.com", "https://www.wikipedia.org", "https://vercel.com",
    "https://create-react-app.dev", "https://legit-startup.team", 
    "https://my-portfolio-123.netlify.app", "https://consulting.pro"
]

blocked_domains_cache = set()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_suspicious_tld(url_clean):
    ext = tldextract.extract(url_clean)
    return ext.suffix.split('.')[-1] in SUSPICIOUS_TLDS if ext.suffix else False

def is_free_hosting(url_clean):
    return any(host in url_clean.lower() for host in FREE_HOSTING)

def get_domain(url_clean):
    ext = tldextract.extract(url_clean)
    return f"{ext.domain}.{ext.suffix}".lower()

def is_dga_domain(url_clean):
    domain = tldextract.extract(url_clean).domain.lower()
    if len(domain) > 6:
        vowels = sum(1 for c in domain if c in 'aeiouy')
        if vowels == 0: return True
    return False

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

    sensitive = ['login', 'verify', 'secure', 'update', 'confirm', 'password', 'credit', 'signin', 'account', 'validate']
    shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 'shorturl', 'is.gd', 'ow.ly']

    return {
        'url_length': len(clean), 'domain_length': len(domain_info.domain) if domain_info.domain else 0,
        'subdomain_length': len(domain_info.subdomain) if domain_info.subdomain else 0, 'path_length': len(parsed.path),
        'num_dots': clean.count('.'), 'num_hyphens': clean.count('-'), 'num_underscores': clean.count('_'),
        'num_slashes': clean.count('/'), 'num_digits': sum(c.isdigit() for c in clean),
        'num_special_chars': sum(not c.isalnum() and c not in ['/', '.', ':', '-', '_'] for c in clean),
        'num_sensitive_words': sum(1 for w in sensitive if w in clean.lower()), 'has_@': 1 if '@' in clean else 0,
        'has_https': has_https, 'has_ip': 1 if re.search(r'\d+\.\d+\.\d+\.\d+', clean) else 0,
        'has_suspicious_tld': 1 if get_suspicious_tld(clean) else 0, 'is_shortened': 1 if any(s in clean for s in shorteners) else 0,
    }

# ============================================================
# LOAD RF MODEL
# ============================================================
print("\n📦 Loading Random Forest model...")
rf = joblib.load('models/random_forest_final_v3.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_cols = ['url_length', 'domain_length', 'subdomain_length', 'path_length', 'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes', 'num_digits', 'num_special_chars', 'num_sensitive_words', 'has_@', 'has_https', 'has_ip', 'has_suspicious_tld', 'is_shortened']

# ============================================================
# CBA RULES
# ============================================================
def apply_cba(url):
    triggered = []
    if is_dga_domain(url): triggered.append(('DGA Entropy', 0.80))
    if get_suspicious_tld(url): triggered.append(('Suspicious TLD', 0.85))
    if re.search(r'\d+\.\d+\.\d+\.\d+', url): triggered.append(('IP Address', 0.95))
    if '@' in url: triggered.append(('@ Symbol', 0.90))
    shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 'shorturl', 'is.gd']
    if any(s in url for s in shorteners): triggered.append(('Shortened URL', 0.70))
    return (True, sum(c for _, c in triggered) / len(triggered), [e for e, _ in triggered]) if triggered else (False, 0, [])

# ============================================================
# BS4 CONTENT
# ============================================================
def host_matches_official(host):
    host = host.lower().replace('www.', '')
    return any(host == od or host.endswith('.' + od) for od in OFFICIAL_DOMAINS)

def analyze_page_content(url):
    signals, score = [], 0.0
    if not url.startswith(('http://', 'https://')): url = 'http://' + url
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, timeout=5, headers=headers, allow_redirects=True)
        html, final_url = response.text, response.url
    except Exception as e:
        return 0.0, [f"fetch_failed"], "failed"

    if not html or len(html.strip()) < 100: return 0.0, ["empty_page"], "failed"
    try:
        soup = BeautifulSoup(html, 'html.parser')
    except: return 0.0, ["parse_failed"], "failed"

    parsed = urlparse(final_url)
    page_domain = parsed.netloc.lower().replace('www.', '')
    is_official = host_matches_official(page_domain)

    title = soup.title.string.lower() if soup.title and soup.title.string else ''
    for brand in BRAND_KEYWORDS:
        if re.search(rf'\b{re.escape(brand)}\b', title) and brand not in page_domain and not is_official:
            score += 0.20; signals.append(f"title_brand_mismatch"); break
    if soup.find_all('input', {'type': 'password'}) and not is_official:
        score += 0.30; signals.append("password_field")
    if any(k in html.lower() for k in ['card_number', 'cardnumber', 'cvv']) and not is_official:
        score += 0.30; signals.append("credit_card_field")
    for form in soup.find_all('form'):
        action = form.get('action', '')
        if action and not action.lower().startswith(('javascript:', '#', 'mailto:')):
            action_domain = urlparse(urljoin(final_url, action)).netloc.lower().replace('www.', '')
            if action_domain and action_domain != page_domain and not host_matches_official(action_domain):
                score += 0.40; signals.append("form_action_external"); break
    text_content = soup.get_text(separator=' ').lower()
    if sum(1 for kw in URGENCY_KEYWORDS if kw in text_content) >= 2:
        score += 0.15; signals.append("urgency_keywords")
    favicon = soup.find('link', rel=re.compile(r'icon', re.I))
    if favicon and favicon.get('href'):
        fav_domain = urlparse(urljoin(final_url, favicon['href'])).netloc.lower().replace('www.', '')
        if fav_domain and fav_domain != page_domain and not host_matches_official(fav_domain):
            score += 0.15; signals.append("favicon_external")
    for script in soup.find_all('script', src=False):
        if script.string:
            m = re.search(r'(?:window|document)\.location(?:\.href)?\s*=\s*[\'"]([^\'"]+)[\'"]', script.string)
            if m:
                target_domain = urlparse(urljoin(final_url, m.group(1))).netloc.lower().replace('www.', '')
                if target_domain and target_domain != page_domain and not host_matches_official(target_domain):
                    score += 0.20; signals.append("js_redirect_external"); break
    if soup.find('meta', attrs={'http-equiv': re.compile(r'refresh', re.I)}):
        score += 0.20; signals.append("meta_refresh")
    if len(soup.find_all('input', {'type': 'hidden'})) > 8:
        score += 0.10; signals.append("many_hidden_fields")
    if len(soup.find_all('iframe')) > 6 and not is_official:
        score += 0.05; signals.append("many_iframes")
    return min(score, 1.0), signals, "success"

# ============================================================
# GROQ LLM
# ============================================================
def call_groq_llm(url, prob, content_signals, fetch_status="success"):
    path_hits = [p for p in SUSPICIOUS_PATH_PATTERNS if re.search(p, url, re.I)]
    free_host = is_free_hosting(url)
    
    if fetch_status == "failed":
        evidence_note = "The page content could NOT be fetched or returned a blank page. This strongly indicates cloaking (hiding from bots)."
    else:
        evidence_note = "Page content was fetched."
        
    free_note = "\nHosted on a FREE HOSTING platform." if free_host else ""
    path_note = f"\nSuspicious path patterns: {path_hits}" if path_hits else ""
    signals_text = ", ".join(content_signals[:5]) if content_signals else "none"

    prompt = f"""Cybersecurity analysis:
URL: {url}
ML confidence: {prob:.1%}
Fetch status: {fetch_status}
{evidence_note}{free_note}
Signals: {signals_text}{path_note}

Default toward PHISHING if fetch failed or cloaked. Only call it legitimate if there is concrete evidence it is an established site.

Reply EXACTLY:
Verdict: YES or NO
Confidence: XX%"""

    try:
        response = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=GROQ_MODEL, temperature=0.1, max_tokens=500)
        text = response.choices[0].message.content or ""
        verdict = 'phishing' if re.search(r'Verdict:\s*\**\s*YES', text, re.I) else 'legitimate'
        conf = int(re.search(r'Confidence:\s*\**\s*(\d+)', text).group(1)) / 100 if re.search(r'Confidence:\s*\**\s*(\d+)', text) else 0.5
        return verdict, conf
    except: return 'phishing' if prob >= 0.45 else 'legitimate', 0.5

# ============================================================
# RUN TEST
# ============================================================
def run_test(urls, label):
    print(f"\n" + "="*70)
    print(f"TESTING {label.upper()} ({len(urls)} URLs)")
    print("="*70)
    
    correct = 0
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] URL: {url}")
        domain = get_domain(url)
        
        if domain in blocked_domains_cache:
            print(f"  🚨 PHISHING (Layer 1.5: Domain Memory) | Domain already flagged!")
            if label == "phishing": correct += 1
            continue
            
        # Layer 1: CBA
        is_phish, cba_conf, cba_signals = apply_cba(url)
        if is_phish and cba_conf >= 0.70:
            print(f"  🚨 PHISHING (Layer 1: CBA) | Conf: {cba_conf:.2f} | Signals: {cba_signals}")
            blocked_domains_cache.add(domain)
            if label == "phishing": correct += 1
            continue
            
        # Layer 2: RF
        features = pd.DataFrame([extract_url_features(url)])[feature_cols]
        X_scaled = scaler.transform(features)
        y_proba = rf.predict_proba(X_scaled)[0][1]
        
        if y_proba >= 0.65:
            print(f"  🚨 PHISHING (Layer 2: RF) | Prob: {y_proba:.2f}")
            blocked_domains_cache.add(domain)
            if label == "phishing": correct += 1
            continue
        elif y_proba < 0.45:
            print(f"  ✅ LEGITIMATE (Layer 2: RF) | Prob: {y_proba:.2f}")
            if label == "legitimate": correct += 1
            continue
            
        print(f"  ⚠️ SUSPICIOUS (Layer 2: RF) | Prob: {y_proba:.2f} -> Checking Content...")
        
        # Layer 3: Content
        score, signals, status = analyze_page_content(url)
        
        # CLOAKING FIX: If it's free hosting and blank, it's phishing
        if status == "failed" and is_free_hosting(url):
            print(f"  🚨 PHISHING (Layer 3: Free Hosting Cloaking) | Blank page on free host")
            blocked_domains_cache.add(domain)
            if label == "phishing": correct += 1
            continue

        if status == "failed":
            if y_proba >= 0.50:
                print(f"  🚨 PHISHING (Layer 3: Fetch Failed) | High RF Prob")
                blocked_domains_cache.add(domain)
                if label == "phishing": correct += 1
                continue
            else:
                print(f"  ❌ Fetch Failed -> LLM")
                v, c = call_groq_llm(url, y_proba, signals, status)
                print(f"     -> LLM Verdict: {v.upper()} ({c:.0%})")
                if v == 'phishing': blocked_domains_cache.add(domain)
                if label == "phishing" and v == 'phishing': correct += 1
                if label == "legitimate" and v == 'legitimate': correct += 1
                continue
                
        if score >= 0.65:
            print(f"  🚨 PHISHING (Layer 3: Content) | Score: {score:.2f} | Signals: {signals}")
            blocked_domains_cache.add(domain)
            if label == "phishing": correct += 1
            continue
            
        # ESCALATION FIX: Lowered cutoff to 0.45
        elif score < 0.30 and y_proba < 0.45:
            print(f"  ✅ LEGITIMATE (Layer 3: Content) | Score: {score:.2f}")
            if label == "legitimate": correct += 1
            continue
            
        print(f"  ❓ Content Clean but High RF -> LLM | Score: {score:.2f}")
        v, c = call_groq_llm(url, y_proba, signals, status)
        print(f"     -> LLM Verdict: {v.upper()} ({c:.0%})")
        if v == 'phishing': blocked_domains_cache.add(domain)
        if label == "phishing" and v == 'phishing': correct += 1
        if label == "legitimate" and v == 'legitimate': correct += 1

    print(f"\n📊 {label.upper()} ACCURACY: {correct}/{len(urls)} ({correct/len(urls)*100:.1f}%)")

# Run both tests
run_test(phishing_urls, "phishing")
blocked_domains_cache.clear()
run_test(legit_urls, "legitimate")