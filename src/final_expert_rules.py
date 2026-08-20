# final_expert_rules.py (FIXED VERSION)
"""
Expert-Defined CBA Rules - FIXED
- Proper TLD matching (domain suffix, not substring)
- Only genuinely abused TLDs (free/near-free)
- Domain-only keyword checks
"""

import re
import joblib
from urllib.parse import urlparse

print("="*60)
print("COMPREHENSIVE EXPERT RULES - FIXED VERSION")
print("="*60)

# ===================== ONLY GENUINELY ABUSED TLDS =====================
# NOT: .in, .co, .io, .me, .uk, .de, etc. (these are legitimate ccTLDs)
# ONLY: free/near-free TLDs with no verification requirements

SUSPICIOUS_TLDS = [
    # Freenom TLDs (completely free, heavily abused)
    '.tk', '.ml', '.ga', '.cf', '.gq',
    
    # Cheap/Free gTLDs with high abuse rates
    '.xyz', '.top', '.click', '.link', '.download',
    '.review', '.stream', '.loan', '.work', '.men',
    '.party', '.science', '.racing', '.bid', '.win',
    '.faith', '.accountant', '.date', '.trade', '.webcam',
    '.zip', '.mov', '.rest', '.sbs', '.aaa', '.wang',
    '.online', '.live', '.site', '.tech', '.store',
    '.shop', '.space', '.pro', '.club', '.biz',
]

# ===================== HELPER FUNCTIONS =====================

def get_domain_from_url(url):
    """Extract clean domain from URL"""
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        parsed = urlparse(url)
        domain = parsed.netloc.lower().split(':')[0]
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return ''

def get_tld_from_domain(domain):
    """Extract TLD from domain"""
    parts = domain.split('.')
    if len(parts) >= 2:
        tld = '.' + parts[-1]
        # Handle .co.in, .com.au, etc.
        if len(parts) >= 3 and parts[-2] in ['co', 'com', 'org', 'net', 'gov', 'edu', 'ac']:
            tld = '.' + parts[-2] + '.' + parts[-1]
        return tld
    return domain

def has_suspicious_tld(domain):
    """Check if domain ends with any suspicious TLD"""
    return any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS)

# ===================== CONFIGURATION =====================

OFFICIAL_DOMAINS = [
    # Social Media
    'google.com', 'youtube.com', 'facebook.com', 'twitter.com',
    'instagram.com', 'linkedin.com', 'reddit.com', 'github.com',
    'tiktok.com', 'snapchat.com', 'pinterest.com', 'whatsapp.com',
    'telegram.org', 'discord.com', 'twitch.tv',
    
    # E-commerce
    'amazon.com', 'amazon.in', 'ebay.com', 'flipkart.com',
    'walmart.com', 'target.com', 'shopify.com', 'etsy.com',
    'alibaba.com', 'aliexpress.com', 'bestbuy.com',
    
    # Banking (International)
    'paypal.com', 'bankofamerica.com', 'chase.com', 'wellsfargo.com',
    'capitalone.com', 'citibank.com', 'hsbc.com', 'barclays.com',
    
    # Banking (Brazil)
    'banestes.com.br', 'banestes.com', 'bradesco.com.br',
    'itau.com.br', 'santander.com.br', 'bb.com.br', 'caixa.gov.br',
    
    # Banking (India)
    'hdfcbank.com', 'icicibank.com', 'axisbank.com', 'sbi.co.in',
    'kotak.com', 'yesbank.in', 'idfcfirstbank.com', 'aufin.com',
    'pnbindia.in', 'canarabank.com', 'indianbank.in',
    'bankofbaroda.in', 'unionbankofindia.co.in',
    
    # Tech
    'microsoft.com', 'apple.com', 'ibm.com', 'oracle.com',
    'salesforce.com', 'adobe.com', 'intel.com', 'nvidia.com',
    'amd.com', 'cisco.com', 'dell.com', 'hp.com',
    
    # News
    'cnn.com', 'bbc.com', 'reuters.com', 'bloomberg.com',
    'nytimes.com', 'wsj.com', 'economictimes.com',
    
    # Other
    'yahoo.com', 'bing.com', 'duckduckgo.com', 'outlook.com',
    'office.com', 'protonmail.com', 'stackoverflow.com',
    'medium.com', 'wikipedia.org', 'github.io',
]

SENSITIVE_WORDS = [
    'login', 'verify', 'secure', 'update', 'confirm',
    'password', 'account', 'signin', 'validate',
    'authenticate', 'verification', 'security', 'captcha',
    'billing', 'payment', 'transaction', 'activate',
    'recover', 'reset', 'unlock', 'access',
]

BRANDS = [
    # Social Media
    'paypal', 'amazon', 'apple', 'google', 'microsoft',
    'facebook', 'github', 'twitter', 'instagram', 'linkedin',
    'netflix', 'spotify', 'tiktok', 'snapchat', 'discord',
    
    # Banks International
    'bankofamerica', 'chase', 'wellsfargo', 'capitalone',
    'citibank', 'hsbc', 'barclays', 'jpmorgan',
    
    # Banks Brazil
    'banestes', 'bradesco', 'itau', 'santander', 'caixa',
    'sicoob', 'sicredi', 'inter', 'c6bank', 'nubank',
    
    # Banks India
    'hdfc', 'icici', 'sbi', 'axis', 'kotak', 'yesbank',
    'idfc', 'pnb', 'canara', 'indianbank', 'iob',
    'bankofbaroda', 'unionbank', 'aufin',
]

SHORTENERS = [
    'bit.ly', 'tinyurl', 'goo.gl', 'shorturl', 'is.gd',
    'ow.ly', 'buff.ly', 'rebrand.ly', 'tiny.cc', 't.co',
    'lnkd.in', 'tiny.one', 'shorte.st',
]

# ===================== RULE CHECKING FUNCTION =====================

def check_phishing_rules_fixed(url):
    """
    Check expert-defined phishing rules - FIXED VERSION
    - Proper TLD matching on domain suffix
    - Keywords checked against DOMAIN only, not path/query
    - No false positives for .in, .co, .io, etc.
    """
    url_lower = url.lower()
    triggered = []
    explanations = []
    
    # Extract domain
    domain = get_domain_from_url(url)
    if not domain:
        return False, 0, [], ['Unable to parse URL']
    
    # Extract TLD properly
    tld = get_tld_from_domain(domain)
    
    # Check official domain (WHITELIST)
    is_official = False
    for official in OFFICIAL_DOMAINS:
        if domain.endswith(official):
            is_official = True
            explanations.append(f"Official domain: {official}")
            break
    
    if is_official:
        return False, 0.99, [], explanations
    
    # Rule 1: Suspicious TLD (ONLY on the actual TLD)
    if has_suspicious_tld(domain):
        triggered.append(('Suspicious TLD', 0.85))
        explanations.append(f"Suspicious TLD: {tld}")
    
    # Rule 2: IP Address
    if re.search(r'\d+\.\d+\.\d+\.\d+', url):
        triggered.append(('IP address used', 0.95))
        explanations.append("IP address instead of domain")
    
    # Rule 3: @ symbol
    if '@' in url:
        triggered.append(('@ symbol in URL', 0.90))
        explanations.append("@ symbol in URL (credential phishing)")
    
    # Rule 4: Shortened URL
    if any(s in url_lower for s in SHORTENERS):
        triggered.append(('Shortened URL', 0.70))
        explanations.append("Shortened URL hides destination")
    
    # Rule 5: Brand misuse (ONLY if brand in DOMAIN, not path)
    has_brand_in_domain = any(brand in domain for brand in BRANDS)
    has_sensitive_in_domain = any(word in domain for word in SENSITIVE_WORDS)
    
    if has_brand_in_domain and has_sensitive_in_domain:
        # Check if official domain
        is_brand_com = any(f'{brand}.com' in domain for brand in BRANDS)
        is_brand_in = any(f'{brand}.in' in domain or f'{brand}.co.in' in domain for brand in BRANDS)
        is_brand_br = any(f'{brand}.com.br' in domain for brand in BRANDS)
        
        if not is_brand_com and not is_brand_in and not is_brand_br:
            triggered.append(('Brand misuse', 0.88))
            explanations.append("Brand name used with sensitive keywords in domain")
    
    # Rule 6: Suspicious subdomain
    parts = domain.split('.')
    subdomain = parts[0] if len(parts) > 2 else ''
    
    if len(subdomain) > 10 and (re.search(r'\d{3,}', subdomain) or re.search(r'[a-z]\d{5,}', subdomain)):
        triggered.append(('Random subdomain', 0.75))
        explanations.append(f"Suspicious random subdomain: {subdomain}")
    
    # Rule 7: Multiple subdomains
    dot_count = domain.count('.')
    if dot_count > 3:
        triggered.append(('Multiple subdomains', 0.75))
        explanations.append(f"Multiple subdomains ({dot_count} dots)")
    
    # Rule 8: HTTP for sensitive pages (path-based, so keep)
    has_sensitive_in_path = any(word in url_lower for word in SENSITIVE_WORDS)
    if url.startswith('http://') and has_sensitive_in_path:
        triggered.append(('HTTP for sensitive page', 0.70))
        explanations.append("HTTP (insecure) for login/secure page")
    
    # Calculate confidence
    if triggered:
        confidence = sum([conf for _, conf in triggered]) / len(triggered)
        confidence = min(confidence, 0.95)
        return True, confidence, triggered, explanations
    else:
        return False, 0, [], explanations

# ===================== TEST =====================

print("\n=== TESTING FIXED RULES ===\n")

test_urls = [
    # Legitimate (should NOT be flagged)
    "https://phishguard.co.in",
    "https://www.google.co.in",
    "https://www.amazon.in",
    "https://www.hdfcbank.com",
    "https://www.sbi.co.in",
    "https://www.icicibank.com",
    "https://www.yesbank.in",
    "https://www.flipkart.com",
    "https://www.github.com",
    "https://stackoverflow.com",
    
    # Phishing (should be flagged)
    "https://hdfc-secure-login.xyz",
    "https://paypai-secure-verify.xyz",
    "https://amaz0n-verify.account-security.top",
    "http://bit.ly/3xYZ123",
    "http://192.168.1.1/login",
    "https://secure-login-paypal.xyz",
]

print("-" * 70)
for url in test_urls:
    is_phishing, confidence, triggered, explanations = check_phishing_rules_fixed(url)
    status = "⚠️ PHISHING" if is_phishing else "✅ LEGITIMATE"
    print(f"{status} ({confidence:.1%}) - {url}")
    if explanations:
        for exp in explanations[:2]:  # Show top 2 reasons
            print(f"  → {exp}")
    print()

# ===================== SAVE =====================

rules_config = {
    'official_domains': OFFICIAL_DOMAINS,
    'suspicious_tlds': SUSPICIOUS_TLDS,
    'sensitive_words': SENSITIVE_WORDS,
    'brands': BRANDS,
    'shorteners': SHORTENERS
}

joblib.dump(rules_config, 'models/cba_expert_rules_final.pkl')
print("\n✅ Fixed expert rules saved to 'models/cba_expert_rules_final.pkl'")
print(f"   - Suspicious TLDs: {len(SUSPICIOUS_TLDS)} (ONLY free/abused)")
print(f"   - Official Domains: {len(OFFICIAL_DOMAINS)}")
print(f"   - Sensitive Words: {len(SENSITIVE_WORDS)}")
print(f"   - Brands: {len(BRANDS)}")