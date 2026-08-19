# final_expert_rules.py (COMPLETE VERSION)
"""
Expert-Defined CBA Rules - COMPREHENSIVE LIST
Includes ALL known suspicious TLDs, domains, and patterns
"""

import re
import joblib

print("="*60)
print("COMPREHENSIVE EXPERT RULES - ALL SUSPICIOUS PATTERNS")
print("="*60)

# ===================== COMPLETE SUSPICIOUS TLDS =====================

# ALL known suspicious TLDs (200+)
SUSPICIOUS_TLDS = [
    # Free/Cheap TLDs (most abused)
    '.tk', '.ml', '.ga', '.cf', '.gq',  # Freenom TLDs
    '.xyz', '.top', '.club', '.work', '.info', '.biz',
    '.online', '.live', '.stream', '.dates', '.download',
    '.review', '.click', '.link', '.site', '.tech',
    '.store', '.shop', '.space', '.pro', '.app',
    
    # New gTLDs (often abused)
    '.agency', '.bar', '.boutique', '.buzz', '.ceo',
    '.center', '.coach', '.company', '.cool', '.credit',
    '.digital', '.direct', '.events', '.expert', '.express',
    '.financial', '.forsale', '.free', '.global', '.guru',
    '.help', '.host', '.international', '.investments', '.io',
    '.legal', '.life', '.ltd', '.market', '.marketing',
    '.media', '.mobile', '.network', '.news', '.ninja',
    '.page', '.partners', '.photo', '.photos', '.pics',
    '.press', '.promo', '.purchase', '.rocks', '.run',
    '.services', '.solutions', '.support', '.systems', '.team',
    '.technology', '.today', '.tools', '.training', '.uno',
    '.ventures', '.vip', '.vision', '.website', '.works',
    
    # Country TLDs often abused
    '.am', '.by', '.cc', '.cm', '.cn', '.co', '.cz',
    '.ee', '.fm', '.gd', '.gs', '.hn', '.ht', '.in',
    '.io', '.ir', '.kh', '.ki', '.li', '.lr', '.lt',
    '.lv', '.ma', '.md', '.me', '.mg', '.mk', '.ml',
    '.mn', '.ms', '.mt', '.mu', '.mv', '.mw', '.mx',
    '.my', '.mz', '.na', '.nf', '.ng', '.ni', '.nl',
    '.no', '.np', '.nr', '.nu', '.nz', '.om', '.pa',
    '.pe', '.pf', '.pg', '.ph', '.pk', '.pl', '.pm',
    '.pn', '.pr', '.ps', '.pt', '.pw', '.py', '.qa',
    '.re', '.ro', '.rs', '.ru', '.rw', '.sa', '.sb',
    '.sc', '.sd', '.se', '.sg', '.sh', '.si', '.sk',
    '.sl', '.sm', '.sn', '.so', '.sr', '.st', '.su',
    '.sv', '.sx', '.sy', '.sz', '.tc', '.td', '.tf',
    '.tg', '.th', '.tj', '.tk', '.tl', '.tm', '.tn',
    '.to', '.tr', '.tt', '.tv', '.tw', '.tz', '.ua',
    '.ug', '.uk', '.us', '.uy', '.uz', '.va', '.vc',
    '.ve', '.vg', '.vi', '.vn', '.vu', '.ws', '.ye',
    '.yt', '.za', '.zm', '.zw',
    
    # New popular abuse TLDs
    '.click', '.rest', '.xyz', '.top', '.loan', '.men',
    '.win', '.bid', '.date', '.download', '.review',
    '.trade', '.webcam', '.science', '.party', '.racing',
    '.accountant', '.faith', '.wang', '.sbs', '.aaa',
    '.zip', '.mov', '.realestate', '.spa', '.song',
]

# ===== SENSITIVE WORDS (Global) =====
sensitive_words = [
    'login', 'verify', 'secure', 'update', 'confirm',
    'password', 'credit', 'signin', 'account', 'validate',
    'authenticate', 'verification', 'security', 'captcha',
    'authentication', 'billing', 'payment', 'transaction',
    'alert', 'notice', 'warning', 'suspended', 'limited',
    'restricted', 'locked', 'frozen', 'unauthorized',
    'activate', 'recover', 'reset', 'unlock', 'access',
]

# ===== BRANDS (Global) =====
brands = [
    'paypal', 'amazon', 'apple', 'google', 'microsoft',
    'facebook', 'github', 'twitter', 'instagram', 'linkedin',
    'netflix', 'spotify', 'tiktok', 'snapchat', 'pinterest',
    'whatsapp', 'telegram', 'discord', 'reddit',
    'bankofamerica', 'chase', 'wellsfargo', 'capitalone',
    'citibank', 'hsbc', 'barclays', 'jpmorgan',
    'banestes', 'bradesco', 'itau', 'santander', 'caixa', 'bb',
    'sicoob', 'sicredi', 'inter', 'c6bank', 'nubank',
    'hdfc', 'icici', 'sbi', 'axis', 'kotak', 'yesbank',
    'idfc', 'pnb', 'canara', 'indianbank', 'iob',
    'bankofbaroda', 'unionbank', 'aufin',
    'binance', 'coinbase', 'kraken', 'blockchain', 'metamask',
]

# ===== SHORTENERS (Global) =====
shorteners = [
    'bit.ly', 'tinyurl', 'goo.gl', 'shorturl', 'is.gd',
    'ow.ly', 'buff.ly', 'rebrand.ly', 'tiny.cc', 't.co',
    'lnkd.in', 'tiny.one', 'shorte.st', 'cli.gs', 'j.mp',
]

# ===================== SUSPICIOUS DOMAINS (Patterns) =====================

SUSPICIOUS_DOMAIN_PATTERNS = [
    # Random alphanumeric domains
    r'[a-z0-9]{15,}\.',
    r'[a-z]{10,}[0-9]{5,}\.',
    r'[0-9]{6,}[a-z]{6,}\.',
    
    # Typosquatting patterns
    'rnicrosoft.com', 'ggoogle.com', 'faccbook.com',
    'amazoon.com', 'paypall.com', 'facebok.com',
    
    # Common phishing keywords
    'verify', 'secure', 'login', 'signin', 'account',
    'update', 'confirm', 'validate', 'authenticate',
    'security', 'billing', 'payment', 'transaction',
    'alert', 'notice', 'warning', 'suspended',
    'limited', 'restricted', 'locked', 'frozen',
    'unauthorized', 'suspicious', 'activity',
]

# ===================== OFFICIAL DOMAINS (WHITELIST) =====================

OFFICIAL_DOMAINS = [
    # Social Media
    'google.com', 'facebook.com', 'twitter.com', 'instagram.com',
    'linkedin.com', 'youtube.com', 'reddit.com', 'github.com',
    'tiktok.com', 'snapchat.com', 'pinterest.com', 'whatsapp.com',
    
    # E-commerce
    'amazon.com', 'ebay.com', 'flipkart.com', 'walmart.com',
    'target.com', 'shopify.com', 'etsy.com', 'alibaba.com',
    'aliexpress.com', 'bestbuy.com', 'homedepot.com',
    
    # Banking (International)
    'paypal.com', 'bankofamerica.com', 'chase.com', 'wellsfargo.com',
    'capitalone.com', 'citibank.com', 'hsbc.com', 'barclays.com',
    'jpmorgan.com', 'goldmansachs.com', 'morganstanley.com',
    
    # Banking (Brazil)
    'banestes.com.br', 'banestes.com', 'bradesco.com.br', 'itau.com.br',
    'santander.com.br', 'bb.com.br', 'caixa.gov.br', 'sicoob.com.br',
    'sicredi.com.br', 'inter.co', 'c6bank.com',
    
    # Banking (India)
    'hdfcbank.com', 'icicibank.com', 'axisbank.com', 'sbi.co.in',
    'kotak.com', 'yesbank.in', 'idfcfirstbank.com', 'aufin.com',
    'pnbindia.in', 'canarabank.com', 'indianbank.in', 'iob.in',
    'bankofbaroda.in', 'unionbankofindia.co.in',
    
    # Banking (Other)
    'anz.com.au', 'commonwealth.com.au', 'westpac.com.au',
    'nab.com.au', 'rbc.com', 'td.com', 'bmo.com', 'scotiabank.com',
    
    # Tech
    'microsoft.com', 'apple.com', 'ibm.com', 'oracle.com',
    'salesforce.com', 'adobe.com', 'intel.com', 'nvidia.com',
    'amd.com', 'cisco.com', 'vmware.com', 'dell.com',
    'hp.com', 'lenovo.com', 'acer.com', 'asus.com',
    
    # News
    'cnn.com', 'bbc.com', 'reuters.com', 'bloomberg.com',
    'nytimes.com', 'wsj.com', 'economictimes.com', 'foxnews.com',
    'nbcnews.com', 'theguardian.com', 'washingtonpost.com',
    
    # Government
    '.gov', '.mil', '.edu', '.ac.uk', '.edu.au',
    '.gov.br', '.gov.in', '.mod.uk', '.gouv.fr',
    
    # Additional common domains
    'yahoo.com', 'bing.com', 'duckduckgo.com', 'protonmail.com',
    'gmail.com', 'outlook.com', 'office.com', 'drive.google.com',
]

def check_phishing_rules_complete(url):
    """
    Check expert-defined phishing rules - COMPLETE VERSION
    """
    url_lower = url.lower()
    triggered = []
    explanations = []
    
    # Clean URL
    clean_url = re.sub(r'^https?://', '', url_lower)
    clean_url = re.sub(r'^www\.', '', clean_url)
    
    # Check official domain (WHITELIST)
    is_official = False
    for domain in OFFICIAL_DOMAINS:
        if clean_url.startswith(domain) or f'.{domain}' in clean_url:
            is_official = True
            explanations.append(f"Official domain: {domain}")
            break
    
    # Check if it's a government/edu domain
    if '.gov' in clean_url or '.edu' in clean_url:
        is_official = True
        explanations.append("Official .gov or .edu domain")
    
    # If official domain → LEGITIMATE immediately
    if is_official:
        return False, 0.99, [], explanations
    
    # Rule 1: Suspicious TLD
    suspicious_tld_found = False
    for tld in SUSPICIOUS_TLDS:
        if tld in url_lower:
            suspicious_tld_found = True
            triggered.append(('Suspicious TLD', 0.85))
            explanations.append(f"Suspicious TLD: {tld}")
            break
    
    # Rule 2: IP Address
    if re.search(r'\d+\.\d+\.\d+\.\d+', url):
        triggered.append(('IP address used', 0.95))
        explanations.append("IP address instead of domain")
    
    # Rule 3: @ symbol
    if '@' in url:
        triggered.append(('@ symbol in URL', 0.90))
        explanations.append("@ symbol in URL (credential phishing)")
    
    # Rule 4: Shortened URL
    shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 'shorturl', 'is.gd', 'ow.ly',
                  'buff.ly', 'rebrand.ly', 'tiny.cc', 't.co', 'lnkd.in']
    if any(s in url_lower for s in shorteners):
        triggered.append(('Shortened URL', 0.70))
        explanations.append("Shortened URL hides destination")
    
    # Rule 5: Brand misuse
    brands = ['paypal', 'amazon', 'apple', 'google', 'microsoft', 'facebook',
              'github', 'twitter', 'instagram', 'linkedin', 'netflix', 'spotify',
              'bankofamerica', 'chase', 'wellsfargo', 'capitalone', 'citibank',
              'banestes', 'bradesco', 'itau', 'santander', 'caixa', 'bb',
              'hdfc', 'icici', 'sbi', 'axis', 'kotak', 'yesbank', 'idfc',
              'hsbc', 'barclays', 'anz', 'commonwealth', 'westpac', 'nab']
    
    sensitive_words = ['login', 'verify', 'secure', 'update', 'confirm',
                       'password', 'credit', 'signin', 'account', 'validate',
                       'authenticate', 'verification', 'security', 'captcha',
                       'authentication', 'billing', 'payment', 'transaction',
                       'alert', 'notice', 'warning', 'suspended', 'limited',
                       'restricted', 'locked', 'frozen', 'unauthorized']
    
    has_sensitive = any(word in url_lower for word in sensitive_words)
    has_brand = any(brand in url_lower for brand in brands)
    
    if has_brand and has_sensitive:
        # Check if exact official domain
        is_brand_com = any(f'{brand}.com' in url_lower for brand in brands)
        is_brand_br = any(f'{brand}.com.br' in url_lower for brand in brands)
        is_brand_in = any(f'{brand}.in' in url_lower or f'{brand}.co.in' in url_lower for brand in brands)
        
        if not is_brand_com and not is_brand_br and not is_brand_in:
            triggered.append(('Brand misuse', 0.88))
            explanations.append("Brand name used with sensitive keywords")
    
    # Rule 6: Suspicious subdomain
    domain_part = clean_url.split('/')[0]
    subdomain = domain_part.split('.')[0] if '.' in domain_part else ''
    
    # Random subdomain check
    if len(subdomain) > 10 and (re.search(r'\d{3,}', subdomain) or re.search(r'[a-z]\d{5,}', subdomain)):
        triggered.append(('Random subdomain', 0.75))
        explanations.append(f"Suspicious random subdomain: {subdomain}")
    
    # Rule 7: Multiple subdomains
    dot_count = domain_part.count('.')
    if dot_count > 3:
        triggered.append(('Multiple subdomains', 0.75))
        explanations.append(f"Multiple subdomains ({dot_count} dots)")
    
    # Rule 8: HTTP for sensitive pages
    if url.startswith('http://') and has_sensitive:
        triggered.append(('HTTP for sensitive page', 0.70))
        explanations.append("HTTP (insecure) for login/secure page")
    
    # Rule 9: PHP script with sensitive words
    if '.php' in url_lower and has_sensitive:
        triggered.append(('PHP script with sensitive words', 0.75))
        explanations.append("PHP script combined with sensitive keywords")
    
    # Rule 10: Very long URL
    if len(url) > 100:
        triggered.append(('Very long URL', 0.65))
        explanations.append("Very long URL (possible obfuscation)")
    
    # Calculate confidence
    if triggered:
        confidence = sum([conf for _, conf in triggered]) / len(triggered)
        confidence = min(confidence, 0.95)
        return True, confidence, triggered, explanations
    else:
        return False, 0, [], explanations

# ===================== TEST =====================
test_urls = [
    # Legitimate
    "https://www.google.com",
    "https://www.amazon.com",
    "https://banestes.com.br",
    "https://www.hdfcbank.com",
    
    # Phishing
    "https://ouhari.direc4571512.pro/captcha.php",
    "http://banestes.instaladorpj.com",
    "https://secure-login.paypal-verify.xyz",
    "http://bit.ly/3xYZ123",
    "http://192.168.1.1/login",
    "https://amazon-verify.account-security.top",
]

print("\n=== TESTING COMPLETE RULES ===\n")
for url in test_urls:
    is_phishing, confidence, triggered, explanations = check_phishing_rules_complete(url)
    status = "⚠️ PHISHING" if is_phishing else "✅ LEGITIMATE"
    print(f"{status} ({confidence:.1%}) - {url}")
    if explanations:
        for exp in explanations[:3]:  # Show top 3 reasons
            print(f"  → {exp}")
    print()

# ===================== SAVE =====================
rules_config = {
    'official_domains': OFFICIAL_DOMAINS,
    'suspicious_tlds': SUSPICIOUS_TLDS,
    'sensitive_words': sensitive_words,
    'brands': brands,
    'shorteners': shorteners,
    'suspicious_domain_patterns': SUSPICIOUS_DOMAIN_PATTERNS
}

joblib.dump(rules_config, '../models/cba_expert_rules_final.pkl')
print("\n✅ Complete expert rules saved to 'models/cba_expert_rules_final.pkl'")
print(f"   - Suspicious TLDs: {len(SUSPICIOUS_TLDS)}")
print(f"   - Official Domains: {len(OFFICIAL_DOMAINS)}")
print(f"   - Sensitive Words: {len(sensitive_words)}")
print(f"   - Brands: {len(brands)}")
print(f"   - Shorteners: {len(shorteners)}")