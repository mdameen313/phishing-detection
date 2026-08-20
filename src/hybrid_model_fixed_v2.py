"""
Hybrid Phishing Detector
Layer 1: Expert CBA Rules (Fast)
Layer 2: Random Forest (Accurate)
"""

import pandas as pd
import joblib
import re
import tldextract
from urllib.parse import urlparse
import os

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
    suspicious_tlds = [
        '.tk', '.ml', '.ga', '.cf', '.gq',  # Freenom
        '.xyz', '.top', '.click', '.link', '.download',
        '.review', '.stream', '.loan', '.work', '.men',
        '.party', '.science', '.racing', '.bid', '.win',
        '.faith', '.accountant', '.date', '.trade', '.webcam',
        '.zip', '.mov', '.rest', '.sbs', '.online', '.live',
        '.site', '.tech', '.store', '.shop', '.space', '.pro',
        '.club', '.biz', '.info'
    ]
    return any(domain.endswith(tld) for tld in suspicious_tlds)

def has_keyword_in_domain(domain, keywords):
    """Check if ANY keyword is in the domain (not path/query)"""
    domain_lower = domain.lower()
    return any(keyword in domain_lower for keyword in keywords)

# ===================== HYBRID DETECTOR =====================

class HybridPhishingDetector:
    def __init__(self, model_dir='models/'):
        self.model_dir = model_dir
        
        try:
            # Load all components
            self.rules_config = joblib.load(f'{model_dir}cba_expert_rules_final.pkl')
            self.rf_model = joblib.load(f'{model_dir}random_forest_final_v3.pkl')
            self.scaler = joblib.load(f'{model_dir}scaler.pkl')
            
            # Load feature names
            train_df = pd.read_csv('processed_data/train_enhanced.csv')
            self.feature_cols = [col for col in train_df.columns if col != 'label']
            
            print("✅ Models loaded successfully!")
            print(f"✅ {len(self.feature_cols)} features loaded")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
        
        # Rule configs
        self.suspicious_tlds = self.rules_config.get('suspicious_tlds', [])
        self.sensitive_words = self.rules_config.get('sensitive_words', [])
        self.brands = self.rules_config.get('brands', [])
        self.shorteners = self.rules_config.get('shorteners', [])
        self.official_domains = self.rules_config.get('official_domains', [])
    
    def check_expert_rules(self, url):
        """
        Check expert-defined CBA rules - FIXED VERSION
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
        for official in self.official_domains:
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
            explanations.append("@ symbol in URL")
        
        # Rule 4: Shortened URL
        if any(s in url_lower for s in self.shorteners):
            triggered.append(('Shortened URL', 0.70))
            explanations.append("Shortened URL")
        
        # Rule 5: Brand misuse (ONLY if brand in DOMAIN, not path)
        has_brand_in_domain = any(brand in domain for brand in self.brands)
        has_sensitive_in_domain = any(word in domain for word in self.sensitive_words)
        
        if has_brand_in_domain and has_sensitive_in_domain:
            # Check if official domain
            is_brand_com = any(f'{brand}.com' in domain for brand in self.brands)
            is_brand_in = any(f'{brand}.in' in domain or f'{brand}.co.in' in domain for brand in self.brands)
            is_brand_br = any(f'{brand}.com.br' in domain for brand in self.brands)
            
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
        has_sensitive_in_path = any(word in url_lower for word in self.sensitive_words)
        if url.startswith('http://') and has_sensitive_in_path:
            triggered.append(('HTTP for sensitive page', 0.70))
            explanations.append("HTTP for login/secure page")
        
        # Calculate confidence
        if triggered:
            confidence = sum([conf for _, conf in triggered]) / len(triggered)
            confidence = min(confidence, 0.95)
            return True, confidence, triggered, explanations
        else:
            return False, 0, [], explanations
    
    def extract_features(self, url):
        """Extract 16 features from URL"""
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
            'num_sensitive_words': sum(1 for w in self.sensitive_words if w in clean.lower()),
            'has_@': 1 if '@' in clean else 0,
            'has_https': has_https,
            'has_ip': 1 if re.search(r'\d+\.\d+\.\d+\.\d+', clean) else 0,
            'has_suspicious_tld': 1 if has_suspicious_tld(clean) else 0,
            'is_shortened': 1 if any(s in clean for s in self.shorteners) else 0,
        }
        return features
    
    def predict(self, url):
        """Hybrid prediction"""
        is_phishing, cba_confidence, triggered_rules, explanations = self.check_expert_rules(url)
        
        # Layer 1: CBA Rules
        if is_phishing and cba_confidence >= 0.70:
            return {
                'verdict': 'phishing',
                'confidence': cba_confidence,
                'layer': 'CBA_Rules',
                'explanation': explanations[0] if explanations else 'Multiple rules',
                'details': explanations
            }
        
        # Official domain whitelist
        if explanations and 'Official domain' in explanations[0]:
            return {
                'verdict': 'legitimate',
                'confidence': 0.99,
                'layer': 'Whitelist',
                'explanation': 'Official domain',
                'details': explanations
            }
        
        # No rules triggered → likely legitimate
        if not is_phishing and not triggered_rules:
            return {
                'verdict': 'legitimate',
                'confidence': 0.85,
                'layer': 'No_Rules',
                'explanation': 'No suspicious patterns',
                'details': explanations
            }
        
        # Layer 2: Random Forest
        features_dict = self.extract_features(url)
        features_df = pd.DataFrame([features_dict])
        features_df = features_df[self.feature_cols]
        
        features_scaled = self.scaler.transform(features_df)
        features_scaled = pd.DataFrame(features_scaled, columns=self.feature_cols)
        
        prob = self.rf_model.predict_proba(features_scaled)[0][1]
        pred = 1 if prob >= 0.5 else 0
        
        if prob >= 0.85:
            return {
                'verdict': 'phishing' if pred == 1 else 'legitimate',
                'confidence': prob if pred == 1 else (1 - prob),
                'layer': 'RandomForest',
                'explanation': f'RF: {prob:.1%} phishing confidence',
                'details': explanations
            }
        else:
            return {
                'verdict': 'legitimate',
                'confidence': 1 - prob,
                'layer': 'Default',
                'explanation': f'Low confidence ({prob:.1%}) - classified as legitimate',
                'details': explanations
            }

# ===================== TEST =====================
if __name__ == "__main__":
    detector = HybridPhishingDetector()
    
    test_urls = [
        # Legitimate (should NOT be flagged)
        "https://phishguard.co.in",
        "https://www.google.co.in",
        "https://www.amazon.in",
        "https://www.hdfcbank.com",
        "https://www.sbi.co.in",
        "https://www.yesbank.in",
        
        # Phishing (should be flagged)
        "https://hdfc-secure-login.xyz",
        "https://paypai-secure-verify.xyz",
        "http://bit.ly/3xYZ123",
        "http://192.168.1.1/login",
    ]
    
    for url in test_urls:
        result = detector.predict(url)
        print(f"{result['verdict'].upper()}: {url}")
        print(f"  Layer: {result['layer']}, Confidence: {result['confidence']:.1%}")
        print(f"  Why: {result['explanation']}\n")