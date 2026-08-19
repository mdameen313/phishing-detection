# hybrid_model_fixed_v2.py
"""
FINAL FIXED HYBRID MODEL - CORRECTED
"""

import pandas as pd
import numpy as np
import joblib
import re
import tldextract
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

class HybridPhishingDetector:
    def __init__(self):
        print("="*60)
        print("INITIALIZING HYBRID PHISHING DETECTOR (FINAL FIX)")
        print("="*60)
        
        try:
            self.rules_config = joblib.load('models/cba_expert_rules_final.pkl')
            print("✅ Layer 1: Expert CBA Rules loaded")
            
            self.rf_model = joblib.load('models/random_forest_final_v3.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            print("✅ Layer 2: Random Forest loaded")
            
            train_df = pd.read_csv('processed_data/train_enhanced.csv')
            self.feature_cols = [col for col in train_df.columns if col != 'label']
            print(f"✅ {len(self.feature_cols)} features loaded")
            
            print("\n🚀 Hybrid Detector Ready!")
        except FileNotFoundError as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def check_expert_rules(self, url):
        """Check expert-defined CBA rules"""
        url_lower = url.lower()
        triggered = []
        explanations = []
        
        clean_url = re.sub(r'^https?://', '', url_lower)
        clean_url = re.sub(r'^www\.', '', clean_url)
        
        # Check official domain (WHITELIST)
        is_official = False
        for domain in self.rules_config['official_domains']:
            if clean_url.startswith(domain) or f'.{domain}' in clean_url:
                is_official = True
                explanations.append(f"Official domain: {domain}")
                break
        
        # FIX: Return False for legitimate domains (NOT phishing)
        if is_official:
            return False, 0.99, [], explanations  # ✅ CORRECT: False = not phishing
        
        # Rule 1: Suspicious TLD
        if any(tld in url_lower for tld in self.rules_config['suspicious_tlds']):
            triggered.append(('Suspicious TLD', 0.85))
            explanations.append("Suspicious TLD (.xyz, .tk, .ml, etc.)")
        
        # Rule 2: IP Address
        if re.search(r'\d+\.\d+\.\d+\.\d+', url):
            triggered.append(('IP address used', 0.95))
            explanations.append("IP address instead of domain")
        
        # Rule 3: @ symbol
        if '@' in url:
            triggered.append(('@ symbol in URL', 0.90))
            explanations.append("@ symbol in URL")
        
        # Rule 4: Shortened URL
        if any(s in url_lower for s in self.rules_config['shorteners']):
            triggered.append(('Shortened URL', 0.70))
            explanations.append("Shortened URL")
        
        # Rule 5: Brand misuse
        has_sensitive = any(word in url_lower for word in self.rules_config['sensitive_words'])
        has_brand = any(brand in url_lower for brand in self.rules_config['brands'])
        
        if has_brand and has_sensitive:
            is_brand_com = any(f'{brand}.com' in url_lower for brand in self.rules_config['brands'])
            is_brand_in = any(f'{brand}.in' in url_lower or f'{brand}.co.in' in url_lower for brand in self.rules_config['brands'])
            
            if not is_brand_com and not is_brand_in:
                triggered.append(('Brand misuse with sensitive words', 0.88))
                explanations.append("Brand name used with login/verify keywords")
        
        # Rule 6: Multiple subdomains
        domain_part = clean_url.split('/')[0]
        dot_count = domain_part.count('.')
        if dot_count > 3:
            triggered.append(('Multiple subdomains', 0.75))
            explanations.append(f"Multiple subdomains ({dot_count} dots)")
        
        # Rule 7: HTTP for sensitive pages
        if url.startswith('http://') and has_sensitive:
            triggered.append(('HTTP for sensitive page', 0.70))
            explanations.append("HTTP for login/secure page")
        
        if triggered:
            confidence = sum([conf for _, conf in triggered]) / len(triggered)
            confidence = min(confidence, 0.95)
            return True, confidence, triggered, explanations  # True = phishing
        else:
            return False, 0, [], explanations  # False = not phishing
    
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
            'num_sensitive_words': sum(1 for w in self.rules_config['sensitive_words'] if w in clean.lower()),
            'has_@': 1 if '@' in clean else 0,
            'has_https': has_https,
            'has_ip': 1 if re.search(r'\d+\.\d+\.\d+\.\d+', clean) else 0,
            'has_suspicious_tld': 1 if domain_info.suffix in self.rules_config['suspicious_tlds'] else 0,
            'is_shortened': 1 if any(s in clean for s in self.rules_config['shorteners']) else 0,
        }
        return features
    
    def predict_rf(self, url):
        """Random Forest prediction"""
        features_dict = self.extract_features(url)
        features_df = pd.DataFrame([features_dict])
        features_df = features_df[self.feature_cols]
        
        features_scaled = self.scaler.transform(features_df)
        features_scaled = pd.DataFrame(features_scaled, columns=self.feature_cols)
        
        prob = self.rf_model.predict_proba(features_scaled)[0][1]
        pred = 1 if prob >= 0.5 else 0
        
        return pred, prob
    
    def predict(self, url):
        """
        Hybrid Prediction Pipeline
        """
        
        # LAYER 1: Expert CBA Rules
        is_phishing, cba_confidence, triggered_rules, explanations = self.check_expert_rules(url)
        
        # If CBA says it's phishing with high confidence → PHISHING
        if is_phishing and cba_confidence >= 0.70:
            return {
                'verdict': 'phishing',
                'confidence': cba_confidence,
                'layer': 'CBA_Expert_Rules',
                'explanation': f"Rule fired: {explanations[0] if explanations else 'Multiple rules'}",
                'triggered_rules': triggered_rules,
                'explanations': explanations
            }
        
        # If CBA says official domain → LEGITIMATE
        if explanations and 'Official domain' in str(explanations):
            return {
                'verdict': 'legitimate',
                'confidence': 0.99,
                'layer': 'CBA_Whitelist',
                'explanation': f"Official domain detected",
                'triggered_rules': [],
                'explanations': explanations
            }
        
        # If CBA has NO rules triggered → likely LEGITIMATE
        if not is_phishing and not triggered_rules:
            return {
                'verdict': 'legitimate',
                'confidence': 0.85,
                'layer': 'CBA_No_Rules',
                'explanation': "No suspicious patterns detected",
                'triggered_rules': [],
                'explanations': explanations
            }
        
        # LAYER 2: Random Forest
        rf_pred, rf_prob = self.predict_rf(url)
        
        # If RF is confident enough
        if rf_prob >= 0.85:
            return {
                'verdict': 'phishing' if rf_pred == 1 else 'legitimate',
                'confidence': rf_prob if rf_pred == 1 else (1 - rf_prob),
                'layer': 'RandomForest',
                'explanation': f"RF prediction with {rf_prob:.1%} phishing confidence",
                'triggered_rules': triggered_rules,
                'explanations': explanations
            }
        else:
            # Default: legitimate if not strongly phishing
            return {
                'verdict': 'legitimate',
                'confidence': 1 - rf_prob,
                'layer': 'Default_Legitimate',
                'explanation': f"Low phishing confidence ({rf_prob:.1%}) - classified as legitimate",
                'triggered_rules': triggered_rules,
                'explanations': explanations
            }

# ===================== TEST =====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING FINAL HYBRID MODEL (CORRECTED)")
    print("="*60)
    
    detector = HybridPhishingDetector()
    
    test_urls = [
        # Legitimate URLs
        "https://www.google.com",
        "https://www.facebook.com",
        "https://www.amazon.com",
        "https://www.hdfcbank.com",
        "https://www.icicibank.com",
        "https://www.paypal.com",
        "https://www.sbi.co.in",
        "https://www.axisbank.com",
        "https://www.kotak.com",
        "https://www.yesbank.in",
        
        # Phishing URLs
        "https://secure-login.paypal-verify.xyz",
        "https://login.paypal.secure-login.xyz",
        "https://hdfc-secure-login.xyz",
        "https://sbi-secure-login.xyz",
        "http://paypal-secure-login.xyz",
        "http://192.168.1.1/login",
        "http://bit.ly/3xYZ123",
    ]
    
    expected = ['Legitimate'] * 10 + ['Phishing'] * 7
    
    print("\n" + "-"*70)
    print(f"{'URL':<45} {'Prediction':<15} {'Layer':<22} {'Confidence':<10}")
    print("-"*70)
    
    correct = 0
    total = len(test_urls)
    
    for i, url in enumerate(test_urls):
        result = detector.predict(url)
        
        is_correct = (result['verdict'] == 'phishing' and expected[i] == 'Phishing') or \
                     (result['verdict'] == 'legitimate' and expected[i] == 'Legitimate')
        if is_correct:
            correct += 1
        
        check = "✓" if is_correct else "✗"
        status = f"{result['verdict']} {check}"
        display_url = url[:40] + '...' if len(url) > 40 else url
        
        print(f"{display_url:<45} {status:<15} {result['layer']:<22} {result['confidence']:.1%}")
    
    print("-"*70)
    print(f"\n📊 Accuracy: {correct}/{total} ({correct/total*100:.0f}%)")
    
    # Test a suspicious URL with explanation
    print("\n" + "="*60)
    print("EXAMPLE WITH EXPLANATIONS")
    print("="*60)
    
    test_url = "https://hdfc-secure-login.xyz"
    result = detector.predict(test_url)
    
    print(f"🔍 URL: {test_url}")
    print(f"📌 Verdict: {result['verdict'].upper()}")
    print(f"🎯 Confidence: {result['confidence']:.1%}")
    print(f"📋 Layer: {result['layer']}")
    print(f"📝 Explanation: {result['explanation']}")
    print("\n🔴 Why it's phishing:")
    for exp in result['explanations']:
        print(f"  - {exp}")
    
    print("\n" + "="*60)
    print("✅ FINAL HYBRID MODEL READY!")