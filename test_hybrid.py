# test_hybrid.py
"""
Test the hybrid model locally before deployment
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hybrid_model_fixed_v2 import HybridPhishingDetector

print("="*70)
print("TESTING HYBRID MODEL (Threshold = 0.65)")
print("="*70)

# Initialize detector
detector = HybridPhishingDetector()

test_urls = [
    # Legitimate URLs (should be safe)
    "https://www.google.com",
    "https://www.facebook.com",
    "https://www.amazon.com",
    "https://www.hdfcbank.com",
    "https://www.paypal.com",
    "https://www.sbi.co.in",
    "https://www.icicibank.com",
    "https://www.axisbank.com",
    "https://www.yesbank.in",
    "https://www.github.com",
    
    # Phishing URLs (should be blocked)
    "https://hdfc-secure-login.xyz",
    "https://secure-login-paypal.xyz",
    "https://paypai-secure-verify.xyz",
    "https://amaz0n-verify.account-security.top",
    "http://bit.ly/3xYZ123",
    "http://192.168.1.1/login",
    "https://sbi-secure-login.xyz",
    "https://bankofamerica-verify.xyz",
]

print("\n" + "-"*70)
print(f"{'URL':<50} {'Verdict':<12} {'Confidence':<12} {'Layer':<15}")
print("-"*70)

phishing_count = 0
legit_count = 0
correct = 0
total = len(test_urls)

for url in test_urls:
    result = detector.predict(url)
    
    # Determine if verdict is correct (expected)
    is_phishing_expected = any(domain in url.lower() for domain in 
                               ['hdfc-secure', 'secure-login-paypal', 'paypai', 
                                'amaz0n', 'bit.ly', '192.168', 'sbi-secure', 'bankofamerica-verify'])
    is_legit_expected = not is_phishing_expected
    
    is_correct = (result['verdict'] == 'phishing' and is_phishing_expected) or \
                 (result['verdict'] == 'legitimate' and is_legit_expected)
    
    if is_correct:
        correct += 1
    
    if result['verdict'] == 'phishing':
        phishing_count += 1
    else:
        legit_count += 1
    
    status = "⚠️ PHISHING" if result['verdict'] == 'phishing' else "✅ LEGITIMATE"
    check = "✓" if is_correct else "✗"
    display_url = url[:47] + '...' if len(url) > 47 else url
    
    print(f"{display_url:<50} {status:<12} {result['confidence']:.1%}      {result['layer']:<15} {check}")

print("-"*70)
print(f"\n📊 SUMMARY:")
print(f"  Total URLs: {total}")
print(f"  Phishing Detected: {phishing_count}")
print(f"  Legitimate Detected: {legit_count}")
print(f"  Correct Predictions: {correct}/{total} ({correct/total*100:.1f}%)")

# Show detailed analysis for suspicious URLs
print("\n" + "="*70)
print("🔍 DETAILED ANALYSIS FOR PHISHING URLS")
print("="*70)

for url in test_urls:
    if 'hdfc-secure' in url or 'paypal' in url and 'secure' in url:
        result = detector.predict(url)
        print(f"\n📌 URL: {url}")
        print(f"   Verdict: {result['verdict'].upper()}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Layer: {result['layer']}")
        if result.get('details'):
            print("   Reasons:")
            for detail in result['details'][:3]:
                print(f"     - {detail}")