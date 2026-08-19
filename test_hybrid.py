# test_hybrid.py
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hybrid_model_fixed_v2 import HybridPhishingDetector

print("="*60)
print("TESTING HYBRID PHISHING DETECTOR")
print("="*60)

# Initialize detector
detector = HybridPhishingDetector()

test_urls = [
    # Legitimate URLs
    "https://www.google.com",
    "https://www.facebook.com",
    "https://www.amazon.com",
    "https://www.hdfcbank.com",
    "https://www.paypal.com",
    "https://www.sbi.co.in",
    
    # Phishing URLs
    "https://secure-login.paypal-verify.xyz",
    "https://hdfc-secure-login.xyz",
    "http://bit.ly/3xYZ123",
    "http://192.168.1.1/login",
    "https://sbi-secure-login.xyz",
]

print("\n" + "-"*70)
print(f"{'URL':<45} {'Verdict':<12} {'Confidence':<12} {'Layer':<15}")
print("-"*70)

for url in test_urls:
    result = detector.predict(url)
    
    status = "⚠️ PHISHING" if result['verdict'] == 'phishing' else "✅ LEGITIMATE"
    display_url = url[:42] + '...' if len(url) > 42 else url
    
    print(f"{display_url:<45} {status:<12} {result['confidence']:.1%}      {result['layer']:<15}")

print("-"*70)
print("\n✅ Test complete!")