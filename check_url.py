# check_url.py
"""
Simple URL Checker - Enter a URL and get instant result with explanation
"""

import sys
import os
import re
from urllib.parse import urlparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hybrid_model_fixed_v2 import HybridPhishingDetector

# ===================== INITIALIZE DETECTOR =====================
print("🔧 Loading Phishing Detection Model...")
detector = HybridPhishingDetector()
print("✅ Model loaded successfully!\n")

# ===================== MAIN LOOP =====================
while True:
    # Get URL from user
    url = input("\n🔗 Enter URL to check (or 'quit' to exit): ").strip()
    
    # Check for exit
    if url.lower() in ['quit', 'exit', 'q']:
        print("👋 Goodbye!")
        break
    
    # Skip empty input
    if not url:
        print("⚠️ Please enter a valid URL")
        continue
    
    print("\n" + "="*70)
    print(f"🔍 Checking: {url}")
    print("="*70)
    
    try:
        # Predict
        result = detector.predict(url)
        
        # ===================== OUTPUT =====================
        print("\n" + "="*70)
        
        # Verdict
        if result['verdict'] == 'phishing':
            print("🚨 VERDICT: PHISHING")
            print("⚠️ This URL appears to be a phishing attempt!")
        else:
            print("✅ VERDICT: LEGITIMATE")
            print("✅ This URL appears to be safe.")
        
        print("="*70)
        
        # Confidence
        print(f"\n📊 Confidence: {result['confidence']:.1%}")
        
        # Detection Layer
        print(f"📋 Detection Layer: {result['layer']}")
        
        # Explanation
        if result.get('explanation'):
            print(f"\n📝 Explanation: {result['explanation']}")
        
        # Detailed reasons
        if result.get('details'):
            print(f"\n🔍 Detailed Analysis:")
            for detail in result['details']:
                print(f"  • {detail}")
        
        # ===================== FEATURE BREAKDOWN =====================
        # Extract features for display
        from hybrid_model_fixed_v2 import HybridPhishingDetector
        # Create a temporary detector just for features
        temp_detector = HybridPhishingDetector()
        features = temp_detector.extract_features(url)
        
        print(f"\n📊 URL Features:")
        print(f"  • URL Length: {features['url_length']}")
        print(f"  • Domain Length: {features['domain_length']}")
        print(f"  • Subdomain Length: {features['subdomain_length']}")
        print(f"  • Path Length: {features['path_length']}")
        print(f"  • Number of Dots: {features['num_dots']}")
        print(f"  • Number of Hyphens: {features['num_hyphens']}")
        print(f"  • Number of Digits: {features['num_digits']}")
        print(f"  • Sensitive Words: {features['num_sensitive_words']}")
        print(f"  • Uses HTTPS: {'Yes' if features['has_https'] else 'No'}")
        print(f"  • Contains IP: {'Yes' if features['has_ip'] else 'No'}")
        print(f"  • Suspicious TLD: {'Yes' if features['has_suspicious_tld'] else 'No'}")
        print(f"  • Shortened URL: {'Yes' if features['is_shortened'] else 'No'}")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check the URL and try again.")