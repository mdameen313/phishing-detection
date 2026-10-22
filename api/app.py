# api/app.py

import sys
import os
import hashlib
import re
import ssl
import socket
import whois
from datetime import datetime
from urllib.parse import urlparse
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from hybrid_model_fixed_v2 import HybridPhishingDetector

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ===================== INITIALIZE DETECTOR =====================
print("="*60)
print("🚀 STARTING PHISHING DETECTION API")
print("="*60)

try:
    detector = HybridPhishingDetector()
    print("✅ Detector initialized")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# ===================== INITIALIZE LLM (CONDITIONAL) =====================
USE_LLM = False
llm_model = None
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

try:
    if GEMINI_API_KEY and GEMINI_API_KEY != '':
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        llm_model = genai.GenerativeModel('gemini-1.5-flash')
        USE_LLM = True
        print("✅ Gemini LLM initialized (conditional mode)")
    else:
        print("⚠️ GEMINI_API_KEY not set. LLM disabled.")
except Exception as e:
    print(f"⚠️ LLM init error: {e}")

# ===================== CACHE =====================
cache = {}
CACHE_SIZE = 1000
app.total_predictions = 0
app.cache_hits = 0
app.cache_misses = 0
app.llm_calls = 0

# ===================== HELPERS =====================

def get_domain_age(url):
    try:
        parsed = urlparse(url if '://' in url else f'//{url}')
        domain = parsed.netloc.split(':')[0].replace('www.', '')
        w = whois.whois(domain)
        creation_date = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
        if creation_date:
            return (datetime.now() - creation_date).days
    except:
        pass
    return None

def check_ssl_certificate(url):
    try:
        parsed = urlparse(url if '://' in url else f'//{url}')
        hostname = parsed.netloc.split(':')[0]
        if not hostname: return False
        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=3) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                expiry = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                return (expiry - datetime.now()).days > 0
    except:
        return False

def analyze_with_llm(url, ml_confidence, ml_explanation, domain_age=None):
    global app
    if not USE_LLM or llm_model is None:
        return 'ambiguous', 0.5, "LLM not available"
    
    app.llm_calls += 1
    print(f"🧠 LLM Triggered: {url[:50]}...")
    
    try:
        # Better prompt context
        age_note = f"Domain age: {domain_age} days." if domain_age is not None else "Domain age: Unknown."
        
        prompt = f"""
Analyze this URL for phishing:
URL: {url}
ML Confidence: {ml_confidence:.1%}
{age_note}
ML Explanation: {ml_explanation}

Respond exactly in this format:
Verdict: YES or NO
Confidence: XX%
Explanation: Brief explanation
"""
        
        response = llm_model.generate_content(prompt)
        text = response.text
        
        verdict = "ambiguous"
        confidence = 0.5
        explanation = text[:300]
        
        # Robust parsing
        if "Verdict: YES" in text or "Verdict:YES" in text: 
            verdict = "phishing"
        elif "Verdict: NO" in text or "Verdict:NO" in text: 
            verdict = "legitimate"
        
        conf_match = re.search(r'Confidence:\s*(\d+)%', text)
        if conf_match:
            confidence = int(conf_match.group(1)) / 100
        
        return verdict, confidence, explanation
        
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return 'ambiguous', 0.5, f"LLM error: {str(e)[:100]}"

# ===================== ROUTES =====================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'Missing URL'}), 400
        
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        
        if url_hash in cache:
            app.cache_hits += 1
            return jsonify({**cache[url_hash], 'cached': True})
        
        app.cache_misses += 1
        app.total_predictions += 1
        
        # === LAYER 1 & 2 ===
        result = detector.predict(url)
        
        # === DYNAMIC DETAILS ===
        domain_age = get_domain_age(url)
        ssl_valid = check_ssl_certificate(url)
        
        details = []
        
        # Add details from ML/CBA
        if result.get('details'):
            details.extend(result['details'])
            
        # Add Domain Age detail
        if domain_age is not None:
            if domain_age < 7:
                details.append(f"⚠️ Domain is only {domain_age} days old (Very new)")
            elif domain_age < 30:
                details.append(f"⚠️ Domain is {domain_age} days old")
            else:
                details.append(f"✅ Domain age: {domain_age} days")
        else:
            details.append("❓ Domain age could not be determined")
            
        # Add SSL detail
        if ssl_valid:
            details.append("✅ Valid SSL certificate")
        else:
            details.append("❌ Invalid or missing SSL certificate")
        
        # === LAYER 3: LLM ===
        llm_used = False
        final_verdict = result['verdict']
        final_confidence = result['confidence']
        final_layer = result['layer']
        final_explanation = result.get('explanation', '')
        
        # Trigger LLM if unsure
        if USE_LLM and (final_verdict == 'suspicious' or final_confidence < 0.65):
            llm_v, llm_c, llm_e = analyze_with_llm(url, result['confidence'], result.get('explanation', ''), domain_age)
            if llm_v != 'ambiguous':
                final_verdict = llm_v
                final_confidence = max(result['confidence'], llm_c)
                final_layer = f"{result['layer']} + LLM"
                final_explanation = f"ML: {result.get('explanation', '')} | LLM: {llm_e}"
                details.append(f"🧠 LLM Analysis: {llm_e}")
                llm_used = True
        
        response = {
            'url': url,
            'url_hash': url_hash,
            'verdict': final_verdict,
            'confidence': final_confidence, # <--- DYNAMIC FLOAT (e.g. 0.865)
            'layer': final_layer,
            'explanation': final_explanation,
            'details': details,             # <--- FULL ARRAY OF CONSTRAINTS
            'domain_age_days': domain_age,
            'ssl_valid': ssl_valid,
            'llm_used': llm_used
        }
        
        if len(cache) >= CACHE_SIZE:
            del cache[next(iter(cache))]
        cache[url_hash] = response
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)