# api/app.py (Full version with conditional LLM)

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
CORS(app)

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
        print(f"   LLM will only trigger for ambiguous URLs")
    else:
        print("⚠️ GEMINI_API_KEY not set. LLM disabled.")
except ImportError:
    print("⚠️ google-generativeai not installed. LLM disabled.")
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
        domain = parsed.netloc.split(':')[0]
        if domain.startswith('www.'):
            domain = domain[4:]
        
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        
        if creation_date:
            days = (datetime.now() - creation_date).days
            return days, creation_date, w.expiration_date
        return None, None, None
    except:
        return None, None, None

def check_ssl_certificate(url):
    try:
        parsed = urlparse(url if '://' in url else f'//{url}')
        hostname = parsed.netloc.split(':')[0]
        if not hostname:
            return False, None, None
        
        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                issuer = dict(x[0] for x in cert['issuer'])
                issuer_name = issuer.get('organizationName', issuer.get('commonName', 'Unknown'))
                expiry = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                days = (expiry - datetime.now()).days
                return days > 0, issuer_name, days
    except:
        return False, None, None

def analyze_with_llm(url, ml_confidence, ml_explanation):
    """Use Gemini LLM for deep analysis"""
    global app
    
    if not USE_LLM or llm_model is None:
        return 'ambiguous', 0.5, "LLM not available"
    
    app.llm_calls += 1
    print(f"🧠 LLM Triggered: {url[:50]}...")
    
    try:
        import requests
        content = ""
        try:
            r = requests.get(url, timeout=3, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code == 200:
                content = r.text[:3000]
        except:
            pass
        
        prompt = f"""
URL: {url}
ML Confidence: {ml_confidence:.1%}
ML Explanation: {ml_explanation}
{'Content: ' + content[:2000] if content else ''}

Is this phishing?
Verdict: YES/NO
Confidence: XX%
Explanation: Brief explanation
"""
        
        response = llm_model.generate_content(prompt)
        text = response.text
        
        verdict = "ambiguous"
        confidence = 50
        explanation = text[:300]
        
        if "Verdict: YES" in text:
            verdict = "phishing"
        elif "Verdict: NO" in text:
            verdict = "legitimate"
        
        conf_match = re.search(r'Confidence:\s*(\d+)%', text)
        if conf_match:
            confidence = int(conf_match.group(1))
        
        return verdict, confidence / 100, explanation
        
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return 'ambiguous', 0.5, f"LLM error: {str(e)[:100]}"

# ===================== ROUTES =====================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'name': 'PhishingNet API',
        'version': '3.0.0',
        'status': 'running',
        'llm_enabled': USE_LLM,
        'llm_trigger_condition': 'ML confidence < 70% OR ambiguous'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'llm_enabled': USE_LLM,
        'cache_size': len(cache)
    })

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
        
        # === LAYER 1 & 2: Hybrid ML ===
        result = detector.predict(url)
        
        # === DOMAIN AGE & SSL ===
        domain_age, creation, expiry = get_domain_age(url)
        ssl_valid, ssl_issuer, ssl_days = check_ssl_certificate(url)
        
        extra_details = []
        
        if domain_age is not None:
            if domain_age < 7:
                extra_details.append(f"⚠️ Domain {domain_age} days old (very new)")
            elif domain_age < 30:
                extra_details.append(f"⚠️ Domain {domain_age} days old")
            else:
                extra_details.append(f"✅ Domain age: {domain_age} days")
        else:
            extra_details.append("❓ Domain age unknown")
        
        if ssl_valid is not None:
            if ssl_valid:
                extra_details.append(f"✅ SSL valid ({ssl_days} days left)")
            else:
                extra_details.append("❌ Invalid SSL certificate")
        else:
            extra_details.append("❓ SSL status unknown")
        
        all_details = result.get('details', []) + extra_details
        
        # === LAYER 3: LLM (ONLY IF NEEDED) ===
        llm_used = False
        llm_result = {}
        
        # Determine if LLM is needed
        needs_llm = (
            USE_LLM and (
                result['verdict'] == 'ambiguous' or
                result['confidence'] < 0.70 or
                (domain_age is not None and domain_age < 7) or
                (ssl_valid is False and result['confidence'] < 0.75)
            )
        )
        
        if needs_llm:
            print(f"🧠 LLM Triggered: {url[:50]}...")
            llm_verdict, llm_confidence, llm_explanation = analyze_with_llm(
                url, result['confidence'], result.get('explanation', '')
            )
            llm_used = True
            llm_result = {
                'verdict': llm_verdict,
                'confidence': llm_confidence,
                'explanation': llm_explanation
            }
            
            if llm_verdict != 'ambiguous':
                final_verdict = llm_verdict
                final_confidence = max(result['confidence'], llm_confidence)
                final_layer = f"{result['layer']} + LLM"
                final_explanation = f"ML: {result.get('explanation', '')} | LLM: {llm_explanation}"
            else:
                final_verdict = result['verdict']
                final_confidence = result['confidence']
                final_layer = result['layer']
                final_explanation = result.get('explanation', '')
        else:
            final_verdict = result['verdict']
            final_confidence = result['confidence']
            final_layer = result['layer']
            final_explanation = result.get('explanation', '')
        
        response = {
            'url': url,
            'url_hash': url_hash,
            'verdict': final_verdict,
            'confidence': final_confidence,
            'layer': final_layer,
            'explanation': final_explanation,
            'details': all_details,
            'domain_age_days': domain_age,
            'ssl_valid': ssl_valid,
            'llm_used': llm_used,
            'llm_triggered': llm_used,
            'timestamp': datetime.now().isoformat()
        }
        
        if llm_used:
            response['llm_result'] = llm_result
        
        if len(cache) >= CACHE_SIZE:
            del cache[next(iter(cache))]
        cache[url_hash] = response
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    return jsonify({
        'total_predictions': app.total_predictions,
        'cache_hits': app.cache_hits,
        'cache_misses': app.cache_misses,
        'llm_calls': app.llm_calls,
        'cache_size': len(cache),
        'llm_enabled': USE_LLM
    })

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    cache.clear()
    return jsonify({'message': 'Cache cleared'})

if __name__ == '__main__':
    print("\n" + "="*60)
    print(f"🚀 API Server Starting...")
    print(f"📍 http://127.0.0.1:5000")
    print(f"🤖 LLM Enabled: {USE_LLM} (conditional)")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)