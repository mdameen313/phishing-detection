# api/app.py
"""
Phishing Detection API - Flask Server
Exposes the hybrid model via REST API
"""

import sys
import os
import hashlib
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hybrid_model_fixed_v2 import HybridPhishingDetector

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for browser extensions

# Initialize detector
print("="*60)
print("🚀 STARTING PHISHING DETECTION API")
print("="*60)

try:
    detector = HybridPhishingDetector()
    print("✅ Detector initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize detector: {e}")
    sys.exit(1)

# In-memory cache (for demo - use Redis/Firestore for production)
cache = {}
CACHE_SIZE = 1000
CACHE_TTL = 3600  # 1 hour in seconds

# ===================== ROUTES =====================

@app.route('/', methods=['GET'])
def home():
    """API root endpoint"""
    return jsonify({
        'name': 'Phishing Detection API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            '/predict': 'POST - Detect phishing URL',
            '/predict_batch': 'POST - Detect multiple URLs',
            '/health': 'GET - Check API health',
            '/stats': 'GET - Get API stats'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'cache_size': len(cache)
    })

@app.route('/stats', methods=['GET'])
def stats():
    """Get API statistics"""
    return jsonify({
        'total_predictions': getattr(app, 'total_predictions', 0),
        'cache_hits': getattr(app, 'cache_hits', 0),
        'cache_misses': getattr(app, 'cache_misses', 0),
        'cache_size': len(cache)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if a URL is phishing
    Expected JSON: {"url": "https://example.com"}
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'Missing "url" field'}), 400
        
        # Hash URL for cache key (privacy!)
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        
        # Check cache
        if url_hash in cache:
            cached_result = cache[url_hash]
            app.cache_hits = getattr(app, 'cache_hits', 0) + 1
            app.total_predictions = getattr(app, 'total_predictions', 0) + 1
            return jsonify({
                **cached_result,
                'cached': True,
                'url_hash': url_hash
            })
        
        app.cache_misses = getattr(app, 'cache_misses', 0) + 1
        app.total_predictions = getattr(app, 'total_predictions', 0) + 1
        
        # Run prediction
        result = detector.predict(url)
        
        # Prepare response
        response = {
            'url': url,
            'url_hash': url_hash,
            'verdict': result['verdict'],
            'confidence': result['confidence'],
            'layer': result['layer'],
            'explanation': result.get('explanation', ''),
            'details': result.get('details', []),
            'timestamp': datetime.now().isoformat(),
            'cached': False
        }
        
        # Store in cache
        if len(cache) >= CACHE_SIZE:
            # Remove oldest entry
            oldest_key = next(iter(cache))
            del cache[oldest_key]
        
        cache[url_hash] = response
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Internal server error'
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict multiple URLs
    Expected JSON: {"urls": ["https://example1.com", "https://example2.com"]}
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        urls = data.get('urls', [])
        
        if not urls:
            return jsonify({'error': 'Missing "urls" field'}), 400
        
        if not isinstance(urls, list):
            return jsonify({'error': '"urls" must be a list'}), 400
        
        if len(urls) > 50:
            return jsonify({'error': 'Maximum 50 URLs per batch'}), 400
        
        results = []
        for url in urls:
            result = detector.predict(url)
            results.append({
                'url': url,
                'verdict': result['verdict'],
                'confidence': result['confidence'],
                'layer': result['layer'],
                'explanation': result.get('explanation', '')
            })
        
        return jsonify({
            'total': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Internal server error'
        }), 500

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the prediction cache"""
    cache.clear()
    return jsonify({
        'message': 'Cache cleared',
        'cache_size': 0
    })

# ===================== ERROR HANDLERS =====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ===================== MAIN =====================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 API Server Starting...")
    print("="*60)
    print(f"📍 Running on: http://127.0.0.1:5000")
    print(f"📊 Health check: http://127.0.0.1:5000/health")
    print(f"🔮 Predict: POST http://127.0.0.1:5000/predict")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)