# check_groq_models.py
from groq import Groq

GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

print("="*70)
print("AVAILABLE GROQ MODELS")
print("="*70)

try:
    models = client.models.list()
    for model in models.data:
        print(f"  ✅ {model.id}")
except Exception as e:
    print(f"❌ Error: {e}")