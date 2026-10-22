# test_groq_live.py
from groq import Groq

client = Groq(api_key="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")

prompt = """Analyze this URL for phishing:
URL: 		https://meetamassklogaiin.webflow.io/

Respond ONLY:
VERDICT: PHISHING or LEGITIMATE
CONFIDENCE: XX%"""

response = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    model="openai/gpt-oss-120b",
    max_tokens=500,
    temperature=0.1,
)

print("Raw response:")
print(response.choices[0].message.content)