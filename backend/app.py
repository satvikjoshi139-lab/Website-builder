import os
from flask import Flask, jsonify
from dotenv import load_dotenv
from langchain_groq import ChatGroq  # Changed from langchain_xai

# Explicitly point to backend/.env
base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, '.env')
load_dotenv(dotenv_path=env_path)

app = Flask(__name__)

# 1. Verify API key is loaded
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError(f"API key not found. Looked in: {env_path}")

# 2. Initialize the Groq model
llm = ChatGroq(
    groq_api_key=api_key,
    model="llama-3.3-70b-versatile",  # High-performance, fast model on Groq
    temperature=0.7,
)

# Health check endpoint
@app.route('/health')
def health():
    return jsonify({"status": "ok", "provider": "groq", "model": "llama-3.3-70b-specdec"})

# Quick test endpoint to confirm Grok/Groq connection
@app.route('/test-grok')
def test_grok():
    try:
        response = llm.invoke("Say 'Hello, builder!' in a friendly way.")
        return jsonify({"response": response.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

