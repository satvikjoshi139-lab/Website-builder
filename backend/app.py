import os
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from langchain_groq import ChatGroq  # Changed from langchain_xai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = Flask(__name__, static_folder='../frontend', static_url_path='')

# LangChain pipeline configured for Groq
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),  # Uses your Groq key
    model="llama-3.3-70b-versatile",         # Excellent, fast model for coding
    temperature=0.7,
)

# This is the magic – it forces the model to output pure HTML, nothing else
system_prompt = """You are an expert full-stack web developer.
Generate a complete, self-contained HTML page that fulfills the user's request.
The page must include all necessary CSS inside <style> tags and all JavaScript inside <script> tags.
Make the design modern, responsive, and visually appealing.
Respond ONLY with the raw HTML code — no markdown, no explanations, no surrounding text."""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

chain = prompt_template | llm | StrOutputParser()

# Serve the frontend
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# API endpoint
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    user_prompt = data.get('prompt', '').strip()
    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        raw_html = chain.invoke({"input": user_prompt})
        # Clean markdown code fences if the model adds them
        if raw_html.startswith("```html"):
            raw_html = raw_html[7:]
        elif raw_html.startswith("```"):
            raw_html = raw_html[3:]
        if raw_html.endswith("```"):
            raw_html = raw_html[:-3]
        return jsonify({"html": raw_html.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
