import os
import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from typing import Dict
from pathlib import Path

# Explicitly load environment variables from the root folder (.env) for local testing
base_dir = Path(__file__).resolve().parent.parent
env_path = base_dir / '.env'
load_dotenv(dotenv_path=env_path)

# Dynamically map the static assets folder to keep path routing robust on Vercel
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.normpath(os.path.join(current_dir, '..', 'frontend'))

app = Flask(__name__, static_folder=static_dir, static_url_path='')
CORS(app)

class CodeOutputParser(BaseOutputParser[Dict[str, str]]):
    def parse(self, text: str) -> Dict[str, str]:
        # Your exact validated layout regex configuration
        pattern = (
            r'(?:```html)?\s*###HTML###\s*(.*?)\s*###END###(?:```)?\s*'
            r'(?:```css)?\s*###CSS###\s*(.*?)\s*###END###(?:```)?\s*'
            r'(?:```javascript|```js)?\s*###JS###\s*(.*?)\s*###END###(?:```)?'
        )
        
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            return {
                "html": match.group(1).strip(),
                "css": match.group(2).strip(),
                "js": match.group(3).strip()
            }
            
        if '###HTML###' not in text:
            return {"html": text.strip(), "css": "", "js": ""}
            
        raise ValueError("Could not parse output. Please try again.")

# Initializing with the verified high-performance Llama-3.3 model
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

system_prompt = """You are an expert full-stack web developer. Generate a complete website according to the user's request. 
You MUST output the code in exactly the following format:
###HTML###
(Your full HTML code here – only what goes inside <body>)
###END###
###CSS###
(Your full CSS code here)
###END###
###JS###
(Your full JavaScript code here)
###END###

Do NOT include any other text or explanations. Make the design modern, responsive, and visually appealing."""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

chain = prompt_template | llm | CodeOutputParser()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json() or {}
    user_prompt = data.get('prompt', '').strip()
    
    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400
        
    try:
        parsed = chain.invoke({"input": user_prompt})
        return jsonify(parsed)
    except Exception as e:
        try:
            raw_response = llm.invoke(f"Generate a website: {user_prompt}").content
            return jsonify({"html": raw_response, "css": "", "js": ""})
        except Exception as fallback_error:
            return jsonify({"error": f"Generation failed: {str(fallback_error)}"}), 500

# Kept for local fallback execution execution block
if __name__ == '__main__':
    app.run(debug=True, port=5000)
