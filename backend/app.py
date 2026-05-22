import os
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from typing import Dict

load_dotenv()

app = Flask(__name__, static_folder='../frontend', static_url_path='')

class CodeOutputParser(BaseOutputParser[Dict[str, str]]):
    def parse(self, text: str) -> Dict[str, str]:
        parts = {}
        for block in ["HTML", "CSS", "JS"]:
            start = f"###{block}###"
            end = "###CSS###" if block == "HTML" else ("###JS###" if block == "CSS" else "###END###")
            try:
                if start in text and end in text:
                    start_idx = text.index(start) + len(start)
                    end_idx = text.index(end, start_idx)
                    parts[block.lower()] = text[start_idx:end_idx].strip()
                else:
                    parts[block.lower()] = ""
            except ValueError:
                parts[block.lower()] = ""
        return parts

# Low temperature ensures strict structural adherence to the design rules
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.2,
)

system_prompt = """You are a master Principal UI/UX Designer and Lead Full-Stack Engineer.
Your task is to generate a gorgeous, world-class, premium web application layout matching the user's specific text request.

CRITICAL UI/UX DESIGN & INTERACTION RULES:
1. PREMIUM COLOR PALETTES & THEMES: Avoid raw default colors or harsh blue schemes. Use sophisticated, modern design tones:
   - For Dark Themes: Use sleek deep obsidian/slate backgrounds (`bg-gray-900`, `bg-slate-900`) combined with subtle borders (`border-slate-800`), glowing accent button shadows, and soft text structures.
   - For Light Themes: Use crisp, elegant cream/off-white canvas backdrops (`bg-slate-50`) paired with charcoal typography (`text-slate-800`) to maximize contrast and reduce eye strain.
2. COMPONENT SHADOWS, RADII, & HOVER STATES: 
   - Every single product card, layout panel, modal popup, and form input container MUST feature modern subtle corner curvatures (`rounded-xl` or `rounded-2xl`) and drop-shadow elevations (`shadow-sm`, `shadow-xl`).
   - Clickable buttons and links must have explicit transition rules (`transition-all duration-300 transform hover:-translate-y-0.5 hover:shadow-lg`) to provide satisfying tactile user feedback.
3. PROFESSIONAL LAYOUT GRIDS & SPACING: Never pack elements tightly together. Incorporate spacious grid containers (`gap-6`, `gap-8`) and clean page paddings (`py-12 px-6`). Major dashboards or landing page flows must use elegant bounding max-widths (`max-w-6xl mx-auto`).
4. DIGITAL SCREEN DISPLAYS & HOUSINGS: If building tools like calculators, track charts, or metric monitors, wrap the app framework neatly inside a centered physical device chassis card. Dedicate a large, high-contrast, beautiful display screen block right at the top featuring sleek, dimmed background text layout fields.
5. COMPLETE BUTTON FUNCTIONALITY: All navigation options, sub-menus, search filter inputs, and forms must perform active DOM state updates using clean vanilla JavaScript logic. Ensure adding items appends beautifully formatted rows or increments badge numbers with smooth, noticeable animation updates.
6. COMPLIANT MEDIA ENGINES: Use real, high-resolution Unsplash photo links matched contextually to the requested topic. Protect all asset proportions from warping using strict container clipping parameters (`object-fit: cover; width: 100%; height: 100%;`).

You MUST format your response exactly like this template, keeping the delimiter tags on separate lines:
###HTML###
(Your complete, premium indented HTML semantic layout markup code here)
###CSS###
(Your clean supplementary custom CSS styling rules overrides here)
###JS###
(Your complete global data state objects, mobile menu triggers, view routing, and interaction handlers here)
###END###

CRITICAL SYSTEM RULES:
1. DO NOT include any introductory text, markdown formatting, explanations, or surrounding commentary.
2. DO NOT use markdown code block wrappers (like ```html or ```css) anywhere in your output.
3. DO NOT minify code into a single line. The output must be written clean, line-by-line, and beautifully indented."""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

chain = prompt_template | llm | CodeOutputParser()

# Fixed index route that dynamically resolves absolute directory paths to fix "Not Found" errors
@app.route('/')
def index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.abspath(os.path.join(base_dir, '..', 'frontend'))
    return send_from_directory(frontend_dir, 'index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    user_prompt = data.get('prompt', '').strip()
    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        parsed = chain.invoke({"input": user_prompt})
        return jsonify(parsed)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
