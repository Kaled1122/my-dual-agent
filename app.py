import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import faiss, numpy as np
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
from pptx import Presentation
from bs4 import BeautifulSoup

# -------------------------------------------------------------------
# APP SETUP
# -------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Short-term memory (RAM only)
memory_vectors = []
memory_texts = []

# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------
def embed_text(text: str) -> np.ndarray:
    try:
        emb = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        ).data[0].embedding
        return np.array(emb, dtype="float32")
    except Exception as e:
        print("Embedding error:", e)
        return np.zeros(1536, dtype="float32")

def extract_text(file):
    name = file.filename.lower()

    if name.endswith(".pdf"):
        try:
            reader = PdfReader(file)
            return "\n".join([p.extract_text() or "" for p in reader.pages])
        except Exception:
            return ""

    elif name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])

    elif name.endswith((".xls", ".xlsx", ".csv")):
        try:
            df = pd.read_excel(file) if not name.endswith(".csv") else pd.read_csv(file)
            return df.to_string()
        except Exception:
            return ""

    elif name.endswith(".pptx"):
        try:
            prs = Presentation(file)
            texts = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        texts.append(shape.text)
            return "\n".join(texts)
        except Exception:
            return ""

    elif name.endswith((".html", ".htm")):
        try:
            soup = BeautifulSoup(file.read(), "html.parser")
            return soup.get_text()
        except Exception:
            return ""

    else:
        try:
            return file.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""

# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload_files():
    global memory_vectors, memory_texts
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded."}), 400

    count = 0
    for f in files:
        text = extract_text(f)
        if text.strip():
            emb = embed_text(text)
            memory_vectors.append(emb)
            memory_texts.append(text[:4000])  # larger snippet for richer context
            count += 1

    return jsonify({"message": f"Uploaded {count} file(s) to short-term memory."})

@app.route("/ask", methods=["POST"])
def ask_question():
    global memory_vectors, memory_texts
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided."}), 400

    if not memory_vectors:
        return jsonify({"answer": "Memory is empty. Please upload files first."})

    # Retrieve top-k relevant snippets
    q_emb = embed_text(question)
    index = faiss.IndexFlatL2(len(q_emb))
    index.add(np.stack(memory_vectors))
    _, I = index.search(np.array([q_emb]), k=min(3, len(memory_vectors)))
    context = "\n\n".join([memory_texts[i] for i in I[0]])

    # Long-form report prompt
    prompt = f"""
You are an AI research and analysis assistant.
Base your answer strictly on the provided context—do not invent details.

Write a detailed 1–2 page analytical report with:
- A short introduction summarizing the context
- Clear, logically ordered sections
- Evidence/examples cited directly from the context
- Concise, professional language
- A brief conclusion
If the context lacks required details, state that transparently.

Context:
{context}

User Question:
{question}
"""

    # Try GPT-5; if not available, fall back
    model_order = ["gpt-5", "gpt-4o", "gpt-4o-mini"]
    last_err = None
    for m in model_order:
        try:
            completion = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": "You are precise, factual, and professional."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1500,         # <-- correct param for chat.completions
                temperature=0.3          # tighter, more factual long-form
            )
            answer = completion.choices[0].message.content
            return jsonify({"answer": answer, "model_used": m})
        except Exception as e:
            last_err = str(e)
            continue

    return jsonify({"answer": f"Error generating answer: {last_err}"}), 500

@app.route("/reset", methods=["POST"])
def reset_memory():
    global memory_vectors, memory_texts
    memory_vectors.clear()
    memory_texts.clear()
    return jsonify({"message": "Short-term memory cleared."})

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
