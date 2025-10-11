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
# ✅ APP SETUP
# -------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- Short-term memory containers (RAM only) ----
memory_vectors = []
memory_texts = []


# -------------------------------------------------------------------
# ✅ UTILITIES
# -------------------------------------------------------------------
def embed_text(text: str) -> np.ndarray:
    """Generate an embedding vector for the given text."""
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
    """Extract readable text from multiple document formats."""
    name = file.filename.lower()

    # PDF
    if name.endswith(".pdf"):
        try:
            reader = PdfReader(file)
            return "\n".join([p.extract_text() or "" for p in reader.pages])
        except Exception:
            return ""

    # DOCX
    elif name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])

    # EXCEL / CSV
    elif name.endswith((".xls", ".xlsx", ".csv")):
        try:
            df = pd.read_excel(file) if not name.endswith(".csv") else pd.read_csv(file)
            return df.to_string()
        except Exception:
            return ""

    # POWERPOINT
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

    # HTML
    elif name.endswith((".html", ".htm")):
        try:
            soup = BeautifulSoup(file.read(), "html.parser")
            return soup.get_text()
        except Exception:
            return ""

    # Plain text / other
    else:
        try:
            return file.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""


# -------------------------------------------------------------------
# ✅ ROUTES
# -------------------------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload_files():
    """Upload and embed files into short-term memory."""
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
            memory_texts.append(text[:4000])  # larger context window
            count += 1

    return jsonify({"message": f"✅ Uploaded {count} file(s) to short-term memory."})


@app.route("/ask", methods=["POST"])
def ask_question():
    """Answer questions based on short-term memory context."""
    global memory_vectors, memory_texts
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided."}), 400

    if not memory_vectors:
        return jsonify({"answer": "Memory is empty. Please upload files first."})

    # ---- find relevant context ----
    q_emb = embed_text(question)
    index = faiss.IndexFlatL2(len(q_emb))
    index.add(np.stack(memory_vectors))
    _, I = index.search(np.array([q_emb]), k=min(3, len(memory_vectors)))

    context = "\n\n".join([memory_texts[i] for i in I[0]])

    # ---- professional long-form prompt ----
    prompt = f"""
You are an AI research and analysis assistant powered by GPT-5.
Your task is to read and interpret the uploaded document excerpts below.

Base your answer strictly on the given content—do not invent details.
Organize the response as a **detailed 1–2 page analytical report** with the following qualities:
- Clear introduction and context summary
- Well-structured explanation of the key concepts
- Examples or evidence drawn directly from the material
- Logical transitions and concise professional language
- Factual accuracy with no assumptions beyond the context

Context:
{context}

User Question:
{question}

Write a full, formal report-style answer that could be shown to executives or trainees.
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-5",  # upgraded model
            messages=[
                {"role": "system",
                 "content": "You are a precise, factual, and professional AI analyst."},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=1500  # extended for 1–2 page report
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        answer = f"⚠️ Error: {e}"

    return jsonify({"answer": answer})


@app.route("/reset", methods=["POST"])
def reset_memory():
    """Clear short-term memory."""
    global memory_vectors, memory_texts
    memory_vectors.clear()
    memory_texts.clear()
    return jsonify({"message": "♻️ Short-term memory cleared."})


# -------------------------------------------------------------------
# ✅ MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
