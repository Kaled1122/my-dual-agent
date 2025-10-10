import os, numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import faiss
from PyPDF2 import PdfReader
import docx
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import pandas as pd
from bs4 import BeautifulSoup
import requests
from pptx import Presentation

# ---------- SETUP ----------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ Missing OPENAI_API_KEY. Set it in Render Environment Variables.")

client = OpenAI(api_key=api_key)

# ---------- UTILITIES ----------
def extract_text(file):
    """Extract readable text or transcriptions from all supported file types."""
    name = file.filename.lower()

    # ---- PDF ----
    if name.endswith(".pdf"):
        try:
            reader = PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception:
            text = ""
        if not text.strip():
            file.seek(0)
            images = convert_from_bytes(file.read())
            text = "\n".join([pytesseract.image_to_string(img) for img in images])
        return text

    # ---- Word ----
    elif name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])

    # ---- Text / URL ----
    elif name.endswith(".txt"):
        text = file.read().decode("utf-8").strip()
        if text.startswith("http://") or text.startswith("https://"):
            try:
                html = requests.get(text).text
                soup = BeautifulSoup(html, "html.parser")
                return soup.get_text(separator="\n", strip=True)
            except Exception as e:
                return f"[Error loading web page: {e}]"
        return text

    # ---- Images ----
    elif name.endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(file)
        return pytesseract.image_to_string(img)

    # ---- Excel / CSV ----
    elif name.endswith((".xlsx", ".xls", ".csv")):
        try:
            if name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            text = "\n".join(df.astype(str).fillna("").values.flatten())
            return text
        except Exception as e:
            return f"[Error reading Excel/CSV: {e}]"

    # ---- HTML ----
    elif name.endswith((".html", ".htm")):
        try:
            soup = BeautifulSoup(file.read(), "html.parser")
            return soup.get_text(separator="\n", strip=True)
        except Exception as e:
            return f"[Error reading HTML: {e}]"

    # ---- PowerPoint ----
    elif name.endswith(".pptx"):
        try:
            prs = Presentation(file)
            text_runs = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text_runs.append(shape.text)
            return "\n".join(text_runs)
        except Exception as e:
            return f"[Error reading PPTX: {e}]"

    # ---- Audio ----
    elif name.endswith((".mp3", ".wav", ".m4a")):
        file.seek(0)
        temp_path = "temp_audio.mp3"
        with open(temp_path, "wb") as f:
            f.write(file.read())
        with open(temp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f
            )
        os.remove(temp_path)
        return transcript.text

    return ""

def make_embeddings(texts):
    """Convert texts into embeddings."""
    chunks, chunk_size = [], 500
    for t in texts:
        if isinstance(t, str) and t.strip():
            words = t.split()
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size]).strip()
                if chunk:
                    chunks.append(chunk)

    if not chunks:
        raise ValueError("No valid text found to embed.")

    embeds = client.embeddings.create(model="text-embedding-3-small", input=chunks)
    vectors = [d.embedding for d in embeds.data]
    return chunks, np.array(vectors).astype("float32")

def ask_gpt(question, context):
    if not context.strip():
        return "⚠️ No relevant information found. Try uploading documents first."

    prompt = f"""
You are a precise assistant that answers **only** using the information inside <context> tags.
If the answer cannot be found in the context, reply exactly:
"I couldn’t find that information in the uploaded files."

<context>
{context}
</context>

Question: {question}
"""
    chat = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}]
    )
    return chat.choices[0].message.content.strip()

# ---------- MEMORY ----------
temp_index = faiss.IndexFlatL2(1536)
master_index = faiss.IndexFlatL2(1536)
temp_chunks, master_chunks = [], []

# new metadata trackers
master_files = []  # [{"name": ..., "timestamp": ..., "count": ...}, ...]
temp_files = []

# ---------- ROUTES ----------
@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "✅ File Agent backend running successfully!"})

@app.route("/upload", methods=["POST"])
def upload():
    try:
        keep = request.form.get("keep") == "true"
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "No files uploaded"}), 400

        texts = [extract_text(f) for f in files]
        chunks, vectors = make_embeddings(texts)

        if keep:
            master_index.add(vectors)
            master_chunks.extend(chunks)
            return jsonify({"message": "📚 Added to long-term memory."})
        else:
            temp_index.add(vectors)
            temp_chunks.extend(chunks)
            return jsonify({"message": "🧠 Added to short-term memory."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True)
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "Missing question"}), 400

        embed_resp = client.embeddings.create(model="text-embedding-3-small", input=[question])
        q_vec = np.array([embed_resp.data[0].embedding]).astype("float32")

        def top_context(index, chunks):
            if index.ntotal == 0:
                return []
            D, I = index.search(q_vec, k=min(3, index.ntotal))
            return [chunks[i] for i in I[0] if i < len(chunks)]

        ctx = "\n".join(top_context(temp_index, temp_chunks) + top_context(master_index, master_chunks))
        if not ctx.strip():
            return jsonify({"answer": "No relevant context found. Try uploading files first."})

        answer = ask_gpt(question, ctx)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset_memory():
    global temp_index, temp_chunks
    temp_index = faiss.IndexFlatL2(1536)
    temp_chunks = []
    return jsonify({"message": "♻️ Short-term memory cleared."})

@app.route("/reset_longterm", methods=["POST"])
def reset_longterm():
    global master_index, master_chunks
    master_index = faiss.IndexFlatL2(1536)
    master_chunks = []
    master_index_path = "vector_stores/master.index"
    if os.path.exists(master_index_path):
        os.remove(master_index_path)
    return jsonify({"message": "🧹 Long-term memory fully cleared."})

@app.route("/list_longterm", methods=["GET"])
def list_longterm():
    if not master_chunks:
        return jsonify({"files": [], "message": "No long-term files found."})
    preview = [chunk[:120] + "..." if len(chunk) > 120 else chunk for chunk in master_chunks]
    return jsonify({"count": len(master_chunks), "previews": preview[:30]})

# ---------- MAIN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
