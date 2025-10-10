from flask import Flask, request, jsonify
from flask_cors import CORS
import os, numpy as np
from openai import OpenAI
import faiss
from PyPDF2 import PdfReader
import docx

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # ✅ allow all domains

# ---------- SETUP ----------
app = Flask(__name__)
CORS(app)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ Missing OPENAI_API_KEY. Set it in Render Environment Variables.")

client = OpenAI(api_key=api_key)

# ---------- UTILITIES ----------
def extract_text(file):
    name = file.filename
    if name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return ""

def make_embeddings(texts):
    chunks, chunk_size = [], 500
    for t in texts:
        words = t.split()
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i+chunk_size]))
    embeds = client.embeddings.create(model="text-embedding-3-small", input=chunks)
    vectors = [d.embedding for d in embeds.data]
    return chunks, np.array(vectors).astype("float32")

def ask_gpt(question, context):
    prompt = f"Answer using this information:\n{context}\n\nQuestion: {question}"
    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return chat.choices[0].message.content

# ---------- MEMORY ----------
temp_index = faiss.IndexFlatL2(1536)
master_index = faiss.IndexFlatL2(1536)
temp_chunks, master_chunks = [], []

# ---------- ROUTES ----------
@app.route("/")
def home():
    return "✅ File Agent backend running successfully!"

@app.route("/upload", methods=["POST"])
def upload():
    keep = request.form.get("keep") == "true"
    files = request.files.getlist("files")
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

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question")
    q_vec = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding
    q_vec = np.array([q_vec]).astype("float32")

    def top_context(index, chunks):
        if index.ntotal == 0: return []
        D, I = index.search(q_vec, k=min(3, index.ntotal))
        return [chunks[i] for i in I[0] if i < len(chunks)]

    ctx = "\n".join(top_context(temp_index, temp_chunks) + top_context(master_index, master_chunks))
    if not ctx.strip():
        return jsonify({"answer": "No relevant context found. Try uploading files first."})
    answer = ask_gpt(question, ctx)
    return jsonify({"answer": answer})

@app.route("/reset", methods=["POST"])
def reset_memory():
    global temp_index, temp_chunks
    temp_index = faiss.IndexFlatL2(1536)
    temp_chunks = []
    return jsonify({"message": "♻️ Short-term memory cleared."})

# ---------- MAIN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
