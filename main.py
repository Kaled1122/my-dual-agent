import os, numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import faiss
from PyPDF2 import PdfReader
import docx

# ---------- SETUP ----------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ Missing OPENAI_API_KEY. Set it in Render Environment Variables.")

client = OpenAI(api_key=api_key)

# ---------- UTILITIES ----------
def extract_text(file):
    name = file.filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif name.endswith(".txt"):
        return file.read().decode("utf-8")
    return ""

def make_embeddings(texts):
    # Clean and chunk text
    chunks, chunk_size = [], 500
    for t in texts:
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not t:
            continue
        words = t.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size]).strip()
            if chunk:
                chunks.append(chunk)

    if not chunks:
        raise ValueError("No valid text found to embed.")

    # ✅ Always send a list of strings
    embeds = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )
    vectors = [d.embedding for d in embeds.data]
    return chunks, np.array(vectors).astype("float32")

def ask_gpt(question, context):
    # If there’s no context at all, skip the API call
    if not context.strip():
        return "⚠️ No relevant information found in memory. Please upload documents first."

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

        # ✅ Always wrap in list
        embed_resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=[question]
        )
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


# ---------- NEW FEATURES ----------

@app.route("/reset_longterm", methods=["POST"])
def reset_longterm():
    """Completely clear long-term memory."""
    global master_index, master_chunks
    master_index = faiss.IndexFlatL2(1536)
    master_chunks = []
    master_index_path = "vector_stores/master.index"
    if os.path.exists(master_index_path):
        os.remove(master_index_path)
    return jsonify({"message": "🧹 Long-term memory fully cleared."})


@app.route("/list_longterm", methods=["GET"])
def list_longterm():
    """Preview stored long-term memory contents."""
    if not master_chunks:
        return jsonify({"files": [], "message": "No long-term files found."})
    preview = [chunk[:120] + "..." if len(chunk) > 120 else chunk for chunk in master_chunks]
    return jsonify({"count": len(master_chunks), "previews": preview[:30]})


# ---------- MAIN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
