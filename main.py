import os, numpy as np
from openai import OpenAI
import faiss
from PyPDF2 import PdfReader
import docx

client = OpenAI(api_key=os.getenv("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"))

# ----------  UTILITIES  ----------

def extract_text(filepath):
    if filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif filepath.endswith(".docx"):
        doc = docx.Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs])
    elif filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return ""

def make_embeddings(texts):
    # Breaks large texts into chunks of ~500 words
    chunks, chunk_size = [], 500
    for t in texts:
        words = t.split()
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i+chunk_size]))
    embeds = client.embeddings.create(model="text-embedding-3-small", input=chunks)
    vectors = [d.embedding for d in embeds.data]
    return chunks, np.array(vectors).astype("float32")

def ask_gpt(question, context):
    prompt = f"Answer using the information below:\n{context}\n\nQuestion: {question}"
    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return chat.choices[0].message.content

# ----------  MEMORY SETUP  ----------

temp_index_path = "vector_stores/temp.index"
master_index_path = "vector_stores/master.index"

def load_index(path):
    return faiss.read_index(path) if os.path.exists(path) else faiss.IndexFlatL2(1536)

temp_index = load_index(temp_index_path)
master_index = load_index(master_index_path)
temp_chunks, master_chunks = [], []

# ----------  CORE FUNCTIONS  ----------

def upload(folder, keep=False):
    global temp_index, master_index, temp_chunks, master_chunks
    texts = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        texts.append(extract_text(path))
        print(f"✅ Loaded {file}")
    chunks, vectors = make_embeddings(texts)
    if keep:
        master_index.add(vectors)
        master_chunks.extend(chunks)
        faiss.write_index(master_index, master_index_path)
        print("📚 Added to long-term memory.")
    else:
        temp_index.add(vectors)
        temp_chunks.extend(chunks)
        faiss.write_index(temp_index, temp_index_path)
        print("🧠 Added to short-term memory.")

def reset_temp():
    global temp_index, temp_chunks
    temp_index = faiss.IndexFlatL2(1536)
    temp_chunks = []
    if os.path.exists(temp_index_path):
        os.remove(temp_index_path)
    print("♻️  Short-term memory cleared.")

def query(question):
    global temp_index, master_index, temp_chunks, master_chunks
    q_vec = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding
    q_vec = np.array([q_vec]).astype("float32")

    def top_context(index, chunks):
        if index.ntotal == 0: return []
        D, I = index.search(q_vec, k=min(3, index.ntotal))
        return [chunks[i] for i in I[0] if i < len(chunks)]

    ctx = "\n".join(top_context(temp_index, temp_chunks) + top_context(master_index, master_chunks))
    answer = ask_gpt(question, ctx)
    print("\n💬 Answer:\n", answer, "\n")

# ----------  SIMPLE COMMAND LOOP  ----------

def menu():
    print("""
======== FILE AGENT ========
1. Upload current batch (short-term)
2. Upload and keep permanently (long-term)
3. Ask a question
4. Reset short-term memory
5. Exit
============================
""")

while True:
    menu()
    choice = input("Select option: ").strip()
    if choice == "1":
        upload("uploads", keep=False)
    elif choice == "2":
        upload("uploads", keep=True)
    elif choice == "3":
        q = input("Ask: ")
        query(q)
    elif choice == "4":
        reset_temp()
    elif choice == "5":
        print("Goodbye 👋")
        break
    else:
        print("Invalid option.")
