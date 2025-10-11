@app.route("/ask", methods=["POST"])
def ask_question():
    """Answer concisely based on short-term memory context."""
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
    _, I = index.search(np.array([q_emb]), k=min(5, len(memory_vectors)))
    context = "\n\n".join([memory_texts[i] for i in I[0]])

    # ---- concise professional prompt ----
    prompt = f"""
You are a precise and factual AI assistant.
Base your answer strictly on the provided context — do not invent information.

Context:
{context}

User Question:
{question}

Provide a concise and direct answer (2–5 sentences maximum) that fully addresses the question.
Avoid extra commentary or section headings.
"""

    model_order = ["gpt-5", "gpt-4o", "gpt-4o-mini"]
    last_err = None
    for m in model_order:
        try:
            completion = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system",
                     "content": "You are a concise, professional assistant that answers in 2–5 sentences."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=250,   # shorter output
                temperature=0.2   # low creativity = crisp answers
            )
            answer = completion.choices[0].message.content
            return jsonify({"answer": answer, "model_used": m})
        except Exception as e:
            last_err = str(e)
            continue

    return jsonify({"answer": f"Error generating answer: {last_err}"}), 500
