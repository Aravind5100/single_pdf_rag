import requests
from index_strategy import hybrid_retrieve_indices, retrieve_chunks

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.1:8b"

SYSTEM_PROMPT = """
You are an assistant that answers questions using ONLY the provided context.

Rules:
1) Use only the information in the context.
2) You MAY summarize or infer descriptive information if it is clearly supported by the context.
   (Example: for narrative text, describe a character based on actions/dialogue shown.)
3) Do NOT invent facts not supported or implied by the context.
4) If the context does not support the answer, say:
   "I don’t have enough information in the provided document to answer this question."
5) Cite page numbers for supporting passages (e.g., [Page 31]).
6) Be concise and factual.
""".strip()


def format_context(retrieved_chunks: list[dict]) -> str:
    #Formats retrieved chunks into a single context string for the prompt.

    blocks = []
    for ch in retrieved_chunks:
        blocks.append(f"[Page {ch['page_number']}]\n{ch['text']}")
    return "\n\n".join(blocks)


def build_prompt(context: str, question: str) -> str:

    return f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}

Answer:
"""


def generate_answer(prompt: str) -> str:
    """
    Calls Ollama LLM to generate an answer from the prompt.
    """
    resp = requests.post(
        OLLAMA_GENERATE_URL,
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
        timeout=120
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()

def answer_question(question: str, index, embedded_chunks: list[dict], top_k: int = 5) -> dict:

    indices = hybrid_retrieve_indices(index, embedded_chunks, question, top_k=top_k)
    retrieved = retrieve_chunks(embedded_chunks, indices)
    
    context = format_context(retrieved)

    # print("\n=== RETRIEVED CHUNKS (PREVIEW) ===")
    # for ch in retrieved:
    #     print(f"Page {ch['page_number']} | {ch['chunk_id']} | {ch['text'][:120].replace('\n',' ')}")
    # print("=================================\n")

    prompt = build_prompt(context, question)
    answer = generate_answer(prompt)


    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": retrieved,
        "indices": indices
    }
