import faiss
import re
import numpy as np
from embedding_strategy import embed_text

def build_faiss_index(embedded_chunks):
    
    vectors = [chunk["embedding"] for chunk in embedded_chunks]
    vectors = np.array(vectors).astype("float32")

    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    return index, vectors

def embed_query(query: str):
    vector = embed_text(query)
    vector = np.array([vector]).astype("float32")
    faiss.normalize_L2(vector)
    return vector

def search_index(index, query_vector, top_k=5):
    distances, indices = index.search(query_vector, top_k)
    return distances[0], indices[0]

def retrieve_chunks(embedded_chunks, indices):
    results = []
    for idx in indices:
        if idx == -1:
            continue
        if 0 <= idx < len(embedded_chunks):
            results.append(embedded_chunks[idx])
    return results

# def debug_find_term(embedded_chunks, term: str, max_hits: int = 10):
#     term_l = term.lower()
#     hits = []
#     for ch in embedded_chunks:
#         if term_l in ch["text"].lower():
#             hits.append(ch)

#     print(f"\nFound {len(hits)} chunks containing '{term}'. Showing up to {max_hits}:\n")
#     for ch in hits[:max_hits]:
#         preview = ch["text"].replace("\n", " ")[:200]
#         print(f"Page {ch['page_number']} | {ch.get('chunk_id','')} | {preview}")

STOPWORDS = {"who", "is", "the", "a", "an", "what", "when", "where", "why", "how", "of", "to", "in", "on", "for", "and"}

def keyword_search_indices(embedded_chunks, query: str, top_k: int = 5):
    q = query.lower().strip()
    terms = [t for t in re.findall(r"[a-zA-Z0-9']+", q) if len(t) >= 3 and t not in STOPWORDS]

    if not terms:
        return []

    scored = []
    for i, ch in enumerate(embedded_chunks):
        text = ch["text"].lower()
        score = sum(text.count(t) for t in terms)
        if score > 0:
            scored.append((score, i))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [i for _, i in scored[:top_k]]

def hybrid_retrieve_indices(index, embedded_chunks, query: str, top_k: int = 5):

    k_keyword = max(2, top_k // 2)
    k_vector = top_k - k_keyword

    k_idx = keyword_search_indices(embedded_chunks, query, top_k=k_keyword)

    qvec = embed_query(query)
    distances, v_idx = search_index(index, qvec, top_k=top_k)
    v_idx = list(v_idx)

    merged = []
    seen = set()

    for idx in k_idx:
        if idx != -1 and idx not in seen:
            merged.append(idx)
            seen.add(idx)

    for idx in v_idx:
        if idx != -1 and idx not in seen:
            merged.append(idx)
            seen.add(idx)
        if len(merged) >= top_k:
            break

    return merged

