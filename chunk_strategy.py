import re

def normalize_for_search(text: str) -> str:

    if text is None:
        return ""

    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 350):
    
    chunks =[]
    start =0
    text = text.strip()
    text_length=len(text)

    while start < text_length:
        end= start + chunk_size
        chunk=text[start:end].strip()

        if chunk:
            chunks.append(chunk)
        
        start += chunk_size - overlap

    return chunks

def chunk_pages(pages):
    all_chunks = []

    for page in pages:
        text = page.get("text", "")
        page_number = page.get("page_number", None)
        text = normalize_for_search(text)
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "page_number": page_number,
                "chunk_id": f'{page_number}_chunk_{i+1}',
                "text": chunk
            })

    return all_chunks





