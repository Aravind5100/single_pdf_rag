import fitz # PyMuPDF

def extract_text_from_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")
        text = text.replace("\u00a0", " ").strip()

        pages.append({
            "page_number": page_index + 1,
            "text": text,
        })

    return pages

