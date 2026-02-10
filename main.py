import os
import sys
from extract_pdf_text import extract_text_from_pdf
from chunk_strategy import chunk_pages
from embedding_strategy import embed_chunks
from index_strategy import build_faiss_index #debug_find_term
from generation_strategy import answer_question

pdf_path = "/Users/aravindkompalli/Downloads/sample_university_handbook.pdf"
pages = extract_text_from_pdf(pdf_path)

#print("Total pages:", len(pages))
#print("\n--- Page preview ---\n")
#print(pages[5]["text"][:])

chunks = chunk_pages(pages)

# print("Total chunks:", len(chunks))
# print("\n--- Chunk preview ---\n")
# #print(chunks[750]["text"][:])
# print("Page:", chunks[633]["page_number"])
# print("ID:", chunks[633]["chunk_id"])


# print("chunk 4:", chunks[633]["text"][:],"\n")
# print("chunk 5:", chunks[634]["text"][:],"\n")
# print("chunk 6:", chunks[635]["text"][:],"\n")

embedded_chunks = embed_chunks(chunks)
#debug_find_term(embedded_chunks, "atlas", max_hits=10)


#lengths = set(len(chunk["embedding"]) for chunk in embedded_chunks)
#print("Unique embedding lengths:", lengths)

#print("Embedding length:", len(embedded_chunks[0]["embedding"]))
#print("First 5 numbers:", embedded_chunks[0]["embedding"][:5])

index, vectors = build_faiss_index(embedded_chunks)

#query = "There is no such thing as bad people. We're all just people who sometimes do bad things."

#query_vec = embed_query(query)
#distances, indices = search_index(index, query_vec, top_k=3)

#results = retrieve_chunks(embedded_chunks, indices)

#for r in results:
#    print(f"[Page {r['page_number']}]")
#    print(r["text"][:])
#    print("-----")

print("\nHi Ark! How Can I Help You? (type 'exit' to quit)\n")

while True:
   question = input("Q: ").strip()
   if question.lower() in {"exit", "quit"}:
       break

   result = answer_question(question, index, embedded_chunks, top_k=10)

   print("\nA:", result["answer"])
   print("\nCitations:")
   for ch in result["retrieved_chunks"]:
       print(f"Page {ch['page_number']} | {ch['chunk_id']}")
   print()