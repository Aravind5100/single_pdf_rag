from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss


CACHE_DIR = Path("cache")


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cache_key(pdf_path: str, *, chunk_size: int, overlap: int, embed_model: str) -> str:
    """Unique signature for (PDF content + chunking params + embedding model)."""
    pdf_hash = _file_sha256(pdf_path)
    meta = {
        "pdf_sha256": pdf_hash,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embed_model": embed_model,
    }
    raw = json.dumps(meta, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def cache_paths(cache_key: str) -> Dict[str, Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "meta": CACHE_DIR / f"{cache_key}.meta.json",
        "chunks": CACHE_DIR / f"{cache_key}.embedded_chunks.jsonl",
        "index": CACHE_DIR / f"{cache_key}.faiss.index",
    }


def save_embedded_chunks(path: Path, embedded_chunks: List[Dict[str, Any]]) -> None:
    # Store only what we need; keep embeddings as lists
    with path.open("w", encoding="utf-8") as f:
        for ch in embedded_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")


def load_embedded_chunks(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_faiss_index(path: Path, index) -> None:
    faiss.write_index(index, str(path))


def load_faiss_index(path: Path):
    return faiss.read_index(str(path))


def save_meta(path: Path, meta: Dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load_meta(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cache_exists(paths: Dict[str, Path]) -> bool:
    return paths["meta"].exists() and paths["chunks"].exists() and paths["index"].exists()
