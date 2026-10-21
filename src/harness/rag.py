import hashlib
from pathlib import Path
import chromadb
import re
from rank_bm25 import BM25Okapi

client = chromadb.PersistentClient(path=".rag_store")   # add to .gitignore
col = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})

def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


_bm25 = None
_bm25_ids: list[str] = []

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())

def _build_bm25() -> None:
    global _bm25, _bm25_ids
    data = col.get()
    _bm25_ids = data["ids"]
    _bm25 = BM25Okapi([_tokenize(d) for d in data["documents"]]) if _bm25_ids else None



def ingest(source: str, text: str) -> int:
    chunks = chunk_text(text)
    ids = [hashlib.md5(f"{source}:{i}:{c}".encode()).hexdigest() for i, c in enumerate(chunks)]
    col.upsert(
        ids=ids,
        documents=chunks,
        metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))],
    )
    global _bm25
    _bm25 = None   # rebuilt lazily on next search
    return len(chunks)

def ingest_folder(folder: str) -> None:
    for p in Path(folder).rglob("*"):
        if p.suffix in {".md", ".txt"}:
            ingest(str(p), p.read_text(encoding="utf-8", errors="ignore"))

def search_knowledge(query: str, k: int = 5, candidates: int = 20) -> str:
    total = col.count()
    if total == 0:
        return "The knowledge base is empty."

    # 1) vector ranking
    vec = col.query(query_texts=[query], n_results=min(candidates, total))
    vec_ids = vec["ids"][0]

    # 2) keyword ranking
    if _bm25 is None:
        _build_bm25()
    scores = _bm25.get_scores(_tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:candidates]
    bm25_ids = [_bm25_ids[i] for i in top if scores[i] > 0]

    # 3) fuse and fetch the winners
    fused = rrf([vec_ids, bm25_ids])[:k]
    got = col.get(ids=fused, include=["documents", "metadatas"])
    by_id = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}

    return "\n\n".join(
        f"[{by_id[i][1]['source']} #{by_id[i][1]['chunk']}]\n{by_id[i][0]}"
        for i in fused if i in by_id
    )