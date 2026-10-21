import hashlib
from pathlib import Path
import chromadb
import re
from rank_bm25 import BM25Okapi
from collections import Counter
from pypdf import PdfReader

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

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None

def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder  # lazy: slow import
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker

def rerank(query: str, ids: list[str], docs: list[str], top_k: int) -> list[str]:
    """Return the ids of the top_k docs, best first, scored against the query."""
    if not ids:
        return []
    scores = _get_reranker().predict([(query, d) for d in docs])
    ranked = sorted(zip(ids, scores), key=lambda pair: pair[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:top_k]]

def ingest(source: str, text: str) -> int:
    col.delete(where={"source": source})   # drop stale chunks from earlier versions
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


SUPPORTED = {".md", ".txt", ".rst", ".py", ".pdf"}
IGNORED_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".rag_store"}

def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def ingest_path(path: str) -> dict:
    """Ingest one file or every supported file under a folder."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Not found: {path}")

    if p.is_file():
        files = [p]
    else:
        files = [
            f for f in p.rglob("*")
            if f.is_file()
            and f.suffix.lower() in SUPPORTED
            and not any(part in IGNORED_DIRS for part in f.relative_to(p).parts)
        ]

    result = {"files": 0, "chunks": 0, "skipped": []}
    for f in files:
        if f.suffix.lower() not in SUPPORTED:
            result["skipped"].append(f"{f.name} (unsupported type)")
            continue
        try:
            text = read_file(f)
        except Exception as e:
            result["skipped"].append(f"{f.name} ({e})")
            continue
        if not text.strip():
            result["skipped"].append(f"{f.name} (no extractable text, maybe a scanned PDF)")
            continue
        result["chunks"] += ingest(str(f), text)
        result["files"] += 1
    return result


def list_sources() -> list[tuple[str, int]]:
    """Return (source, chunk_count) pairs, sorted by source."""
    data = col.get(include=["metadatas"])
    counts = Counter(m["source"] for m in data["metadatas"])
    return sorted(counts.items())


def list_sources_text() -> str:
    sources = list_sources()
    if not sources:
        return "The knowledge base is empty."
    lines = [f"- {src} ({n} chunks)" for src, n in sources]
    return f"{len(sources)} sources in the knowledge base:\n" + "\n".join(lines)


def forget(source: str) -> None:
    global _bm25
    col.delete(where={"source": source})
    _bm25 = None

def search_knowledge(query: str, k: int = 5, candidates: int = 20,
                     use_rerank: bool = True) -> str:
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

    # 3) fuse, keeping the full candidate pool for the reranker
    fused = rrf([vec_ids, bm25_ids])[:candidates]
    got = col.get(ids=fused, include=["documents", "metadatas"])
    by_id = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}
    fused = [i for i in fused if i in by_id]

    # 4) rerank (or just truncate)
    if use_rerank:
        final = rerank(query, fused, [by_id[i][0] for i in fused], top_k=k)
    else:
        final = fused[:k]

    return "\n\n".join(
        f"[{by_id[i][1]['source']} #{by_id[i][1]['chunk']}]\n{by_id[i][0]}"
        for i in final
    )