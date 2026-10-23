from src.harness.rag import rrf
from src.harness import rag

class ChunkCol:
    def __init__(self, text):
        chunks = rag.chunk_text(text)
        self.docs = list(reversed(chunks))                       # out of order on purpose
        self.metas = [{"source": "x.pdf", "chunk": i} for i in reversed(range(len(chunks)))]
    def get(self, where=None, include=None):
        return {"documents": self.docs, "metadatas": self.metas}

def test_get_document_text_roundtrip(monkeypatch):
    text = "".join(str(i % 10) for i in range(2000))
    monkeypatch.setattr(rag, "col", ChunkCol(text))
    assert rag.get_document_text("x.pdf") == text

def test_resolve_source_by_filename(monkeypatch):
    monkeypatch.setattr(rag, "list_sources", lambda: [("C:/docs/report.pdf", 5), ("C:/docs/notes.md", 2)])
    assert rag.resolve_source("report.pdf") == ("C:/docs/report.pdf", [])

def test_resolve_source_ambiguous(monkeypatch):
    monkeypatch.setattr(rag, "list_sources", lambda: [("a/report.pdf", 1), ("b/report.pdf", 1)])
    source, matches = rag.resolve_source("report")
    assert source is None and len(matches) == 2

def test_read_source_text_pages(monkeypatch):
    monkeypatch.setattr(rag, "resolve_source", lambda n: ("x.pdf", []))
    monkeypatch.setattr(rag, "get_document_text", lambda s: "a" * 20000)
    out = rag.read_source_text("x.pdf")
    assert "offset=8000" in out
    assert "End of document" in rag.read_source_text("x.pdf", offset=16000)

class FakeReranker:
    """Scores a doc by how many query words it contains."""
    def predict(self, pairs):
        return [sum(w in doc.lower() for w in q.lower().split()) for q, doc in pairs]

def test_rerank_orders_by_score(monkeypatch):
    monkeypatch.setattr(rag, "_get_reranker", lambda: FakeReranker())
    ids = ["a", "b", "c"]
    docs = ["unrelated text", "python tools for agents", "python"]
    assert rag.rerank("python agents", ids, docs, top_k=2) == ["b", "c"]

def test_rerank_empty_input():
    assert rag.rerank("anything", [], [], top_k=5) == []

def test_rrf_prefers_docs_ranked_high_in_both_lists():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert set(fused[:2]) == {"a", "b"}
    assert fused[-1] in {"c", "d"}

def test_rrf_handles_empty_list():
    assert rrf([["a", "b"], []]) == ["a", "b"]