from src.harness.rag import rrf
from src.harness import rag

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