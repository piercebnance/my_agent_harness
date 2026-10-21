from src.harness.rag import rrf

def test_rrf_prefers_docs_ranked_high_in_both_lists():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert set(fused[:2]) == {"a", "b"}
    assert fused[-1] in {"c", "d"}

def test_rrf_handles_empty_list():
    assert rrf([["a", "b"], []]) == ["a", "b"]