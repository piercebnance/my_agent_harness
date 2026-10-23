from src.harness.tools import calculator, run_tool
from unittest.mock import patch
from src.harness import tools

LONG_TEXT = "Python is a language. " * 50

def test_read_webpage_ingests_full_text_and_returns_preview():
    with patch.object(tools, "_is_safe_url", return_value=True), \
         patch.object(tools, "_fetch_clean", return_value=("T", LONG_TEXT)), \
         patch.object(tools.rag, "ingest", return_value=3) as ingest:
        out = tools.read_webpage("https://example.com", max_chars=100)
    ingest.assert_called_once()
    assert "Saved 3 chunks" in out and "[preview truncated]" in out

def test_read_webpage_survives_ingest_failure():
    with patch.object(tools, "_is_safe_url", return_value=True), \
         patch.object(tools, "_fetch_clean", return_value=("T", LONG_TEXT)), \
         patch.object(tools.rag, "ingest", side_effect=RuntimeError("boom")):
        out = tools.read_webpage("https://example.com")
    assert "Title: T" in out

def test_calculator():
    assert calculator("2+2") == "4"

def test_calculator_rejects_bad_input():
    assert "Error" in calculator("__import__('os')")

def test_unknown_tool_returns_error():
    assert "Error" in run_tool("nope", {})

def test_search_knowledge_uses_package_relative_rag(monkeypatch):
    monkeypatch.setattr(tools.rag, "search_knowledge", lambda query, k: f"{query}:{k}")
    assert run_tool("search_knowledge", {"query": "test", "k": 3}) == "test:3"

def test_web_search_reports_ddgs_failure_without_traceback(monkeypatch, capsys):
    from ddgs.exceptions import DDGSException

    class FailingDDGS:
        def text(self, query, max_results):
            raise DDGSException("No results found.")

    monkeypatch.setattr(tools, "DDGS", FailingDDGS)
    result = run_tool("web_search", {"query": "test"})

    assert "Web search failed: No results found." in result
    assert capsys.readouterr().err == ""