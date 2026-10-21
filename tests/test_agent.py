from types import SimpleNamespace as NS
from src.harness import agent
from unittest.mock import patch, MagicMock
from src.harness import tools

def text_block(t): return NS(type="text", text=t)
def tool_block(id, name, input): return NS(type="tool_use", id=id, name=name, input=input)

class FakeClient:
    def __init__(self, responses):
        self._it = iter(responses)
        self.messages = self
    def create(self, **kwargs):
        return next(self._it)

def test_agent_calls_tool_then_answers(monkeypatch):
    fake = FakeClient([
        NS(stop_reason="tool_use",
           content=[tool_block("t1", "calculator", {"expression": "2+2"})]),
        NS(stop_reason="end_turn", content=[text_block("The answer is 4")]),
    ])
    monkeypatch.setattr(agent, "client", fake)
    assert "4" in agent.run_agent("what is 2+2?")

def test_agent_stops_at_max_steps(monkeypatch):
    looping = NS(stop_reason="tool_use",
                 content=[tool_block("t1", "calculator", {"expression": "1+1"})])
    monkeypatch.setattr(agent, "client", FakeClient([looping] * 5))
    assert "max steps" in agent.run_agent("loop", max_steps=3)



def test_read_webpage_extracts_text():
    fake = MagicMock(text="<html><title>T</title><body><nav>menu</nav>"
                          "<main>Hello world</main></body></html>",
                     headers={"Content-Type": "text/html"})
    fake.raise_for_status = lambda: None
    with patch.object(tools.requests, "get", return_value=fake), \
         patch.object(tools, "_is_safe_url", return_value=True):
        out = tools.read_webpage("https://example.com")
    assert "Hello world" in out and "menu" not in out

def test_blocks_localhost():
    assert "not allowed" in tools.read_webpage("http://127.0.0.1:8000")