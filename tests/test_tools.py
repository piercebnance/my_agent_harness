from src.harness.tools import calculator, run_tool

def test_calculator():
    assert calculator("2+2") == "4"

def test_calculator_rejects_bad_input():
    assert "Error" in calculator("__import__('os')")

def test_unknown_tool_returns_error():
    assert "Error" in run_tool("nope", {})