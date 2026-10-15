import json

def calculator(expression: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: invalid characters"
    return str(eval(expression))  # fine for a demo; use a real parser later

TOOLS = {
    "calculator": {
        "fn": calculator,
        "schema": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression.",
            "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }
}

def run_tool(name: str, args: dict) -> str:
    try:
        return TOOLS[name]["fn"](**args)
    except Exception as e:
        return f"Error: {e}"