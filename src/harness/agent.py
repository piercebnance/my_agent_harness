import anthropic
from dotenv import load_dotenv
from .tools import TOOLS, run_tool

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"

def run_agent(user_input: str, max_steps: int = 10) -> str:
    messages = [{"role": "user", "content": user_input}]
    tool_schemas = [t["schema"] for t in TOOLS.values()]

    for _ in range(max_steps):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system="You are a helpful agent. Use tools when needed.",
            tools=tool_schemas,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        results = []
        for block in resp.content:
            if block.type == "tool_use":
                output = run_tool(block.name, block.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})

    return "Stopped: max steps reached."