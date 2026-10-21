from . import rag
from .agent import run_agent

HELP = """Commands:
  /ingest <path>   add a file or folder (.md .txt .rst .py .pdf)
  /sources         list what's in the knowledge base
  /forget <path>   remove a source (use the path shown by /sources)
  /help            show this message
  exit             quit"""


def handle_command(line: str) -> None:
    cmd, _, arg = line.partition(" ")
    arg = arg.strip().strip('"').strip("'")   # drag-and-drop adds quotes

    if cmd == "/ingest":
        if not arg:
            print("Usage: /ingest <path>")
            return
        try:
            r = rag.ingest_path(arg)
        except FileNotFoundError as e:
            print(e)
            return
        print(f"Ingested {r['files']} file(s), {r['chunks']} chunks.")
        for s in r["skipped"]:
            print(f"  skipped: {s}")
    elif cmd == "/sources":
        print(rag.list_sources_text())
    elif cmd == "/forget":
        rag.forget(arg)
        print(f"Removed {arg}")
    else:
        print(HELP)


if __name__ == "__main__":
    print(HELP)
    while True:
        q = input("you> ").strip()
        if q in {"exit", "quit"}:
            break
        if not q:
            continue
        if q.startswith("/"):
            handle_command(q)
        else:
            print(run_agent(q))