import calendar
import json
import string
import ipaddress
import socket
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from . import rag
from .rag import search_knowledge
import traceback

MAX_PAGE_CHARS = 8000
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; agent-harness/0.1)"}

def calculator(expression: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: invalid characters"
    return str(eval(expression))  # fine for a demo; use a real parser later

def clock (expression: str = "") -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def web_search(query: str, max_results: int = 5) -> str:
    """Return titles, URLs, and snippets for a search query."""
    results = DDGS().text(query, max_results=max_results)
    if not results:
        return "No results found."
    return "\n\n".join(
        f"{i}. {r['title']}\n   {r['href']}\n   {r['body']}"
        for i, r in enumerate(results, 1)
    )


def _is_safe_url(url: str) -> bool:
    """Block non-http(s) schemes and private/internal addresses."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(p.hostname))
    except socket.gaierror:
        return False
    return not (ip.is_private or ip.is_loopback or ip.is_link_local)


def read_webpage(url: str, max_chars: int = MAX_PAGE_CHARS) -> str:
    """Fetch a page and return its cleaned main text, truncated."""
    if not _is_safe_url(url):
        return "Error: URL is not allowed."

    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    if "text/html" not in resp.headers.get("Content-Type", ""):
        return "Error: not an HTML page."

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else "(no title)"
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = " ".join(root.get_text(separator=" ").split())

    truncated = len(text) > max_chars
    text = text[:max_chars]
    return f"Title: {title}\nURL: {url}\n\n{text}" + (
        "\n\n[truncated]" if truncated else ""
    )

def search_knowledge(query: str, k: int = 5) -> str:
    from harness.rag import search_knowledge as rag_search_knowledge
    return rag_search_knowledge(query, k=k)

def _fetch_clean(url: str) -> tuple[str, str]:
    """Fetch a page and return (title, full cleaned text)."""
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    if "text/html" not in resp.headers.get("Content-Type", ""):
        raise ValueError("not an HTML page")

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else "(no title)"
    root = soup.find("main") or soup.find("article") or soup.body or soup
    return title, " ".join(root.get_text(separator=" ").split())


def read_webpage(url: str, max_chars: int = MAX_PAGE_CHARS) -> str:
    """Fetch a page, save it to the knowledge base, return a preview."""
    if not _is_safe_url(url):
        return "Error: URL is not allowed."

    title, text = _fetch_clean(url)

    note = ""
    if len(text) >= 200:                      # skip empty or junk pages
        try:
            n = rag.ingest(url, f"{title}\n\n{text}")
            note = (f"\n\n[Saved {n} chunks to the knowledge base. "
                    f"Use search_knowledge to query other parts of this page.]")
        except Exception:
            pass                              # ingestion must never break reading

    preview = text[:max_chars]
    if len(text) > max_chars:
        preview += "\n\n[preview truncated]"
    return f"Title: {title}\nURL: {url}\n\n{preview}{note}"

def list_sources() -> str:
    return rag.list_sources_text()

def read_source(source: str, offset: int = 0) -> str:
    return rag.read_source_text(source, offset)

TOOLS = { #planned tools: code generation, PDF scanner (for articles found online and on my machine)
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
    },
    "calendar": {
        "fn": calendar,  # tells the current date and events if any
        "schema": {
            "name": "calendar",
            "description": "Access calendar events.",
            "input_schema": {
                "type": "object",
                "properties": {"date": {"type": "string"}},
                "required": ["date"],
            },
        },
    },
    "clock": {
        "fn": clock,  # tells the current time (dont forget to add timezones)
        "schema": {
            "name": "clock",
            "description": "Access the current time.",
            "input_schema": {
                "type": "object",
                "properties": {"clock": {"type": "string"}},
                "required": [],
            },
        },
    },
    "web_search": {
        "fn": web_search, #searches the web (currently doesnt work)
        "schema": {
            "name": "web_search",
            "description": "Perform a web search.",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    "webpage_scanning": {
        "fn": read_webpage, #scans a webpage (doesnt work also yet)
        "schema": {
            "name": "webpage_scanning",
            "description": "Scan a webpage for content.",
            "input_schema": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    "search_knowledge": {
        "fn": search_knowledge,
        "schema": {
            "name": "search_knowledge",
            "description": "Search the local knowledge base of saved documents and previously read web pages. Try this before web_search.",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                "required": ["query"],
            },
        },
    },
    "list_sources": {
        "fn": list_sources,
        "schema": {
            "name": "list_sources",
            "description": "List every document and web page saved in the knowledge base, "
                       "with chunk counts. Use when asked what you know or have access to, "
                       "or to check whether a source is already saved before searching the web.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    "read_source": {
        "fn": read_source,
        "schema": {
            "name": "read_source",
            "description": "Read the text of a document already in the knowledge base, in order, "
                        "8000 characters at a time. Use to summarize or review a whole "
                        "document. Pass the name from list_sources (a filename is enough). "
                        "If the result says more remains, call again with the given offset.",
            "input_schema": {
                "type": "object",
                "properties": {"source": {"type": "string"}, "offset": {"type": "integer", "default": 0}},
                "required": ["source"],
            },
        },
    },
    
}

def run_tool(name: str, args: dict) -> str:
    try:
        return TOOLS[name]["fn"](**args)
    except Exception as e:
        traceback.print_exc()          # full trace goes to your terminal
        return f"Error: {e}"