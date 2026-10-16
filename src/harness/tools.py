import calendar
import json
import string
import ipaddress
import socket
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

def calculator(expression: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: invalid characters"
    return str(eval(expression))  # fine for a demo; use a real parser later

def clock (expression: str = "") -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def web_search(query: str) -> str:
    return f"Searching the web for: {query}"

def webpage_scanning(url: str) -> str:
    return f"Scanning webpage: {url}"

TOOLS = { #planned tools: clock, calendar, web search, webpage scanning, memory (BIG ONE!!), code generation
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
        "fn": clock,  # tells the current time
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
        "fn": webpage_scanning, #scans a webpage (doesnt work also yet)
        "schema": {
            "name": "webpage_scanning",
            "description": "Scan a webpage for content.",
            "input_schema": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    }
}

def run_tool(name: str, args: dict) -> str:
    try:
        return TOOLS[name]["fn"](**args)
    except Exception as e:
        return f"Error: {e}"