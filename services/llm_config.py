"""Shared, environment-driven configuration for Ollama-compatible LLM endpoints."""

from __future__ import annotations

import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()


def ollama_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    return headers
