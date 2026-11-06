"""Shared, environment-driven configuration for Ollama-compatible LLM endpoints."""

from __future__ import annotations

import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_MAX_RETRIES = min(max(int(os.getenv("OLLAMA_MAX_RETRIES", "1")), 0), 3)
OLLAMA_RETRY_DELAY_SECONDS = min(
    max(float(os.getenv("OLLAMA_RETRY_DELAY_SECONDS", "0.5")), 0), 5
)
_cost_per_1000 = os.getenv("OLLAMA_COST_PER_1000_TOKENS", "").strip()
OLLAMA_COST_PER_1000_TOKENS = float(_cost_per_1000) if _cost_per_1000 else None


def ollama_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    return headers
