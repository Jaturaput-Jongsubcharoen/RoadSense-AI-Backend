import json
import logging
import time
import uuid
from datetime import datetime, timezone

import requests

from services.llm_config import (
    OLLAMA_COST_PER_1000_TOKENS,
    OLLAMA_MAX_RETRIES,
    OLLAMA_MODEL,
    OLLAMA_RETRY_DELAY_SECONDS,
    OLLAMA_URL,
    ollama_headers,
)

logger = logging.getLogger(__name__)
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
UNAVAILABLE_MESSAGE = "Local AI service is unavailable. Start Ollama and try again."


def estimate_tokens(text):
    """Estimate tokens by counting whitespace-separated words."""
    return len(text.split())


def _token_count(value):
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _log_request(request_id, started_at, prompt, status, attempts,
                 response_data=None, error_category=None, http_status=None):
    latency_ms = round((time.monotonic() - started_at) * 1000, 2)
    prompt_tokens = _token_count(
        response_data.get("prompt_eval_count") if response_data else None
    )
    completion_tokens = _token_count(
        response_data.get("eval_count") if response_data else None
    )
    prompt_source = "provider" if prompt_tokens is not None else "estimate"
    if prompt_tokens is None:
        prompt_tokens = estimate_tokens(prompt)

    answer = response_data.get("response") if response_data else None
    completion_source = "provider" if completion_tokens is not None else "unavailable"
    if completion_tokens is None and isinstance(answer, str):
        completion_tokens = estimate_tokens(answer)
        completion_source = "estimate"

    total_tokens = None
    if completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    token_source = (
        prompt_source if prompt_source == completion_source
        else f"{prompt_source}+{completion_source}"
    )
    estimated_cost = None
    if total_tokens is not None and OLLAMA_COST_PER_1000_TOKENS is not None:
        estimated_cost = round(total_tokens * OLLAMA_COST_PER_1000_TOKENS / 1000, 8)

    event = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": OLLAMA_MODEL,
        "status": status,
        "attempts": attempts,
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "token_source": token_source,
        "cost_per_1000_tokens": OLLAMA_COST_PER_1000_TOKENS,
        "estimated_provider_cost": estimated_cost,
        "error_category": error_category,
        "http_status": http_status,
    }
    message = json.dumps(event, separators=(",", ":"))
    if status == "success":
        logger.info("%s", message)
    else:
        logger.warning("%s", message)


def request_ollama(prompt, timeout, options=None):
    """Send a bounded request and record one privacy-safe JSON log event."""
    request_id = uuid.uuid4().hex
    started_at = time.monotonic()
    retry_count = 0
    error_category = None
    http_status = None

    while True:
        http_status = None
        try:
            payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
            if options is not None:
                payload["options"] = options
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                headers=ollama_headers(),
                timeout=timeout,
            )
            http_status = response.status_code
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Ollama response must be a JSON object")
            _log_request(
                request_id, started_at, prompt, "success", retry_count + 1, data
            )
            return data
        except requests.Timeout:
            error_category = "timeout"
            retryable = True
        except requests.ConnectionError:
            error_category = "connection_error"
            retryable = True
        except requests.HTTPError as error:
            status_code = error.response.status_code if error.response is not None else None
            http_status = status_code
            error_category = "http_error"
            retryable = status_code in RETRYABLE_STATUS_CODES
        except ValueError:
            error_category = "invalid_response"
            retryable = False
        except requests.RequestException:
            error_category = "request_error"
            retryable = False

        if retryable and retry_count < OLLAMA_MAX_RETRIES:
            retry_count += 1
            time.sleep(OLLAMA_RETRY_DELAY_SECONDS)
            continue

        _log_request(
            request_id, started_at, prompt, "failure", retry_count + 1,
            error_category=error_category, http_status=http_status,
        )
        return None


def chat_with_ollama(message):
    data = request_ollama(message, timeout=120)
    if data is None:
        return UNAVAILABLE_MESSAGE
    return data.get("response", "No response.")
