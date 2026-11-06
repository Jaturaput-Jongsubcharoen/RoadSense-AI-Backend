"""Run a bounded local Flask API load test with mocked or real Ollama inference."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import threading
import time

import requests
from werkzeug.serving import make_server

import app as flask_app
from services.llm_config import OLLAMA_MODEL

MAX_REQUESTS = 100
MAX_DURATION_SECONDS = 180
MAX_REAL_CONCURRENCY = 3


def _percentile_95(values):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, int(0.95 * len(ordered) + 0.999999) - 1)
    return round(ordered[index], 2)


def _send_request(url, timeout_seconds):
    started_at = time.monotonic()
    try:
        response = requests.post(
            url,
            json={"message": "What should drivers do when they see a pothole?"},
            timeout=timeout_seconds,
        )
        return {
            "success": 200 <= response.status_code < 300,
            "status_code": response.status_code,
            "latency_ms": round((time.monotonic() - started_at) * 1000, 2),
        }
    except requests.RequestException as error:
        return {
            "success": False,
            "status_code": None,
            "latency_ms": round((time.monotonic() - started_at) * 1000, 2),
            "error": type(error).__name__,
        }


def run_load_test(mode, request_count, concurrency, duration_seconds):
    """Run requests against a private loopback Flask server and report measurements."""
    if not 1 <= request_count <= MAX_REQUESTS:
        raise ValueError(f"Requests must be between 1 and {MAX_REQUESTS}")
    if not 1 <= duration_seconds <= MAX_DURATION_SECONDS:
        raise ValueError(
            f"Duration must be between 1 and {MAX_DURATION_SECONDS} seconds"
        )
    concurrency_limit = MAX_REAL_CONCURRENCY if mode == "ollama" else 10
    if not 1 <= concurrency <= concurrency_limit:
        raise ValueError(
            f"{mode} concurrency must be between 1 and {concurrency_limit}"
        )

    server = make_server("127.0.0.1", 0, flask_app.app, threaded=True)
    base_url = f"http://127.0.0.1:{server.server_port}"
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    patcher = None
    if mode == "mocked":
        from unittest.mock import patch

        patcher = patch.object(
            flask_app, "chat_with_ollama", return_value="Mocked road-safety answer."
        )
        patcher.start()

    started_at = time.monotonic()
    deadline = started_at + duration_seconds
    records = []
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            pending = set()
            submitted = 0
            while pending or (submitted < request_count and time.monotonic() < deadline):
                while (
                    submitted < request_count
                    and len(pending) < concurrency
                    and time.monotonic() < deadline
                ):
                    pending.add(executor.submit(
                        _send_request,
                        f"{base_url}/api/chat",
                        max(2, min(30, duration_seconds)),
                    ))
                    submitted += 1
                if not pending:
                    break
                done, pending = _wait_for_one(pending, deadline)
                records.extend(future.result() for future in done)
    finally:
        if patcher is not None:
            patcher.stop()
        server.shutdown()
        server_thread.join(timeout=5)

    elapsed_seconds = time.monotonic() - started_at
    latencies = [item["latency_ms"] for item in records]
    successes = sum(item["success"] for item in records)
    failures = len(records) - successes
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "endpoint": "POST /api/chat (loopback only)",
        "model": OLLAMA_MODEL if mode == "ollama" else None,
        "concurrency": concurrency,
        "requested_requests": request_count,
        "total_requests": len(records),
        "successful_requests": successes,
        "failed_requests": failures,
        "requests_per_second": round(len(records) / elapsed_seconds, 3)
        if elapsed_seconds else 0,
        "average_response_ms": round(statistics.mean(latencies), 2)
        if latencies else None,
        "p95_response_ms": _percentile_95(latencies),
        "error_rate": round(failures / len(records), 4) if records else None,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "duration_limit_seconds": duration_seconds,
        "request_results": records,
        "limitations": [
            "Mocked mode measures local Flask HTTP and routing overhead, not inference.",
            "Ollama mode measures this machine and local model only.",
            "Requests are sent only to a loopback server; production URLs are not used.",
        ],
    }


def _wait_for_one(pending, deadline):
    from concurrent.futures import FIRST_COMPLETED, wait

    remaining = max(0, deadline - time.monotonic())
    if remaining == 0:
        remaining = 30
    return wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mocked", "ollama"), required=True)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = run_load_test(
        args.mode, args.requests, args.concurrency, args.duration_seconds
    )
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        key: report[key] for key in (
            "mode", "total_requests", "successful_requests", "failed_requests",
            "requests_per_second", "average_response_ms", "p95_response_ms",
            "error_rate", "elapsed_seconds",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
