"""Deployment-safe acquisition for the required TensorFlow model artifact."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import requests


class ModelArtifactError(RuntimeError):
    """Raised when the required model artifact cannot be prepared."""


def ensure_model_artifact(model_path: Path) -> None:
    """Ensure the configured model path exists, downloading it only when needed."""
    if model_path.is_file():
        return

    download_url = os.getenv("MODEL_DOWNLOAD_URL", "").strip()
    if not download_url:
        raise ModelArtifactError(
            f"Model artifact is missing at {model_path}. "
            "Set MODEL_PATH to an existing file or configure MODEL_DOWNLOAD_URL."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    expected_sha256 = os.getenv("MODEL_SHA256", "").strip().lower()
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(dir=model_path.parent, suffix=".download", delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
            hasher = hashlib.sha256()
            with requests.get(download_url, stream=True, timeout=(10, 300)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        temporary_file.write(chunk)
                        hasher.update(chunk)

        if temporary_path.stat().st_size == 0:
            raise ModelArtifactError("Downloaded model artifact is empty.")

        if expected_sha256 and hasher.hexdigest() != expected_sha256:
            raise ModelArtifactError("Downloaded model checksum does not match MODEL_SHA256.")

        os.replace(temporary_path, model_path)
        temporary_path = None
    except requests.RequestException as error:
        raise ModelArtifactError("Model artifact download failed. Check MODEL_DOWNLOAD_URL.") from error
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
