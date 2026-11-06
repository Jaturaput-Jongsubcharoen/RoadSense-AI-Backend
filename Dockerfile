FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/efficientnet_best_model.keras \
    HF_HOME=/home/roadsense/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system roadsense \
    && useradd --system --gid roadsense --home-dir /home/roadsense roadsense \
    && mkdir -p /app/uploads /app/models /home/roadsense/.cache/huggingface \
    && chown -R roadsense:roadsense /app/uploads /home/roadsense

COPY requirements-render.txt .
RUN pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-render.txt

COPY app.py .
COPY services ./services
COPY examples ./examples

USER roadsense

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:5000/api/health', timeout=3)"

CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "210", "--graceful-timeout", "30", "--bind", "0.0.0.0:5000", "app:app"]
