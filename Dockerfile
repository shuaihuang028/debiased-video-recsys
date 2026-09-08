FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# Serving consumes prebuilt snapshots, so the image intentionally omits the
# training-only PyTorch dependency while retaining the package metadata.
RUN pip install --no-cache-dir \
      "duckdb>=1.1,<2" \
      "faiss-cpu>=1.8,<2" \
      "fastapi>=0.115,<1" \
      "lightgbm>=4.5,<5" \
      "numpy>=1.24,<3" \
      "pandas>=2.0,<3" \
      "PyYAML>=6,<7" \
      "redis>=5,<7" \
      "uvicorn[standard]>=0.30,<1" \
    && pip install --no-cache-dir --no-deps .

EXPOSE 8000

CMD ["python", "-m", "short_video_recsys.cli", "serve", "--config", "configs/serving.yaml"]
