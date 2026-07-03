# =============================================================================
# Monte Carlo Stock Predictor — Production Dockerfile
# Multi-stage build: builder → runtime
# Target image < 200 MB | Non-root user | Security hardened
# =============================================================================

# ── Stage 1: Dependency builder ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

LABEL org.opencontainers.image.title="Monte Carlo Stock Predictor"
LABEL org.opencontainers.image.description="PE-grade stochastic stock simulation API"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_ENV=production \
    LOG_FORMAT=json \
    LOG_LEVEL=INFO \
    PORT=8000

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Non-root user for security
RUN groupadd --gid 1001 mcuser && \
    useradd --uid 1001 --gid mcuser --shell /bin/bash --create-home mcuser

# Copy application code
COPY --chown=mcuser:mcuser src/ ./src/
COPY --chown=mcuser:mcuser run_simulation.py .

RUN mkdir -p /app/outputs && chown mcuser:mcuser /app/outputs

USER mcuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "src.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--log-config", "/dev/null"]
