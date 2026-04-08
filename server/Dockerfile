# Email Triage Environment — Dockerfile
# Multi-stage build for the OpenEnv Email Triage environment.
#
# Build:   docker build -t email-triage-env -f server/Dockerfile .
# Run:     docker run -p 8000:8000 email-triage-env
# Health:  curl http://localhost:8000/health

# ── Stage 1: Builder ─────────────────────────────────────────────
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Build argument to control standalone vs in-repo builds
ARG BUILD_MODE=standalone

# Copy environment code
COPY . /app/env

WORKDIR /app/env

# Ensure uv is available
RUN if ! command -v uv >/dev/null 2>&1; then \
        pip install uv || \
        (curl -LsSf https://astral.sh/uv/install.sh | sh && \
         mv /root/.local/bin/uv /usr/local/bin/uv && \
         mv /root/.local/bin/uvx /usr/local/bin/uvx); \
    fi

# Install git for building from git repos
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy environment code
COPY --from=builder /app/env /app/env

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Expose the server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI server
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
