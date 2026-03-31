# Stage 1: Build dependencies
FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --upgrade pip && pip install --no-cache-dir -e ".[dev]"

# Stage 2: Runtime image
FROM python:3.11-slim AS runtime
WORKDIR /app

# Non-root user
RUN useradd -m -u 1000 appuser

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy source
COPY src/ ./src/
COPY pyproject.toml .

# Model dir (artifacts mounted at runtime)
RUN mkdir -p models && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
