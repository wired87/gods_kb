FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --create-home appuser
WORKDIR /app

# System libs for PySCF (BLAS / LAPACK)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopenblas-dev liblapack-dev gfortran && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Runtime artifact directory
RUN mkdir -p data && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# CHAR: SSE server may take time to bind; start-period avoids false unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import socket; s=socket.create_connection(('127.0.0.1',8000),2); s.close()"

# GEMINI_API_KEY must be injected via --env or docker-compose environment
CMD ["python", "server.py"]
