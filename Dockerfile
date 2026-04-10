FROM python:3.11-slim

RUN useradd --create-home appuser
WORKDIR /app

# System libs for pyscf (BLAS/LAPACK) and matplotlib (freetype)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopenblas-dev liblapack-dev gfortran libfreetype6-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Directories used by the server at runtime
RUN mkdir -p data docs && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# GEMINI_API_KEY must be injected via --env or docker-compose environment
CMD ["python", "server.py"]
