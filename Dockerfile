FROM python:3.11-slim

# Non-root user for security
RUN useradd --create-home appuser
WORKDIR /app

# Install dependencies before copying source (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source (secrets are injected at runtime, not baked in)
COPY . .

# Ensure output directory exists and is writable by appuser
RUN mkdir -p output/fasta && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# GEMINI_API_KEY must be injected via --env or docker-compose environment
CMD ["python", "server.py"]
