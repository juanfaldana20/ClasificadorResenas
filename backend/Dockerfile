FROM python:3.10-slim

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv
RUN pip install uv

WORKDIR /app

# Copiar pyproject primero (para cache de dependencias normales)
COPY pyproject.toml .

# Instalar dependencias normales (sin torch)
RUN uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install --system -r requirements.txt --no-cache-dir

# Instalar Torch (CPU only, sin CUDA)
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Copiar código
COPY . .

EXPOSE 50051

CMD ["python", "ML/server.py"]
