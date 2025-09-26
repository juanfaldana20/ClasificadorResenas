FROM python:3.11-slim

# Variables para cachear modelos HuggingFace
ENV TRANSFORMERS_CACHE=/models/transformers_cache \
    HF_HOME=/models/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv
RUN pip install --upgrade pip && pip install uv

# Copiar archivos
COPY pyproject.toml .
COPY ML ./ML

# Instalar dependencias usando uv (sin torch)
RUN uv pip install --system -r pyproject.toml

# ⚡ Instalar torch versión CPU solamente
RUN uv pip install --system torch==2.3.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Generar archivos gRPC (dentro de ML/)
RUN python -m grpc_tools.protoc -I ./ML --python_out=./ML --grpc_python_out=./ML ./ML/sentiment.proto

# ❌ Quitar precarga del modelo (causaba error en build)
# RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='finiteautomata/beto-sentiment-analysis')"

EXPOSE 50051

CMD ["python", "ML/server.py"]

