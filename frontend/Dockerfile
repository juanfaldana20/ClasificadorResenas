FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv
RUN pip install --upgrade pip && pip install uv

# Copiar archivos
COPY pyproject.toml .
COPY App ./App

# Instalar dependencias con uv
RUN uv pip install --system -r pyproject.toml

EXPOSE 8501

CMD ["streamlit", "run", "App/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
