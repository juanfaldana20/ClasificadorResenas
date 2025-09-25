# Imagen base
FROM python:3.11-slim

# Directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código del proyecto
COPY ML/ ./ML/
COPY app/ ./app/

# Exponer puertos
# 50051 -> gRPC
# 8501 -> Streamlit
# 5000 -> MLflow UI
EXPOSE 50051 8501 5000

# Comando por defecto (puede ser sobrescrito por docker-compose)
CMD ["bash"]

