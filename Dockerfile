# Imagen base
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de requerimientos
COPY requirements.txt .

# Instalar dependencias
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Exponer puertos (gRPC, Streamlit, MLflow)
EXPOSE 50051 8501 5000

# Comando por defecto (se sobrescribirá en docker-compose)
CMD ["bash"]
