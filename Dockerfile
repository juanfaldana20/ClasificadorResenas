# Imagen base de Python
FROM python:3.10-slim

# Evitar que Python cree archivos __pycache__ y buffers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Crear directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto de la app
COPY src/ ./src/
COPY conv_MLP_84.h5 ./  

CMD ["python", "src/app.py"]

