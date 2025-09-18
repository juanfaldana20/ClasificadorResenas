# Clasificador de Reseñas con IA (Streamlit + gRPC + Transformers)

Aplicación para gestionar y analizar reseñas de restaurantes con apoyo de IA. La interfaz está construida en Streamlit (App/main.py) y consume un servicio gRPC (ML/server.py) que utiliza un modelo de Hugging Face (BETO: finiteautomata/beto-sentiment-analysis) para análisis de sentimientos. Opcionalmente se registran métricas y el modelo en MLflow.


## Contenido
- Descripción del proyecto
- Requisitos
- Instalación y ejecución local
- Ejecución con Docker (docker-compose)
- Arquitectura del software
- Estructura del repositorio
- Variables de entorno
- Tablero Kanban
- Notas y solución de problemas (FAQ)


## Descripción del proyecto
La aplicación permite:
- Escribir reseñas y almacenarlas en una “base de datos” simulada en memoria (sesión de Streamlit).
- Analizar el sentimiento de un texto (positivo/negativo/neutral) desde la UI usando gRPC.
- Visualizar métricas básicas de las reseñas capturadas.
- Analizar un archivo (CSV/XLSX) vía gRPC en lote y descargar un CSV con predicciones.
- Registrar el pipeline del modelo y una inferencia de validación en MLflow al iniciar el servicio gRPC.

Tecnologías clave:
- Frontend: Streamlit (App/main.py).
- Backend IA: Servicio gRPC en Python con Transformers (ML/server.py).
- Modelo: BETO fine-tuned para análisis de sentimientos.
- Observabilidad de ML: MLflow (almacenamiento local ./mlruns).


## Requisitos
- Python 3.11
- pip (incluido con Python)
- (Opcional) Docker y Docker Compose v2

Se recomienda crear y activar un entorno virtual (venv) para la instalación local.


## Instalación y ejecución local
1) Crear y activar entorno virtual
- Windows PowerShell:
  - python -m venv .venv
  - .\.venv\Scripts\Activate.ps1
- Linux/macOS (bash/zsh):
  - python -m venv .venv
  - source .venv/bin/activate

2) Instalar dependencias
- pip install --upgrade pip
- pip install -r requirements.txt

3) Arrancar el servicio gRPC (IA)
- python ML/server.py

4) (Opcional) MLflow UI
- mlflow ui --host 0.0.0.0 --port 5000
- Abrir http://127.0.0.1:5000

5) Ejecutar la interfaz Streamlit
- Windows PowerShell (opcionalmente define la dirección del servicio):
  - $env:APP_GRPC_ADDR = "localhost:50051"
  - streamlit run App/main.py
- Linux/macOS:
  - export APP_GRPC_ADDR="localhost:50051"
  - streamlit run App/main.py

La UI quedará disponible en http://localhost:8501.


## Ejecución con Docker (docker-compose)
Asegúrate de tener Docker Desktop (Windows/macOS) o Docker Engine (Linux) instalado.

1) Levantar servicios principales
- docker compose up -d --build grpc streamlit

2) Acceso
- UI: http://localhost:8501
- gRPC: expuesto en localhost:50051

3) (Opcional) MLflow UI
En el compose incluido, el servicio mlflow usa la imagen base de Python y no tiene MLflow preinstalado. Tienes dos opciones:
- Opción A (local): Ejecuta “mlflow ui” localmente como se indica en la sección local.
- Opción B (en Compose): Cambia la definición del servicio mlflow para que use la misma imagen del proyecto (build .), donde ya está instalado MLflow, por ejemplo:

  services:
    mlflow:
      build: .
      container_name: sentiment_mlflow
      working_dir: /app
      command: sh -c "mlflow ui --host 0.0.0.0 --port 5000"
      volumes:
        - ./mlruns:/app/mlruns
      ports:
        - "5000:5000"
      depends_on:
        - grpc
      restart: unless-stopped

4) Logs
- docker compose logs -f grpc streamlit

Nota importante sobre mayúsculas/minúsculas en rutas (Linux/macOS): este repositorio usa App y ML en mayúsculas. docker-compose.yaml ya referencia App/main.py correctamente. Además, App/main.py añade ambos paths (ML y ml) al sys.path para compatibilidad, pero se recomienda mantener App y ML en mayúsculas de forma consistente en disco y en las referencias.


## Arquitectura del software
Componentes:
- Interfaz (Streamlit) – App/main.py
  - Tabs: Escribir Reseña, Análisis IA, Base de Datos, Archivo.
  - Cliente gRPC: consume Predict, PredictBatch y Ping del servicio.
  - Variable APP_GRPC_ADDR para apuntar al host:puerto del servicio.
- Servicio IA (gRPC) – ML/server.py
  - Servidor gRPC en puerto 50051.
  - Pipeline de Transformers: finiteautomata/beto-sentiment-analysis.
  - Registra en MLflow (experimento configurable con MLFLOW_EXPERIMENT_NAME) y guarda artefactos en ./mlruns.
- MLflow (opcional)
  - UI para explorar corridas (runs) y artefactos del modelo.

Flujo de datos:
1) Usuario interactúa en Streamlit.
2) Streamlit llama al servicio gRPC con el texto.
3) El servicio retorna etiqueta y score.
4) (Opcional) MLflow registra ejecución/modelo al iniciar el servicio.


## Estructura del repositorio
- App/
  - main.py  (Interfaz de Streamlit)
- ML/
  - server.py (Servidor gRPC con Transformers y MLflow)
  - client.py (Cliente de prueba para gRPC)
  - sentiment_pb2.py, sentiment_pb2_grpc.py (stubs generados)
- requirements.txt (dependencias del proyecto)
- Dockerfile (imagen base con dependencias)
- docker-compose.yaml (servicios grpc, streamlit y ejemplo de mlflow)
- mlruns/ (se crea al ejecutar MLflow en modo archivo)


## Variables de entorno
- APP_GRPC_ADDR: Dirección del servicio gRPC (por defecto: localhost:50051 en la UI).
- MLFLOW_EXPERIMENT_NAME: Nombre del experimento MLflow (por defecto: beto-sentiment).
- (Opcional) MLFLOW_TRACKING_URI: URI del tracking de MLflow. Para archivo local: file:./mlruns

Ejemplos:
- Windows PowerShell: $env:APP_GRPC_ADDR = "grpc:50051"
- Linux/macOS: export APP_GRPC_ADDR="grpc:50051"


## Tablero Kanban
Enlace al tablero Kanban del proyecto:
- Reemplaza este placeholder con tu URL real: {{KANBAN_URL}}
  (por ejemplo: https://github.com/tu-org/tu-repo/projects/1 o Trello/Jira equivalente)


## Notas y solución de problemas (FAQ)
- La UI abre pero no muestra resultados de IA:
  - Asegúrate de que el servicio gRPC esté corriendo (python ML/server.py) y que APP_GRPC_ADDR apunte a ese servicio.
  - Verifica que los stubs sentiment_pb2*.py existan en ML/ y sean importables.
- Error en Docker por rutas con mayúsculas/minúsculas:
  - Actualiza docker-compose.yaml a App/main.py (A mayúscula) o renombra los directorios a minúsculas y ajusta imports.
- MLflow en Docker no levanta:
  - Usa la opción A (local) o ajusta el servicio mlflow del compose como se describe.
- Puertos ocupados:
  - Cambia los puertos mapeados en docker-compose.yaml o al ejecutar mlflow y streamlit.
