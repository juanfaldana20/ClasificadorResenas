# Makefile para Sistema de Reseñas con IA

# Variables
VENV := .venv
PY := $(VENV)/Scripts/python
PIP := $(VENV)/Scripts/pip

# -----------------------------
# Targets principales
# -----------------------------

# Crear entorno virtual e instalar dependencias
configurar:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Ejecutar servidor gRPC
server:
	$(PY) ML/server.py

# Ejecutar cliente Streamlit
aplicacion:
	$(PY) -m streamlit run ML/app.py

# Limpiar MLflow
limpiar_mlflow:
	rm -rf mlruns
	echo "MLflow limpio."

# Ejecutar comparación de modelos con MLflow
comparar:
	$(PY) compare_models.py

# Limpiar todo (MLflow + entorno)
limpiar_todo:
	rm -rf mlruns
	rm -rf $(VENV)
	echo "Entorno y MLflow eliminados."

# Entrar al shell del entorno virtual
shell:
	$(PY)

# Inicio rápido: servidor + app + mlflow UI
inicio_rapido:
	powershell -Command "& {Start-Process -NoNewWindow -FilePath '$(PY)' -ArgumentList 'ML/server.py'}"
	powershell -Command "& {Start-Process -NoNewWindow -FilePath '$(PY)' -ArgumentList '-m streamlit run App/main.py'}"
	powershell -Command "& {Start-Process -NoNewWindow -FilePath 'mlflow' -ArgumentList 'ui'}"


# -----------------------------
# Phony targets
# -----------------------------
.PHONY: configurar server aplicacion limpiar_mlflow comparar limpiar_todo shell inicio_rapido

