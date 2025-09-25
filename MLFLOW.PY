# mlflow_eval.py
import os
import mlflow
import mlflow.transformers
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline

# -------------------------
# 1. Cargar dataset local
# -------------------------
df = pd.read_csv("reseñas.csv")  # columnas: text,label

# Mapear etiquetas a enteros
label_map = {"NEG": 0, "NEU": 1, "POS": 2}
df["label_id"] = df["label"].map(label_map)

# Dividir en train/test (aunque no entrenamos, solo usamos test)
test_df = df.sample(frac=0.2, random_state=42)

# -------------------------
# 2. Función de evaluación
# -------------------------
def evaluate_model(model_id):
    """Evalúa un modelo Hugging Face ya entrenado en tu dataset local"""
    print(f"\n🔹 Evaluando modelo: {model_id}")
    clf = pipeline("sentiment-analysis", model=model_id)

    y_true = []
    y_pred = []

    for _, row in test_df.iterrows():
        text, label = row["text"], row["label_id"]
        result = clf(text[:512])[0]  # truncamos a 512 tokens
        pred_label = result["label"].upper()

        # Normalizar salida del modelo a nuestro esquema NEG/NEU/POS
        if pred_label.startswith("NEG"):
            pred = 0
        elif pred_label.startswith("NEU"):
            pred = 1
        else:
            pred = 2

        y_true.append(label)
        y_pred.append(pred)

    # Calcular métricas
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"📊 Resultados {model_id}: Accuracy={acc:.3f}, F1={f1:.3f}")

    # Log en MLflow
    with mlflow.start_run(run_name=model_id):
        mlflow.log_param("huggingface_model_id", model_id)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.transformers.log_model(
            transformers_model=clf,
            artifact_path="model",
            task="sentiment-analysis",
        )

# -------------------------
# 3. Ejecutar comparación
# -------------------------
if __name__ == "__main__":
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "comparacion_beto_distilbert"

    # Modelos en español
    modelos = [
        "finiteautomata/beto-sentiment-analysis",   # BETO fine-tuned español
        "distilbert-base-uncased-finetuned-sst-2-english" # DistilBERT español fine-tuned
    ]

    for modelo in modelos:
        evaluate_model(modelo)
