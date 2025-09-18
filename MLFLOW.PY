# compare_models.py
import os
import mlflow
import mlflow.transformers
from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# 1) Modelos a comparar
MODELS = [
    "finiteautomata/beto-sentiment-analysis",          # Español
    "nlptown/bert-base-multilingual-uncased-sentiment", # Multilingüe
    "distilbert-base-uncased-finetuned-sst-2-english"   # Inglés
]

# 2) Dataset de reseñas (usaremos IMDb, inglés; para español se puede usar otro)
dataset = load_dataset("imdb")
test_data = dataset["test"].shuffle(seed=42).select(range(200))  # 200 ejemplos para demo

# 3) Configurar experimento
exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "sentiment-comparison")
mlflow.set_experiment(exp_name)

def evaluate_model(model_id):
    """Evalúa un modelo Hugging Face y guarda resultados en MLflow"""
    clf = pipeline("sentiment-analysis", model=model_id)

    # Abrir un run en MLflow
    with mlflow.start_run(run_name=model_id):
        mlflow.log_param("huggingface_model_id", model_id)

        y_true = []
        y_pred = []

        for example in test_data:
            text = example["text"]
            label = 1 if example["label"] == 1 else 0  # IMDb: 0=neg, 1=pos

            result = clf(text[:512])[0]  # truncamos textos largos
            pred = 1 if result["label"].upper().startswith("POS") else 0

            y_true.append(label)
            y_pred.append(pred)

        # Calcular métricas
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # Registrar métricas en MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Guardar modelo en MLflow
        mlflow.transformers.log_model(
            transformers_model=clf,
            artifact_path="model",
            task="sentiment-analysis",
        )

        print(f"Modelo {model_id} -> Accuracy={acc:.3f}, F1={f1:.3f}")

if __name__ == "__main__":
    for m in MODELS:
        evaluate_model(m)
