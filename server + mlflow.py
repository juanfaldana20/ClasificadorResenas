import os
from concurrent import futures
import grpc
from transformers import pipeline
import mlflow
import mlflow.transformers

import sentiment_pb2
import sentiment_pb2_grpc

#hola
class SentimentService(sentiment_pb2_grpc.SentimentServiceServicer):
    def __init__(self):
        """
        Carga el pipeline de BETO y asegura que MLflow esté configurado.
        Registra (log_model) el pipeline en MLflow si aún no existe un run activo.
        """
        # 1) Cargar modelo HF (BETO ya fine-tuned en sentimiento)
        self.model_id = "finiteautomata/beto-sentiment-analysis"
        self.clf = pipeline("sentiment-analysis", model=self.model_id)

        # 2) Configurar experimento MLflow (usa env MLFLOW_* si existen)
        exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "beto-sentiment")
        mlflow.set_experiment(exp_name)

        # 3) Registrar el pipeline como modelo en MLflow una vez por arranque
        #    Guardamos artifacts en ./mlruns (si MLFLOW_TRACKING_URI=file:./mlruns)
        with mlflow.start_run(run_name="startup-register-model"):
            # Parámetros útiles para rastrear qué cargamos
            mlflow.log_param("huggingface_model_id", self.model_id)
            mlflow.set_tag("flavor", "transformers-pipeline")
            mlflow.set_tag("service", "grpc-sentiment")

            # Registrar el pipeline en MLflow bajo artifacts/model
            mlflow.transformers.log_model(
                transformers_model=self.clf,
                artifact_path="model",
                task="sentiment-analysis",
            )

            # Métrica de “sanidad” opcional: inferencia de prueba corta
            test_out = self.clf("Este servicio funciona correctamente")[0]
            mlflow.log_metric("startup_score_example", float(test_out["score"]))

            # Guardar el run_id por si luego quieres mostrarlo en Ping()
            self.startup_run_id = mlflow.active_run().info.run_id

    def Predict(self, request, context):
        """
        Recibe un texto y devuelve etiqueta y score.
        """
        result = self.clf(request.text)[0]
        return sentiment_pb2.PredictResponse(label=result["label"], score=result["score"])

    def PredictBatch(self, request, context):
        """
        Recibe lista de textos y devuelve listas paralelas de etiquetas y scores.
        """
        results = self.clf(list(request.texts))
        labels = [r["label"] for r in results]
        scores = [r["score"] for r in results]
        return sentiment_pb2.PredictBatchResponse(labels=labels, scores=scores)

    def Ping(self, request, context):
        """
        Verifica que el servicio esté vivo. Devuelve 'ok' y, si existe, el run_id del registro.
        """
        status = "ok"
        if hasattr(self, "startup_run_id"):
            status = f"ok|run_id={self.startup_run_id}"
        return sentiment_pb2.PingResponse(status=status)


def serve():
    """
    Arranca servidor gRPC en puerto 50051.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    sentiment_pb2_grpc.add_SentimentServiceServicer_to_server(SentimentService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("SentimentService gRPC corriendo en puerto 50051")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()