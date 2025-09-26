import os
from concurrent import futures
import grpc
from transformers import pipeline

import sentiment_pb2
import sentiment_pb2_grpc


class SentimentService(sentiment_pb2_grpc.SentimentServiceServicer):
    def __init__(self):
        """
        Carga el pipeline de BETO (fine-tuned en análisis de sentimientos).
        """
        # 1) Cargar modelo de HuggingFace
        self.model_id = "finiteautomata/beto-sentiment-analysis"
        self.clf = pipeline("sentiment-analysis", model=self.model_id)

    def Predict(self, request, context):
        """
        Recibe un texto y devuelve etiqueta y score.
        """
        result = self.clf(request.text)[0]
        return sentiment_pb2.PredictResponse(
            label=result["label"],
            score=result["score"]
        )

    def PredictBatch(self, request, context):
        """
        Recibe lista de textos y devuelve listas paralelas de etiquetas y scores.
        """
        results = self.clf(list(request.texts))
        labels = [r["label"] for r in results]
        scores = [r["score"] for r in results]
        return sentiment_pb2.PredictBatchResponse(
            labels=labels,
            scores=scores
        )

    def Ping(self, request, context):
        """
        Verifica que el servicio esté vivo.
        """
        return sentiment_pb2.PingResponse(status="ok")


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
