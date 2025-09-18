import os
import sys
import grpc

# Añade la carpeta actual al sys.path para importar los stubs generados
sys.path.append(os.path.dirname(__file__))

import sentiment_pb2 as pb
import sentiment_pb2_grpc as pb_grpc


def make_stub(host: str = "localhost:50051"):
    """Crea el canal gRPC y el stub del servicio."""
    channel = grpc.insecure_channel(host)
    return pb_grpc.SentimentServiceStub(channel)


def ping(stub) -> str:
    """Llama al RPC Ping para verificar salud del servicio."""
    resp = stub.Ping(pb.PingRequest())
    return resp.status


def predict(stub, text: str):
    """Predicción individual. Retorna (label, score)."""
    resp = stub.Predict(pb.PredictRequest(text=text))
    return resp.label, resp.score


def predict_batch(stub, texts):
    """Predicción en lote. Retorna lista de (label, score)."""
    req = pb.PredictBatchRequest(texts=list(texts))
    resp = stub.PredictBatch(req)
    return list(zip(resp.labels, resp.scores))


def main():
    """Smoke test: ping + ejemplos de predicción."""
    stub = make_stub()
    print("ping:", ping(stub))
    print("one:", predict(stub, "Esto es excelente"))
    print("batch:", predict_batch(stub, ["Me encanta este lugar", "Esto es terrible"]))


if __name__ == "__main__":
    main()
