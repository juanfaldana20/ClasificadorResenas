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
    print("one:", predict(stub, "Vengo por la comida y solo por la comida. Los tacos al pastor están en otro nivel: tortilla caliente, carne bien dorada y jugosa, piña fresca en el punto, y una salsa de habanero que pica sin matar el sabor. El guacamole es cremoso y con buen limeado, y el arroz sale suelto, no pastoso. Hasta el café, simple, sale correcto. Pero el servicio arruina la experiencia. Nos ignoraron al llegar, tardaron más de 20 minutos en tomar la orden, trajeron los platos desparejos y tuve que pedir tres veces las bebidas. La mesera fue cortés pero ausente, y la cuenta vino con cargos que no pedimos. No es un mal día aislado, ya me pasó algo similar antes. La cocina merece aplauso, el salón necesita gestión básica: tiempos, atención y seguimiento. Si pudiera pedir en ventanilla y comer de pie, lo haría feliz. Volvería por los sabores, pero solo si mejoran el servicio o si voy con paciencia de sobra."))
    print("batch:", predict_batch(stub, ["Me encanta este lugar", "amo"]))


if __name__ == "__main__":
    main()
