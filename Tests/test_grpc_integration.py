import sys, pathlib, grpc
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "ml"))
import sentiment_pb2 as pb
import sentiment_pb2_grpc as pb_grpc
import pytest

@pytest.mark.slow
def test_grpc_roundtrip(grpc_server):
    """E2E: cliente gRPC llama Ping y Predict contra el servidor real."""
    with grpc.insecure_channel(grpc_server) as ch:
        stub = pb_grpc.SentimentServiceStub(ch)
        pong = stub.Ping(pb.PingRequest())
        assert pong.status.startswith("ok")
        pred = stub.Predict(pb.PredictRequest(text="La comida es fantástica"))
        assert pred.label in {"POS", "NEG", "NEU"}
