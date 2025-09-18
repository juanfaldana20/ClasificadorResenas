# Fixtures comunes: MLflow en tmp, pipeline falso, servicio y servidor gRPC
import os, grpc, threading
import pytest
from concurrent import futures

# Import tardío para respetar rutas del repo
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "ml"))
import server as srv
import sentiment_pb2_grpc as pb_grpc

class FakePipeline:
    """Pipeline falso que imita HF pipeline para pruebas."""
    def __call__(self, inputs):
        # Acepta string o lista de strings y devuelve siempre POS con score alto.
        if isinstance(inputs, str):
            return [{"label": "POS", "score": 0.99}]
        return [{"label": ("NEG" if "mal" in t.lower() else "POS"), "score": 0.95} for t in inputs]

@pytest.fixture
def mlflow_tmp_env(monkeypatch, tmp_path):
    """Configura MLflow a un directorio temporal para no ensuciar el repo."""
    uri = f"file:///{tmp_path.as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "beto-sentiment-test")
    return uri

@pytest.fixture
def service(monkeypatch, mlflow_tmp_env):
    """Instancia el servicio inyectando el FakePipeline."""
    monkeypatch.setattr(srv, "pipeline", lambda *a, **k: FakePipeline())
    return srv.SentimentService()

@pytest.fixture
def grpc_server(service):
    """Levanta un servidor gRPC real en puerto dinámico para pruebas E2E."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb_grpc.add_SentimentServiceServicer_to_server(service, server)
    port = server.add_insecure_port("[::]:0")  # puerto asignado por el SO
    server.start()
    addr = f"localhost:{port}"
    yield addr
    server.stop(grace=None)
