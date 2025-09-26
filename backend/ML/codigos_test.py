# tests/test_server.py
import pytest
from unittest.mock import patch
from server import SentimentService
import sentiment_pb2

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def pipeline_simulado():
    """
    Crea un pipeline simulado que siempre devuelve POSITIVE con score 0.95.
    """
    def fake_pipeline(inputs):
        if isinstance(inputs, list):
            return [{"label": "POSITIVE", "score": 0.95} for _ in inputs]
        else:
            return [{"label": "POSITIVE", "score": 0.95}]
    return fake_pipeline

# -----------------------------
# Tests
# -----------------------------
def test_predict_con_texto_vacio(pipeline_simulado):
    with patch("server.pipeline", return_value=pipeline_simulado):
        servicio = SentimentService()
        solicitud = sentiment_pb2.PredictRequest(text="")
        respuesta = servicio.Predict(solicitud, None)

        assert respuesta.label == "POSITIVE"
        assert 0 <= respuesta.score <= 1


def test_predict_batch_grande(pipeline_simulado):
    with patch("server.pipeline", return_value=pipeline_simulado):
        servicio = SentimentService()
        solicitud = sentiment_pb2.PredictBatchRequest(texts=["texto"] * 1000)
        respuesta = servicio.PredictBatch(solicitud, None)

        assert len(respuesta.labels) == 1000
        assert all(l == "POSITIVE" for l in respuesta.labels)


def test_predict_input_invalido(pipeline_simulado):
    with patch("server.pipeline", return_value=pipeline_simulado):
        servicio = SentimentService()
        solicitud_invalida = sentiment_pb2.PredictRequest()
        respuesta = servicio.Predict(solicitud_invalida, None)

        assert respuesta.label == "POSITIVE"
        assert isinstance(respuesta.score, float)


def test_inicializacion_servicio(pipeline_simulado):
    with patch("server.pipeline", return_value=pipeline_simulado):
        servicio = SentimentService()

        assert servicio.model_id == "finiteautomata/beto-sentiment-analysis"
        assert servicio.clf is not None
