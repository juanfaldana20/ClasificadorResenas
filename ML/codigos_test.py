# test_server.py
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
    Crea un pipeline simulado que siempre devuelve POS con score 0.95.
    """
    def fake_pipeline(inputs):
        if isinstance(inputs, list):
            return [{"label": "POS", "score": 0.95} for _ in inputs]
        else:
            return [{"label": "POS", "score": 0.95}]
    return fake_pipeline


# -----------------------------
# Tests
# -----------------------------
def test_predict_un_texto(pipeline_simulado):
    """
    Test para un solo texto en Predict.
    """
    with patch("server.pipeline", return_value=pipeline_simulado):
        servicio = SentimentService()
        solicitud = sentiment_pb2.PredictRequest(text="Me gusta este servicio")
        respuesta = servicio.Predict(solicitud, None)

        assert respuesta.label == "POS"
        assert isinstance(respuesta.score, float)


def test_predict_batch_varios(pipeline_simulado):
    """
    Test para varios textos en PredictBatch.
    """
    with patch("server.pipeline", return_value=pipeline_simulado):
        servicio = SentimentService()
        solicitud = sentiment_pb2.PredictBatchRequest(texts=["uno", "dos", "tres"])
        respuesta = servicio.PredictBatch(solicitud, None)

        assert len(respuesta.labels) == 3
        assert all(l == "POS" for l in respuesta.labels)
        assert all(isinstance(s, float) for s in respuesta.scores)


def test_predict_texto_vacio(pipeline_simulado):
    """
    Test para texto vacío en Predict.
    """
    with patch("server.pipeline", return_value=pipeline_simulado):
        servicio = SentimentService()
        solicitud = sentiment_pb2.PredictRequest(text="")
        respuesta = servicio.Predict(solicitud, None)

        assert respuesta.label == "POS"
        assert 0 <= respuesta.score <= 1


def test_ping(pipeline_simulado):
    """
    Test del método Ping.
    """
    with patch("server.pipeline", return_value=pipeline_simulado):
        servicio = SentimentService()
        solicitud = sentiment_pb2.PingRequest()
        respuesta = servicio.Ping(solicitud, None)

        assert respuesta.status == "ok"



def test_predict_batch_grande(pipeline_simulado, mlflow_simulado):
    """
    Test para predecir un batch grande de textos.
    """
    with patch("server.pipeline", return_value=pipeline_simulado):
        servicio = SentimentService()
        solicitud = sentiment_pb2.PredictBatchRequest(texts=["texto"] * 1000)
        respuesta = servicio.PredictBatch(solicitud, None)

        assert len(respuesta.labels) == 1000
        assert all(l == "POSITIVE" for l in respuesta.labels)


def test_predict_input_invalido(pipeline_simulado, mlflow_simulado):
    """
    Test para manejar un request sin texto (input inválido).
    """
    with patch("server.pipeline", return_value=pipeline_simulado):
        servicio = SentimentService()
        solicitud_invalida = sentiment_pb2.PredictRequest()
        respuesta = servicio.Predict(solicitud_invalida, None)

        assert respuesta.label == "POSITIVE"
        assert isinstance(respuesta.score, float)


def test_mlflow_log_model_llamado(pipeline_simulado, mlflow_simulado):
    """
    Valida que mlflow.transformers.log_model sea llamado durante la inicialización.
    """
    with patch("server.pipeline", return_value=pipeline_simulado):
        servicio = SentimentService()

        # Se llamó log_model
        assert mlflow_simulado.transformers.log_model.called

        # Se loggeó el parámetro huggingface_model_id
        mlflow_simulado.log_param.assert_any_call(
            "huggingface_model_id", servicio.model_id
        )


def test_inicializacion_servicio(pipeline_simulado, mlflow_simulado):
    """
    Valida que el servicio inicializa correctamente el modelo y el pipeline.
    """
    with patch("server.pipeline", return_value=pipeline_simulado):
        servicio = SentimentService()

        assert servicio.model_id == "finiteautomata/beto-sentiment-analysis"
        assert servicio.clf is not None
