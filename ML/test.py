import pytest
from unittest.mock import patch, MagicMock

# Importar directamente porque están en la misma carpeta
from server import SentimentService
import sentiment_pb2


@pytest.fixture
def mock_pipeline():
    """Crea un pipeline simulado que devuelve POSITIVE por cada texto recibido"""
    def fake_pipeline(inputs):
        if isinstance(inputs, list):
            return [{"label": "POSITIVE", "score": 0.95} for _ in inputs]
        else:
            return [{"label": "POSITIVE", "score": 0.95}]
    return fake_pipeline



@pytest.fixture
def mock_mlflow():
    """Mock de mlflow con tracking de llamadas"""
    with patch("server.mlflow") as mock_ml:
        # Configurar run_id
        mock_ml.start_run.return_value.__enter__.return_value.info.run_id = "run123"
        yield mock_ml


def test_predict_empty_string(mock_pipeline, mock_mlflow):
    with patch("server.pipeline", return_value=mock_pipeline):
        service = SentimentService()
        req = sentiment_pb2.PredictRequest(text="")
        resp = service.Predict(req, None)

        assert resp.label == "POSITIVE"
        assert 0 <= resp.score <= 1


def test_predict_batch_many_texts(mock_pipeline, mock_mlflow):
    with patch("server.pipeline", return_value=mock_pipeline):
        service = SentimentService()
        req = sentiment_pb2.PredictBatchRequest(texts=["texto"] * 1000)
        resp = service.PredictBatch(req, None)

        assert len(resp.labels) == 1000
        assert all(l == "POSITIVE" for l in resp.labels)


def test_predict_invalid_input(mock_pipeline, mock_mlflow):
    with patch("server.pipeline", return_value=mock_pipeline):
        service = SentimentService()

        bad_req = sentiment_pb2.PredictRequest()  # sin text
        resp = service.Predict(bad_req, None)

        assert resp.label == "POSITIVE"
        assert isinstance(resp.score, float)


def test_mlflow_log_model_called(mock_pipeline, mock_mlflow):
    with patch("server.pipeline", return_value=mock_pipeline):
        service = SentimentService()

        # Validar que se llamó a log_model
        assert mock_mlflow.transformers.log_model.called

        # Validar que se loggeó el parámetro huggingface_model_id
        mock_mlflow.log_param.assert_any_call(
            "huggingface_model_id", service.model_id
        )


def test_service_initialization_sets_model_and_clf(mock_pipeline, mock_mlflow):
    with patch("server.pipeline", return_value=mock_pipeline):
        service = SentimentService()

        assert service.model_id == "finiteautomata/beto-sentiment-analysis"
        assert service.clf is not None
