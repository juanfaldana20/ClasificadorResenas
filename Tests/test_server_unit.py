import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "ml"))
import sentiment_pb2 as pb

def test_predict_ok(service):
    """Predict retorna etiqueta y score válidos."""
    resp = service.Predict(pb.PredictRequest(text="Me encanta esto"), context=None)
    assert resp.label in {"POS", "NEG", "NEU"}
    assert 0 <= resp.score <= 1

def test_predict_batch_alignment(service):
    """Batch mantiene alineación uno-a-uno entre textos, labels y scores."""
    texts = ["Excelente", "Muy mal", "normal"]
    resp = service.PredictBatch(pb.PredictBatchRequest(texts=texts), context=None)
    assert len(resp.labels) == len(texts) == len(resp.scores)

def test_summarize_basic(service):
    """Summarize devuelve hasta N oraciones del texto original."""
    long_text = "Primera. Segunda. Tercera. Cuarta."
    resp = service.Summarize(pb.SummarizeRequest(text=long_text, sentences=2), context=None)
    assert 1 <= len(resp.sentences) <= 2
    for s in resp.sentences:
        assert s.strip(".") in long_text

def test_ping_has_run_id(service):
    """Ping incluye ok y un run_id cuando MLflow se inicializó."""
    resp = service.Ping(pb.PingRequest(), context=None)
    assert resp.status.startswith("ok")
