def test_mlflow_run_created(monkeypatch, mlflow_tmp_env):
    """Instanciar el servicio crea un run y expone run_id en Ping."""
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "ml"))
    import server as srv, Tests.sentiment_pb2 as pb

    # Inyectar pipeline falso para evitar descarga
    class FakePipeline: 
        def __call__(self, x): return [{"label": "POS", "score": 0.99}]
    monkeypatch.setattr(srv, "pipeline", lambda *a, **k: FakePipeline())

    svc = srv.SentimentService()
    pong = svc.Ping(pb.PingRequest(), context=None).status
    assert "run_id=" in pong
