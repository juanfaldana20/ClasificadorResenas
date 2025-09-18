def test_stubs_import():
    """Verifica que los stubs generados existen y cargan."""
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "ml"))
    import sentiment_pb2, sentiment_pb2_grpc  # noqa: F401
