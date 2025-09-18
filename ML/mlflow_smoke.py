# ml/mlflow_smoke.py
import os, mlflow, tempfile, pathlib
os.environ.setdefault("MLFLOW_TRACKING_URI","file:./mlruns")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME","beto-sentiment")
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))
with mlflow.start_run(run_name="smoke"):
    mlflow.log_param("ok", True)
    p = pathlib.Path(tempfile.gettempdir())/"dummy.txt"
    p.write_text("hello")
    mlflow.log_artifact(str(p))
    print("artifact_uri:", mlflow.get_artifact_uri())
