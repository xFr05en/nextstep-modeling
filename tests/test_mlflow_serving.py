"""The MLflow-served model must return probabilities, not class labels."""
import os

import mlflow
import numpy as np
import pandas as pd

from src.features import feature_sets
from src.train import log_final_model
from tests.helpers import final_model

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def test_pyfunc_returns_probabilities(tmp_path):
    est = final_model()
    feats = feature_sets()["all"]
    rng = np.random.default_rng(0)
    X = pd.DataFrame(np.abs(rng.normal(size=(20, len(feats)))), columns=feats)
    previous = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        exp = mlflow.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
        with mlflow.start_run(experiment_id=exp) as run:
            log_final_model(est, X.iloc[:5])
        served = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")
        out = np.asarray(served.predict(X))
    finally:
        mlflow.set_tracking_uri(previous)
    assert out.shape == (len(X), 2)
    assert ((out >= 0) & (out <= 1)).all()
    np.testing.assert_allclose(out.sum(axis=1), 1, atol=1e-5)
    assert not np.all(np.isin(out, [0, 1])), "looks like class labels"
    np.testing.assert_allclose(out[:, 1], est.predict_proba(X)[:, 1], atol=1e-6)
