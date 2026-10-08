"""Versioned model file, MLflow tracking location, and MLflow model artifacts."""
import json
import os
import re

import numpy as np
import pytest

from src.features import feature_sets
from src.train import model_path, resolve_tracking_uri
from tests.helpers import ROOT, final_model_path, load_yaml, require

CFG = load_yaml("train.yaml")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def test_versioned_final_model_file_exists():
    path = final_model_path()
    assert path.exists(), path
    assert path == model_path(CFG, json.loads((ROOT / CFG["outputs"]["final_summary"]).read_text())["winner"]["model"])
    assert re.fullmatch(r"[a-z_]+_v\d+\.\d+\.joblib", path.name), path.name
    assert f"_v{CFG['model_version']}." in path.name


def test_no_unversioned_model_files():
    for p in (ROOT / "models").glob("*.joblib"):
        assert re.fullmatch(r"[a-z_]+_v\d+\.\d+\.joblib", p.name), f"unversioned model file: {p.name}"


def test_tracking_uri_from_env(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri, local = resolve_tracking_uri(CFG)
    assert local == ROOT / "mlruns" and uri == f"sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "./mlruns")           # value from .env.example
    assert resolve_tracking_uri(CFG)[1] == ROOT / "mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")  # docker-compose server
    assert resolve_tracking_uri(CFG) == ("http://mlflow:5000", None)


@pytest.fixture(scope="module")
def tracking():
    import mlflow

    require(ROOT / "mlruns" / "mlflow.db")
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}")
    yield mlflow
    mlflow.set_tracking_uri(previous)


def test_one_experiment_per_algorithm_with_model_artifact(tracking):
    mlflow = tracking
    for key in CFG["model_names"]:
        exp = mlflow.get_experiment_by_name(CFG["mlflow"]["experiments"][key])
        assert exp is not None, key
        runs = mlflow.search_runs([exp.experiment_id], output_format="list")
        assert len(runs) >= 9, key
        models = mlflow.search_logged_models(experiment_ids=[exp.experiment_id], output_format="list")
        assert models, f"no model artifact in {exp.name}"
        for run in runs:
            assert {"auc_mean", "ks_mean", "f1_mean"} <= set(run.data.metrics) or "mono" in run.info.run_name


def test_final_model_loads_from_mlflow_and_predicts_input_example(tracking):
    mlflow = tracking
    summary = json.loads((ROOT / CFG["outputs"]["final_summary"]).read_text())
    uri = f"runs:/{summary['mlflow_run_id']}/model"
    info = mlflow.models.get_model_info(uri)
    assert info.signature is not None
    assert [c.name for c in info.signature.inputs.inputs] == feature_sets()["all"]
    model = mlflow.pyfunc.load_model(uri)
    example = model.input_example
    assert example is not None and list(example.columns) == feature_sets()["all"]
    out = np.asarray(model.predict(example))
    assert out.shape == (len(example), 2) and np.allclose(out.sum(axis=1), 1, atol=1e-5)
