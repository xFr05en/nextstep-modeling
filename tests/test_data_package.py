"""src/data package: loader schema printout, mission-style simulator wrapper, ANOVA handoff file."""
import pandas as pd
import pytest

from src.data.loader import load_gmsc
from src.data.simulator import generate_alternative_data, load_config, simulate
from tests.helpers import ROOT, clean_data, load_yaml, require


def test_loader_prints_columns_dtypes_and_missing_ratios(capsys):
    require(ROOT / load_yaml("data.yaml")["raw_path"])
    df = load_gmsc()
    out = capsys.readouterr().out
    assert "column" in out and "dtype" in out and "missing_ratio" in out
    for col in df.columns:
        assert col in out, col
    assert "float64" in out and "int64" in out
    assert "0.1982" in out                       # MonthlyIncome missing ratio in the raw file


def test_generate_alternative_data_wraps_simulate():
    sample = clean_data().sample(3000, random_state=2).reset_index(drop=True)
    cfg = load_config()
    a = generate_alternative_data(sample)
    b, _ = simulate(sample, cfg, thin_filer_ratio=None, bias_ratio=0.0)
    pd.testing.assert_frame_equal(a, b)
    for n in ("telecom_payment_rate", "utility_payment_rate", "spending_consistency",
              "regular_payment_count", "app_login_frequency"):
        assert n in a.columns
    assert "0.3" in generate_alternative_data.__doc__ and "0.1" in generate_alternative_data.__doc__


def test_cv_fold_auc_has_all_segments_for_every_run_and_fold():
    path = require(ROOT / load_yaml("train.yaml")["outputs"]["cv_fold_auc"])
    d = pd.read_csv(path)
    assert list(d.columns) == ["run", "model", "features", "resampling", "repeat", "fold",
                               "segment", "n_rows", "n_defaults", "auc"]
    assert set(d["features"]) == {"gmsc", "alt", "both"} and set(d["model"]) == {"lr", "xgb", "lgbm"}
    segs = d.groupby(["run", "repeat", "fold"])["segment"].apply(frozenset)
    assert (segs == frozenset({"all", "thin", "general"})).all()
    assert len(segs) == 27 * load_yaml("train.yaml")["n_splits"] * load_yaml("train.yaml").get("cv_repeats", 1)
    w = d.pivot_table(index=["run", "repeat", "fold"], columns="segment", values="n_rows")
    assert (w["thin"] + w["general"] == w["all"]).all()
    assert d["auc"].between(0.5, 1).all() and (d["n_defaults"] > 0).all()


def test_old_module_paths_are_gone():
    assert not (ROOT / "src" / "data.py").exists() and not (ROOT / "src" / "simulator.py").exists()
    for f in ("loader.py", "preprocessor.py", "simulator.py"):
        assert (ROOT / "src" / "data" / f).exists(), f
