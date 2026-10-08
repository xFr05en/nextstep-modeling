"""The held-out fold must never reach a fit step (imputation, scaling, SMOTE, LR caps)."""
import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE

from src.features import feature_sets
from src.data.preprocessor import LogCap
from src.train import build_pipeline, make_folds
from tests.helpers import load_yaml

CFG = load_yaml("train.yaml")
FEATS = feature_sets()["all"]


@pytest.fixture(scope="module")
def split():
    """Small synthetic data; held-out rows carry canary values that would shift any fit."""
    rng = np.random.default_rng(0)
    n_tr, n_te = 400, 200
    X = pd.DataFrame(rng.normal(size=(n_tr + n_te, len(FEATS))), columns=FEATS)
    y = np.r_[rng.random(n_tr) < 0.2, rng.random(n_te) < 0.2].astype(int)
    te = np.arange(n_tr, n_tr + n_te)
    X.loc[te, "MonthlyIncome"] = 1e9                      # extreme canary
    X.loc[te[::2], "DebtRatio"] = np.nan                  # many NaNs only in held-out rows
    X.loc[te, "age"] = 500.0
    X.loc[:5, "DebtRatio"] = np.nan                       # a few NaNs in training rows too
    return X.iloc[:n_tr], y[:n_tr], X.iloc[n_tr:], y[n_tr:]


def test_imputer_uses_training_rows_only(split):
    X_tr, y_tr, X_te, _ = split
    pipe = build_pipeline("xgb", FEATS, "smote", CFG).fit(X_tr, y_tr)
    stats = pipe.named_steps["impute"].statistics_
    np.testing.assert_allclose(stats, np.nanmedian(X_tr.to_numpy(), axis=0))
    full = np.nanmedian(pd.concat([X_tr, X_te]).to_numpy(), axis=0)
    assert not np.allclose(stats, full), "canaries should make full-data medians differ"


def test_scaler_uses_training_rows_only(split):
    X_tr, y_tr, _, _ = split
    pipe = build_pipeline("lr", FEATS, "none", CFG).fit(X_tr, y_tr)
    imputed = pipe.named_steps["impute"].transform(X_tr)
    transformed = pipe.named_steps["logcap"].transform(imputed)
    scaler = pipe.named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, transformed.mean(axis=0))
    np.testing.assert_allclose(scaler.scale_, transformed.std(axis=0))


class SpySMOTE(SMOTE):
    calls: list[int] = []

    def fit_resample(self, X, y, **kw):
        SpySMOTE.calls.append(len(X))
        return super().fit_resample(X, y, **kw)


def test_smote_sees_training_rows_only_and_never_at_predict(split):
    X_tr, y_tr, X_te, _ = split
    pipe = build_pipeline("lgbm", FEATS, "smote", CFG)
    i = [n for n, _ in pipe.steps].index("smote")
    pipe.steps[i] = ("smote", SpySMOTE(random_state=0))
    SpySMOTE.calls = []
    pipe.fit(X_tr, y_tr)
    assert SpySMOTE.calls == [len(X_tr)]
    p = pipe.predict_proba(X_te)
    assert SpySMOTE.calls == [len(X_tr)], "SMOTE must not run at prediction time"
    assert p.shape == (len(X_te), 2)


def test_lr_log_cap_learned_on_training_rows_only(split):
    X_tr, _, X_te, _ = split
    col = FEATS.index("MonthlyIncome")
    lc = LogCap(cols=[col], q=0.99).fit(X_tr.fillna(0))
    expected = np.quantile(np.log1p(np.clip(X_tr.fillna(0).to_numpy()[:, col], 0, None)), 0.99)
    assert lc.caps_[col] == pytest.approx(expected)
    assert lc.transform(X_te.fillna(0))[:, col].max() <= expected + 1e-12  # canary 1e9 is capped


def test_folds_disjoint_complete_and_stratified():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({CFG["target"]: (rng.random(2000) < 0.07).astype(int),
                       CFG["thin_col"]: (rng.random(2000) < 0.08).astype(int)})
    folds = make_folds(df, CFG)
    k, repeats = CFG["n_splits"], CFG.get("cv_repeats", 1)
    assert len(folds) == k * repeats
    for r in range(repeats):                                 # every row held out exactly once per repeat
        held = np.concatenate([te for _, te in folds[r * k:(r + 1) * k]])
        assert sorted(held) == list(range(len(df)))
    for tr, te in folds:
        assert not set(tr) & set(te)
        assert abs(df.iloc[te][CFG["target"]].mean() - df[CFG["target"]].mean()) < 0.01


def test_no_target_or_protected_columns_in_features():
    fs = feature_sets()
    for name, cols in fs.items():
        for banned in (CFG["target"], "gender_female", CFG["thin_col"]):
            assert banned not in cols, f"{banned} in feature set {name}"
    act = load_yaml("actionability.yaml")["features"]
    expected = [v for v in load_yaml("simulator.yaml")["variables"] if act[v].get("model_feature", True)]
    assert fs["alt"] == expected
