"""German Credit secondary model: decoding, feature exclusions, constraints, leakage, OOF file."""
import numpy as np
import pandas as pd
import pytest

from src.german import ROOT, build_pipeline, decode, feature_lists, load_config, load_raw, onehot_name
from tests.helpers import require

CFG = load_config()


def _raw_rows():
    """Two handmade rows in file format: one bad female, one good male."""
    a = ["A11", 6, "A34", "A43", 1169, "A65", "A75", 4, "A92", "A101", 4, "A121", 67, "A143", "A152", 2,
         "A173", 1, "A192", "A201", 2]
    b = ["A14", 12, "A32", "A40", 2000, "A61", "A73", 2, "A93", "A101", 2, "A123", 35, "A143", "A151", 1,
         "A172", 1, "A191", "A202", 1]
    return pd.DataFrame([a, b], columns=CFG["columns"])


def test_decode_labels_target_and_sex():
    df = decode(_raw_rows(), CFG)
    assert list(df["default"]) == [1, 0]                      # original 2 = bad -> 1
    assert list(df["sex"]) == ["female", "male"]              # A92 female, A93 male
    assert df.loc[0, "checking_status"] == "< 0 DM"
    assert df.loc[1, "foreign_worker"] == "no"
    assert "target" not in df.columns


def test_decode_rejects_unknown_codes():
    raw = _raw_rows()
    raw.loc[0, "purpose"] = "A499"
    with pytest.raises(ValueError, match="purpose"):
        decode(raw, CFG)


def test_labels_are_strings_and_only_a92_is_female():
    for col, mapping in CFG["labels"].items():
        for code, label in mapping.items():
            assert isinstance(code, str) and code.startswith("A") and isinstance(label, str), (col, code)
    assert CFG["female_codes"] == ["A92"]


def test_sex_and_foreign_worker_excluded_age_kept():
    numeric, categorical = feature_lists(CFG)
    feats = numeric + categorical
    assert "personal_status_sex" not in feats and "foreign_worker" not in feats and "sex" not in feats
    assert "age" in feats and "default" not in feats
    assert len(feats) == 18


def test_constraints_only_on_numeric_with_clear_direction():
    numeric, _ = feature_lists(CFG)
    assert set(CFG["monotone"]) <= set(numeric)
    assert CFG["monotone"] == {"duration_months": 1, "installment_rate": 1}
    assert "age" not in CFG["monotone"] and "credit_amount" not in CFG["monotone"]


def test_encoder_and_scaler_fit_on_training_rows_only():
    rng = np.random.default_rng(0)
    df = decode(pd.concat([_raw_rows()] * 30, ignore_index=True), CFG)
    df["default"] = rng.integers(0, 2, len(df))
    numeric, categorical = feature_lists(CFG)
    X = df[numeric + categorical].copy()
    tr, te = np.arange(40), np.arange(40, 60)
    X.loc[te, "purpose"] = "retraining"                       # category that appears only in held-out rows
    X.loc[te, "duration_months"] = 999
    pipe = build_pipeline("lr", CFG).fit(X.iloc[tr], df["default"].iloc[tr])
    prep = pipe.named_steps["prep"]
    cats = dict(zip(categorical, prep.named_transformers_["cat"].categories_))
    assert "retraining" not in cats["purpose"]
    scaler = prep.named_transformers_["num"]
    np.testing.assert_allclose(scaler.mean_, X.iloc[tr][numeric].mean().to_numpy())
    assert pipe.predict_proba(X.iloc[te]).shape == (len(te), 2)


def test_onehot_names_are_xgboost_safe():
    for col, mapping in CFG["labels"].items():
        for label in mapping.values():
            name = onehot_name(col, label)
            assert not any(ch in name for ch in "[]<"), name


def test_constrained_model_never_moves_wrong_way():
    df = decode(load_raw(CFG, require(ROOT / CFG["raw_path"])), CFG)
    numeric, categorical = feature_lists(CFG)
    X, y = df[numeric + categorical], df["default"]
    pipe = build_pipeline("xgb", CFG).fit(X, y)
    sample = X.sample(200, random_state=0)
    for col, sign in CFG["monotone"].items():
        preds = []
        for v in np.unique(np.quantile(X[col], np.linspace(0, 1, 12))):
            s = sample.copy()
            s[col] = v
            preds.append(pipe.predict_proba(s)[:, 1])
        assert ((np.diff(np.vstack(preds), axis=0) * sign) >= -1e-9).all(), col


def test_oof_file_for_fairness_check():
    path = ROOT / CFG["outputs"]["oof"]
    assert path.exists(), "committed file missing; rerun python -m src.german"
    oof = pd.read_csv(path)
    assert list(oof.columns) == ["row_id", "pd_xgb", "pd_lr", "default", "sex", "age",
                                 "personal_status", "foreign_worker"]
    assert not any(c in oof.columns for c in ("approved", "decision", "cutoff", "grade"))
    assert len(oof) == 1000 and oof["row_id"].is_unique
    assert oof[["pd_xgb", "pd_lr"]].stack().between(0, 1).all()
    assert oof["default"].mean() == pytest.approx(0.30)
    assert oof["sex"].value_counts().to_dict() == {"male": 690, "female": 310}
