"""Final model: never moves the wrong way on constrained features."""
import pandas as pd
from xgboost import XGBClassifier

from src.features import feature_sets
from src.train import monotonic_violations, monotone_vector
from tests.helpers import final_model, sim_data

FEATS = feature_sets()["all"]


def _sample(n=5000):
    df = sim_data().sample(n, random_state=0)
    return df[FEATS].astype(float), df["SeriousDlqin2yrs"].to_numpy()


def test_final_model_inputs_match_feature_order():
    assert list(final_model().feature_names_in_) == FEATS


def test_final_model_never_moves_wrong_way():
    X, _ = _sample()
    mono = monotone_vector(FEATS)
    assert sum(m != 0 for m in mono) == 13
    violations = monotonic_violations(final_model(), X, FEATS, mono, n_rows=300)
    assert violations and all(v == 0 for v in violations.values()), violations


def test_violation_check_catches_unconstrained_model():
    """Control: the same check must flag an unconstrained model, or test above proves nothing."""
    X, y = _sample()
    model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=0).fit(X, y)
    violations = monotonic_violations(model, X, FEATS, monotone_vector(FEATS), n_rows=300)
    assert sum(violations.values()) > 0
