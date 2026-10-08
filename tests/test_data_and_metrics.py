"""Cleaning rules, thin-filer flag, metric functions, and the REQUIRE_DATA switch."""
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import clean
from src.evaluate import ks_stat, psi
from src.features import add_thin_filer_flag
from tests import helpers
from tests.helpers import ROOT, load_yaml

DATA_CFG = load_yaml("data.yaml")


def _raw(rows):
    cols = ["Unnamed: 0", "SeriousDlqin2yrs", "RevolvingUtilizationOfUnsecuredLines", "age",
            "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines",
            "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"]
    return pd.DataFrame(rows, columns=cols).drop(columns="Unnamed: 0")


def test_cleaning_rules():
    raw = _raw([
        [1, 0, 0.5, 40, 1, 0.3, 5000, 5, 0, 1, 0, 2],       # normal
        [2, 1, 0.5, 40, 98, 0.3, 5000, 5, 98, 1, 98, 2],    # special code
        [3, 0, 0.5, 40, 0, 1500, np.nan, 5, 0, 1, 0, 2],    # income missing
        [4, 0, 0.5, 40, 0, 2000, 0, 5, 0, 1, 0, 2],         # income zero
        [5, 0, 50.0, 40, 0, 0.3, 5000, 5, 0, 1, 0, 2],      # utilization outlier
        [6, 0, 0.5, 0, 0, 0.3, 5000, 5, 0, 1, 0, 2],        # age 0
    ])
    df, log = clean(raw, DATA_CFG)
    assert len(df) == 5 and log["age_dropped_rows"] == 1
    pd_cols = DATA_CFG["pastdue_columns"]
    assert df.loc[1, pd_cols].isna().all() and df.loc[1, "pastdue_special_code"] == 1
    assert df.loc[2, "income_missing"] == 1 and np.isnan(df.loc[2, "DebtRatio"])
    assert df.loc[3, "income_zero"] == 1 and df.loc[3, "MonthlyIncome"] == 0 and np.isnan(df.loc[3, "DebtRatio"])
    assert df.loc[4, "util_outlier"] == 1 and np.isnan(df.loc[4, "RevolvingUtilizationOfUnsecuredLines"])
    assert df.loc[0, ["pastdue_special_code", "income_missing", "income_zero", "util_outlier"]].sum() == 0
    assert df.loc[0, "DebtRatio"] == 0.3


def test_thin_filer_rule():
    df = pd.DataFrame({"NumberOfOpenCreditLinesAndLoans": [0, 2, 3, 1],
                       "NumberRealEstateLoansOrLines": [0, 0, 0, 1]})
    out = add_thin_filer_flag(df, load_yaml("simulator.yaml")["thin_filer"]["rule"])
    assert list(out["thin_filer"]) == [1, 1, 0, 0]


def test_ks_and_psi():
    rng = np.random.default_rng(0)
    x = rng.random(10_000)
    assert psi(x, x) < 1e-6
    assert psi(x, np.clip(x + 0.3, 0, 1)) > 0.1
    assert ks_stat(np.r_[np.zeros(50), np.ones(50)], np.r_[np.zeros(50), np.ones(50)]) == 1.0


def test_require_data_switch(monkeypatch):
    missing = ROOT / "data" / "processed" / "does_not_exist.csv"
    monkeypatch.delenv("REQUIRE_DATA", raising=False)
    with pytest.raises(pytest.skip.Exception):
        helpers.require(missing)
    monkeypatch.setenv("REQUIRE_DATA", "1")
    with pytest.raises(pytest.fail.Exception):
        helpers.require(missing)
