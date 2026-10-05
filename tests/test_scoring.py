"""PD -> score -> grade -> approval mapping from config/scoring.yaml."""
import numpy as np

from src.scoring import approve, load_config, pd_to_score, score_frame, score_to_grade

CFG = load_config()


def test_score_within_range_and_decreasing_in_pd():
    pd_ = np.r_[0.0, 1e-9, np.linspace(0.001, 0.999, 500), 1.0]
    s = pd_to_score(pd_, CFG)
    assert s.min() >= CFG["score"]["min"] and s.max() <= CFG["score"]["max"]
    assert (np.diff(s) <= 0).all(), "higher PD must never give a higher score"


def test_grades_follow_score_order():
    s = np.arange(0, 1001)
    g = score_to_grade(s, CFG)
    order = {gr["grade"]: i for i, gr in enumerate(sorted(CFG["grades"], key=lambda x: x["min_score"]))}
    ranks = np.array([order[x] for x in g])
    assert (np.diff(ranks) >= 0).all()
    assert set(g) == {gr["grade"] for gr in CFG["grades"]}


def test_pd_ten_percent_is_the_approval_cutoff():
    f = score_frame(np.array([0.10, 0.105]), CFG)
    assert f.loc[0, "grade"] == "C" and f.loc[0, "approved"] == 1
    assert f.loc[1, "grade"] == "D" and f.loc[1, "approved"] == 0
    c_min = next(g["min_score"] for g in CFG["grades"] if g["grade"] == "C")
    assert abs(int(pd_to_score(np.array([0.10]), CFG)[0]) - c_min) <= 2
    assert list(approve(np.array(list("ABCDE")), CFG)) == [1, 1, 1, 0, 0]


def test_recourse_target_above_approval_cutoff():
    cutoff = min(g["min_score"] for g in CFG["grades"] if g["grade"] in CFG["approval"]["approve_grades"])
    target = CFG["recourse_target_score"]
    assert cutoff < target <= CFG["score"]["max"]
    assert score_to_grade(np.array([target]), CFG)[0] in CFG["approval"]["approve_grades"]
