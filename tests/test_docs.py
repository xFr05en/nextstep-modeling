"""Every document deliverable carries the disclaimer and exists in Korean and English."""
import pytest

from tests.helpers import ROOT

PHRASES = ("Educational use only", "educational purposes", "교육용", "교육 목적")
DOCS = sorted([*ROOT.glob("README*.md"), *(ROOT / "reports").glob("*.md")])


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_disclaimer_present(path):
    text = path.read_text(encoding="utf-8")
    assert any(p in text for p in PHRASES), f"{path.name} has no disclaimer"


def test_every_report_has_ko_and_en():
    names = {p.name for p in (ROOT / "reports").glob("*.md")}
    for n in names:
        if n.endswith("_en.md"):
            assert n.replace("_en.md", "_ko.md") in names, n
        if n.endswith("_ko.md"):
            assert n.replace("_ko.md", "_en.md") in names, n
