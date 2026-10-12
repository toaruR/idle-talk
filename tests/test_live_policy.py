"""実機(live)試験が skip で空虚に通過しないことの検査。"""
from pathlib import Path

FORBIDDEN = ["pytest.skip", "skipif", "importorskip", "skip(", "xfail"]


def find_violations(src: str) -> list[str]:
    return [t for t in FORBIDDEN if t in src]


def test_detector_self_check():
    for tok in FORBIDDEN:
        assert tok in find_violations(f"x = 1\n# {tok}\n")
    assert find_violations("def test_a():\n    assert True\n") == []


def test_live_sources_have_no_skip_tokens():
    root = Path(__file__).parent
    bad = {}
    for p in root.glob("*live*.py"):
        if p.name == Path(__file__).name:
            continue  # 検出器自身はトークン定義を含む
        v = find_violations(p.read_text(encoding="utf-8"))
        if v:
            bad[p.name] = v
    assert bad == {}
