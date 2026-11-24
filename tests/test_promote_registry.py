from vitoxic.promote import decide
from vitoxic.registry import get_production, set_production


def test_decide():
    assert decide(0.6, None, 0.005)[0]
    assert decide(0.70, 0.69, 0.005)[0]
    assert not decide(0.692, 0.69, 0.005)[0]
    assert not decide(0.60, 0.69, 0.005)[0]


def test_registry_roundtrip(tmp_path):
    assert get_production(tmp_path) is None
    set_production(tmp_path, "run-a", {"m": 1})
    e = set_production(tmp_path, "run-b", {"m": 2})
    assert e["previous_run_id"] == "run-a"
    prod = get_production(tmp_path)
    assert prod["adapter_name"] == "vitoxic-run-b" and prod["adapter_path"] == "candidates/run-b"
    assert len((tmp_path / "registry" / "history.jsonl").read_text().splitlines()) == 2
