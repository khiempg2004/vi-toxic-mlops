"""Kiểm thử API với vLLM giả lập (httpx MockTransport) - chạy được trong CI không cần GPU."""

import json

import httpx
import joblib
import pytest
from fastapi.testclient import TestClient

from vitoxic import baseline
from vitoxic.registry import set_production
from vitoxic.serving.app import Settings, create_app

from .conftest import synthetic_frame


def fake_vllm(answer="HATE", fail=False, loaded=None):
    loaded = loaded if loaded is not None else set()
    calls = {"chat": [], "load": []}

    def handler(request: httpx.Request):
        if fail:
            raise httpx.ConnectError("down")
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "base"}, *({"id": n} for n in loaded)]})
        if path == "/v1/load_lora_adapter":
            body = json.loads(request.content)
            loaded.add(body["lora_name"])
            calls["load"].append(body)
            return httpx.Response(200, text="ok")
        if path == "/v1/chat/completions":
            body = json.loads(request.content)
            calls["chat"].append(body["model"])
            return httpx.Response(200, json={"choices": [{"message": {"content": answer}}]})
        if path == "/health":
            return httpx.Response(200)
        return httpx.Response(404)

    return httpx.MockTransport(handler), calls


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    def _make(transport, with_baseline=False):
        settings = Settings(
            vllm_url="http://vllm", base_model="base", models_dir=tmp_path / "models",
            log_path=tmp_path / "logs" / "req.jsonl", drift_report=tmp_path / "drift.json",
        )
        if with_baseline:
            df = synthetic_frame(30)
            pipe = baseline.build_pipeline(max_iter=300, min_df=1).fit(df["text"], df["label"])
            (tmp_path / "models" / "baseline").mkdir(parents=True)
            joblib.dump(pipe, tmp_path / "models" / "baseline" / "model.joblib")
        real = httpx.AsyncClient

        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=transport))
        return TestClient(create_app(settings)), settings

    return _make


def test_predict_zero_shot_and_log(make_client):
    transport, calls = fake_vllm("HATE")
    client, s = make_client(transport)
    with client:
        r = client.post("/predict", json={"text": "thằng ngu"})
        assert r.status_code == 200
        body = r.json()
        assert body["label"] == "HATE" and body["model_version"] == "base-zeroshot"
        assert calls["chat"] == ["base"]
        assert json.loads(s.log_path.read_text().splitlines()[0])["text"] == "thằng ngu"
        assert "vitoxic_predictions_total" in client.get("/metrics").text


def test_predict_uses_production_adapter(make_client):
    transport, calls = fake_vllm("CLEAN")
    client, s = make_client(transport)
    set_production(s.models_dir, "run-x")
    with client:
        r = client.post("/predict/batch", json={"texts": ["a", "b", "c"]})
        assert [p["model_version"] for p in r.json()] == ["run-x"] * 3
        assert calls["load"] == [{"lora_name": "vitoxic-run-x", "lora_path": "/models/candidates/run-x"}]
        assert set(calls["chat"]) == {"vitoxic-run-x"}


def test_fallback_when_vllm_down(make_client):
    transport, _ = fake_vllm(fail=True)
    client, _ = make_client(transport, with_baseline=True)
    with client:
        r = client.post("/predict", json={"text": "cảm ơn bạn đã chia sẻ"})
        assert r.status_code == 200 and r.json()["source"] == "baseline-fallback"
        assert client.get("/health").json()["status"] == "degraded"


def test_503_without_fallback(make_client):
    transport, _ = fake_vllm(fail=True)
    client, _ = make_client(transport)
    with client:
        assert client.post("/predict", json={"text": "x"}).status_code == 503


def test_validation(make_client):
    transport, _ = fake_vllm()
    client, _ = make_client(transport)
    with client:
        assert client.post("/predict", json={"text": ""}).status_code == 422
