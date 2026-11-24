"""FastAPI bọc vLLM: /predict, /predict/batch, /health, /model, /metrics.

- Tự theo dõi models/registry/production.json -> chuyển LoRA không cần restart.
- Ghi log từng yêu cầu (JSONL) phục vụ phát hiện trôi dữ liệu.
- Dự phòng TF-IDF khi vLLM lỗi hoặc đầu ra không hợp lệ.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from vitoxic.client import classify, ensure_adapter_loaded
from vitoxic.labels import LABELS

# ---------------------------------------------------------------- settings


@dataclass
class Settings:
    vllm_url: str = field(default_factory=lambda: os.environ.get("VLLM_URL", "http://localhost:8001"))
    base_model: str = field(default_factory=lambda: os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    models_dir: Path = field(default_factory=lambda: Path(os.environ.get("MODELS_DIR", "models")))
    vllm_models_dir: str = field(default_factory=lambda: os.environ.get("VLLM_MODELS_DIR", "/models"))
    log_path: Path = field(default_factory=lambda: Path(os.environ.get("REQUEST_LOG", "logs/requests.jsonl")))
    drift_report: Path = field(
        default_factory=lambda: Path(os.environ.get("DRIFT_REPORT", "reports/drift_latest.json"))
    )
    max_chars: int = field(default_factory=lambda: int(os.environ.get("MAX_CHARS", "512")))
    timeout_s: float = field(default_factory=lambda: float(os.environ.get("VLLM_TIMEOUT", "30")))


# ---------------------------------------------------------------- metrics

PREDICTIONS = Counter("vitoxic_predictions_total", "Số dự đoán", ["label", "model_version", "source"])
LATENCY = Histogram(
    "vitoxic_request_latency_seconds", "Độ trễ yêu cầu", ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1, 2, 5),
)
ERRORS = Counter("vitoxic_errors_total", "Lỗi", ["kind"])
MODEL_INFO = Gauge("vitoxic_model_info", "Phiên bản mô hình đang phục vụ", ["version"])
TEXT_LEN = Histogram("vitoxic_input_chars", "Độ dài bình luận", buckets=(10, 25, 50, 100, 200, 400, 800))
DRIFT_DETECTED = Gauge("vitoxic_drift_detected", "1 nếu báo cáo drift gần nhất phát hiện trôi")
DRIFT_SHARE = Gauge("vitoxic_drift_share", "Tỉ lệ đặc trưng bị trôi")
FEATURE_DRIFT = Gauge("vitoxic_feature_drift_statistic", "Thống kê KS từng đặc trưng", ["feature"])
FEATURE_MEAN = Gauge("vitoxic_feature_mean", "Giá trị TB đặc trưng (current/reference)", ["feature", "window"])


# ---------------------------------------------------------------- schemas


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=256)


class Prediction(BaseModel):
    label: str
    model_version: str
    source: str
    latency_ms: float


# ---------------------------------------------------------------- state


class ModelResolver:
    """Xác định model cần gọi trên vLLM dựa vào registry (cache theo mtime)."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.s = settings
        self.client = client
        self._mtime: float | None = None
        self._entry: dict | None = None
        self._loaded: set[str] = set()
        self._lock = asyncio.Lock()
        self._current_version: str | None = None

    def _read_registry(self) -> dict | None:
        f = self.s.models_dir / "registry" / "production.json"
        try:
            mtime = f.stat().st_mtime
        except FileNotFoundError:
            self._entry, self._mtime = None, None
            return None
        if mtime != self._mtime:
            self._entry, self._mtime = json.loads(f.read_text()), mtime
        return self._entry

    def _set_version(self, version: str):
        if version != self._current_version:
            MODEL_INFO.clear()
            MODEL_INFO.labels(version=version).set(1)
            self._current_version = version

    def invalidate(self, name: str):
        self._loaded.discard(name)

    async def resolve(self) -> tuple[str, str]:
        entry = self._read_registry()
        if entry is None:
            self._set_version("base-zeroshot")
            return self.s.base_model, "base-zeroshot"
        name = entry["adapter_name"]
        if name not in self._loaded:
            async with self._lock:
                if name not in self._loaded:
                    await ensure_adapter_loaded(
                        self.client, self.s.vllm_url, name, f"{self.s.vllm_models_dir}/{entry['adapter_path']}"
                    )
                    self._loaded.add(name)
        self._set_version(entry["run_id"])
        return name, entry["run_id"]


class Fallback:
    def __init__(self, models_dir: Path):
        self.path = models_dir / "baseline" / "model.joblib"
        self._model = None

    @property
    def available(self) -> bool:
        return self.path.exists()

    def predict(self, texts: list[str]) -> list[str]:
        if self._model is None:
            import joblib

            self._model = joblib.load(self.path)
        return list(self._model.predict(texts))


class RequestLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def write(self, records: list[dict]):
        async with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")


class DriftGauges:
    def __init__(self, path: Path):
        self.path = path
        self._mtime = None

    def refresh(self):
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            return
        if mtime == self._mtime:
            return
        self._mtime = mtime
        rep = json.loads(self.path.read_text())
        DRIFT_DETECTED.set(1 if rep.get("drift_detected") else 0)
        DRIFT_SHARE.set(rep.get("drift_share", 0))
        for feat, info in rep.get("features", {}).items():
            FEATURE_DRIFT.labels(feature=feat).set(info.get("statistic", 0))
            FEATURE_MEAN.labels(feature=feat, window="current").set(info.get("current_mean", 0))
            FEATURE_MEAN.labels(feature=feat, window="reference").set(info.get("reference_mean", 0))


# ---------------------------------------------------------------- app


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        limits = httpx.Limits(max_connections=256, max_keepalive_connections=64)
        async with httpx.AsyncClient(limits=limits) as client:
            app.state.client = client
            app.state.resolver = ModelResolver(s, client)
            app.state.fallback = Fallback(s.models_dir)
            app.state.logger = RequestLogger(s.log_path)
            app.state.drift = DriftGauges(s.drift_report)
            yield

    app = FastAPI(title="vi-toxic moderation API", version="0.1.0", lifespan=lifespan)

    async def _predict_one(text: str) -> Prediction:
        t0 = time.perf_counter()
        resolver: ModelResolver = app.state.resolver
        fallback: Fallback = app.state.fallback
        label, version, source = None, "unknown", "llm"
        vllm_failed = False
        try:
            for attempt in range(2):
                name, version = await resolver.resolve()
                try:
                    label, _, _ = await classify(app.state.client, s.vllm_url, name, text, s.max_chars, s.timeout_s)
                    break
                except httpx.HTTPStatusError as e:
                    # vLLM khởi động lại -> mất LoRA đã nạp: nạp lại một lần.
                    if e.response.status_code == 404 and attempt == 0:
                        resolver.invalidate(name)
                        continue
                    raise
            if label is None:
                ERRORS.labels(kind="invalid_output").inc()
        except Exception:  # noqa: BLE001
            ERRORS.labels(kind="vllm").inc()
            label, vllm_failed = None, True
        if label is None:
            if fallback.available:
                label, source = fallback.predict([text])[0], "baseline-fallback"
            elif not vllm_failed:
                label = LABELS[0]  # LLM trả lời nhưng không parse được
            else:
                raise HTTPException(status_code=503, detail="vLLM không khả dụng và không có mô hình dự phòng")
        latency = time.perf_counter() - t0
        PREDICTIONS.labels(label=label, model_version=version, source=source).inc()
        TEXT_LEN.observe(len(text))
        return Prediction(label=label, model_version=version, source=source, latency_ms=latency * 1000)

    async def _log(texts: list[str], preds: list[Prediction]):
        ts = datetime.now(UTC).isoformat()
        await app.state.logger.write(
            [{"ts": ts, "text": t, **p.model_dump()} for t, p in zip(texts, preds, strict=True)]
        )

    @app.post("/predict", response_model=Prediction)
    async def predict(req: PredictRequest):
        with LATENCY.labels(endpoint="predict").time():
            pred = await _predict_one(req.text)
        await _log([req.text], [pred])
        return pred

    @app.post("/predict/batch", response_model=list[Prediction])
    async def predict_batch(req: BatchRequest):
        with LATENCY.labels(endpoint="predict_batch").time():
            preds = await asyncio.gather(*(_predict_one(t) for t in req.texts))
        await _log(req.texts, list(preds))
        return preds

    @app.get("/health")
    async def health():
        try:
            r = await app.state.client.get(f"{s.vllm_url}/health", timeout=5)
            vllm_ok = r.status_code == 200
        except Exception:  # noqa: BLE001
            vllm_ok = False
        return {"status": "ok" if vllm_ok else "degraded", "vllm": vllm_ok, "fallback": app.state.fallback.available}

    @app.get("/model")
    async def model_info():
        entry = app.state.resolver._read_registry()
        return {"production": entry, "base_model": s.base_model}

    @app.get("/metrics")
    async def metrics():
        app.state.drift.refresh()
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
