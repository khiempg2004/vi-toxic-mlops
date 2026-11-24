"""Client bất đồng bộ cho máy chủ vLLM (OpenAI-compatible). Dùng chung cho API và đánh giá."""

from __future__ import annotations

import time

import httpx

from vitoxic.labels import build_messages, parse_label


def chat_payload(model: str, text: str, max_chars: int = 512) -> dict:
    return {
        "model": model,
        "messages": build_messages(text, max_chars),
        "temperature": 0.0,
        "max_tokens": 6,
    }


async def classify(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    text: str,
    max_chars: int = 512,
    timeout: float = 60.0,
) -> tuple[str | None, str, float]:
    """Trả về (nhãn hoặc None, đầu ra thô, độ trễ giây)."""
    t0 = time.perf_counter()
    r = await client.post(
        f"{base_url}/v1/chat/completions", json=chat_payload(model, text, max_chars), timeout=timeout
    )
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"].get("content") or ""
    return parse_label(raw), raw, time.perf_counter() - t0


async def list_models(client: httpx.AsyncClient, base_url: str) -> list[str]:
    r = await client.get(f"{base_url}/v1/models", timeout=10.0)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]


async def ensure_adapter_loaded(client: httpx.AsyncClient, base_url: str, name: str, path: str) -> None:
    """Nạp LoRA vào vLLM lúc chạy (cần VLLM_ALLOW_RUNTIME_LORA_UPDATING=True)."""
    if name in await list_models(client, base_url):
        return
    r = await client.post(
        f"{base_url}/v1/load_lora_adapter", json={"lora_name": name, "lora_path": path}, timeout=120.0
    )
    if r.status_code >= 400 and "already" not in r.text.lower():
        raise RuntimeError(f"Không nạp được LoRA {name} từ {path}: {r.status_code} {r.text}")
