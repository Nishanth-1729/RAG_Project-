"""LLM client for any OpenAI-compatible endpoint (default: local vLLM serving Qwen on the GPU).

* schema-constrained JSON (vLLM guided decoding / OpenAI json_schema) with repair fallbacks
* deterministic-call disk cache (the same prompt never costs compute twice)
* thread-pool batching (vLLM does continuous batching on the server)
* token / latency / cost tracing without logging secrets
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence, Type, TypeVar

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

from ..config import settings
from ..utils.logging import current_run, get_logger, trace
from ..utils.text import approx_tokens

log = get_logger("llm")
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class UsageMeter:
    def __init__(self):
        self.lock = threading.Lock()
        self.prompt = self.completion = self.calls = self.cached = 0
        self.latency = 0.0

    def add(self, p: int, c: int, lat: float, cached: bool):
        with self.lock:
            self.calls += 1
            self.cached += int(cached)
            if not cached:
                self.prompt += p
                self.completion += c
                self.latency += lat

    def snapshot(self) -> dict:
        with self.lock:
            cost = (self.prompt * settings.llm_price_in + self.completion * settings.llm_price_out) / 1e6
            return {"calls": self.calls, "cached_calls": self.cached, "prompt_tokens": self.prompt,
                    "completion_tokens": self.completion, "llm_seconds": round(self.latency, 2), "cost_usd": round(cost, 5)}


meter = UsageMeter()


def _extract_json(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if m:
        cand = m.group(1)
        for fix in (lambda s: s, lambda s: re.sub(r",\s*([}\]])", r"\1", s), lambda s: s.replace("\n", " ")):
            try:
                return json.loads(fix(cand))
            except json.JSONDecodeError:
                continue
    raise LLMError(f"could not parse JSON from model output: {text[:200]!r}")


class LLM:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        self.base_url = base_url or settings.llm_base_url
        self.model = model or settings.llm_model
        self.client = OpenAI(base_url=self.base_url, api_key=api_key or settings.llm_api_key,
                             timeout=settings.llm_timeout, max_retries=0)
        self.cache_dir = settings.cache_dir / "llm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._served: str | None = None
        self.use_cache = True          # switch off for clean latency measurements

    # ------------------------------------------------------------------ status
    def is_up(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:  # noqa: BLE001
            return False

    def served_model(self) -> str:
        if self._served is None:
            try:
                ids = [m.id for m in self.client.models.list().data]
                self._served = self.model if self.model in ids else (ids[0] if ids else self.model)
            except Exception:  # noqa: BLE001
                self._served = self.model
        return self._served

    # -------------------------------------------------------------------- core
    def chat(self, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 1024,
             json_schema: dict | None = None, cache: bool = True, stop: list[str] | None = None,
             tag: str = "chat") -> str:
        key = None
        if cache and self.use_cache and temperature == 0.0:
            raw = json.dumps([self.model, messages, max_tokens, json_schema, stop], sort_keys=True, ensure_ascii=False)
            key = self.cache_dir / (hashlib.sha256(raw.encode()).hexdigest() + ".json")
            if key.exists():
                meter.add(0, 0, 0.0, True)
                return json.loads(key.read_text())["text"]
        kwargs: dict[str, Any] = dict(model=self.served_model(), messages=messages, temperature=temperature,
                                      max_tokens=max_tokens)
        if stop:
            kwargs["stop"] = stop
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "output", "schema": json_schema, "strict": True}}
        last: Exception | None = None
        for attempt in range(4):
            t0 = time.perf_counter()
            try:
                r = self.client.chat.completions.create(**kwargs)
                text = r.choices[0].message.content or ""
                u = r.usage
                lat = time.perf_counter() - t0
                p, c = (u.prompt_tokens, u.completion_tokens) if u else (approx_tokens(str(messages)), approx_tokens(text))
                meter.add(p, c, lat, False)
                trace("llm_call", tag=tag, prompt_tokens=p, completion_tokens=c, latency_s=round(lat, 3), status="ok")
                if key is not None and text:
                    key.write_text(json.dumps({"text": text}, ensure_ascii=False))
                return text
            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last = e
                trace("llm_call", tag=tag, status="retry", error=repr(e)[:200])
                time.sleep(1.5 * (2 ** attempt))
            except Exception as e:  # noqa: BLE001
                msg = repr(e)
                if json_schema is not None and ("response_format" in msg or "json_schema" in msg or "guided" in msg):
                    kwargs.pop("response_format", None)      # backend without schema support: fall back to prompt-only JSON
                    json_schema = None
                    last = e
                    continue
                trace("llm_call", tag=tag, status="error", error=msg[:300])
                raise LLMError(msg) from e
        raise LLMError(f"LLM unavailable after retries: {last!r}")

    def complete(self, system: str, user: str, **kw) -> str:
        return self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)

    def generate(self, schema: Type[T], system: str, user: str, *, temperature: float = 0.0,
                 max_tokens: int = 1500, retries: int = 2, tag: str = "generate") -> T:
        """Structured output validated by pydantic. Retries once or twice, feeding the error back."""
        js = schema.model_json_schema()
        sys_msg = system + "\n\nReturn ONLY a JSON object that matches the required schema. No prose, no markdown."
        msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]
        last: Exception | None = None
        for attempt in range(retries + 1):
            text = self.chat(msgs, temperature=temperature if attempt == 0 else min(0.4, 0.2 + 0.2 * attempt),
                             max_tokens=max_tokens, json_schema=js, cache=(attempt == 0), tag=tag)
            try:
                return schema.model_validate(_extract_json(text))
            except (LLMError, ValidationError, ValueError) as e:
                last = e
                msgs = msgs + [{"role": "assistant", "content": text},
                               {"role": "user", "content": f"Invalid output ({str(e)[:300]}). Return corrected JSON only."}]
        raise LLMError(f"structured generation failed for {schema.__name__}: {last!r}")

    def generate_or_none(self, schema: Type[T], system: str, user: str, **kw) -> T | None:
        try:
            return self.generate(schema, system, user, **kw)
        except LLMError as e:
            log.warning("generation failed (%s): %s", schema.__name__, str(e)[:200])
            return None

    # ------------------------------------------------------------------- batch
    def map(self, fn, items: Sequence, workers: int = 32) -> list:
        """Run fn(item) over items concurrently (vLLM batches the requests server side)."""
        run = current_run()

        def wrapped(x):
            from ..utils.logging import set_run
            set_run(run)
            try:
                return fn(x)
            except Exception as e:  # noqa: BLE001
                log.warning("task failed: %r", e)
                return e
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(wrapped, items))


_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm
