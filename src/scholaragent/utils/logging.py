"""Structured JSONL tracing of LLM calls / tools / agents (never logs secrets)."""
from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from ..config import settings

_lock = threading.Lock()
_current_run: ContextVar[str | None] = ContextVar("current_run", default=None)
_current_agent: ContextVar[str | None] = ContextVar("current_agent", default=None)

_SECRET_KEYS = {"api_key", "authorization", "password", "token", "secret"}


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"scholar.{name}")
    if not logging.getLogger("scholar").handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        root = logging.getLogger("scholar")
        root.addHandler(h)
        root.setLevel(logging.INFO)
    return logger


def _scrub(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: ("<redacted>" if k.lower() in _SECRET_KEYS else _scrub(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    if isinstance(obj, str) and len(obj) > 2000:
        return obj[:2000] + f"...<{len(obj)} chars>"
    return obj


def set_run(run_id: str | None):
    return _current_run.set(run_id)


def current_run() -> str | None:
    return _current_run.get()


@contextmanager
def agent_scope(name: str) -> Iterator[None]:
    tok = _current_agent.set(name)
    try:
        yield
    finally:
        _current_agent.reset(tok)


def trace(event: str, **fields: Any) -> None:
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, "run": _current_run.get(),
           "agent": _current_agent.get(), **_scrub(fields)}
    line = json.dumps(rec, default=str, ensure_ascii=False)
    with _lock:
        with open(settings.log_dir / "trace.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")


@contextmanager
def timed(event: str, **fields: Any) -> Iterator[dict]:
    """Time a block and emit a trace event with latency and status."""
    t0 = time.perf_counter()
    info: dict[str, Any] = {}
    try:
        yield info
        trace(event, status="ok", latency_s=round(time.perf_counter() - t0, 3), **fields, **info)
    except Exception as e:  # noqa: BLE001
        trace(event, status="error", error=repr(e)[:500], latency_s=round(time.perf_counter() - t0, 3), **fields, **info)
        raise
