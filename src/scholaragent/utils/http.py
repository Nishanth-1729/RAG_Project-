"""HTTP client with retry/backoff, per-host rate limiting and an on-disk JSON cache (saves API calls)."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import requests

from ..config import settings
from .logging import get_logger, trace

log = get_logger("http")
_UA = "ScholarAgent/1.0 (research tool)"
_host_lock = threading.Lock()
_last_call: dict[str, float] = {}
MIN_INTERVAL = {"export.arxiv.org": 3.1, "arxiv.org": 1.0, "api.semanticscholar.org": 1.1,
                "api.crossref.org": 0.3, "api.openalex.org": 0.2}


def _wait(host: str) -> None:
    gap = MIN_INTERVAL.get(host, 0.0)
    if gap <= 0:
        return
    with _host_lock:
        now = time.monotonic()
        wait = _last_call.get(host, 0.0) + gap - now
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.monotonic()


class HttpError(RuntimeError):
    pass


class _Breaker:
    """Per-host circuit breaker: after `threshold` consecutive failed requests the host is skipped for `cooldown` s."""
    def __init__(self, threshold: int = 3, cooldown: float = 300.0):
        self.threshold, self.cooldown = threshold, cooldown
        self.fails: dict[str, int] = {}
        self.open_until: dict[str, float] = {}

    def check(self, host: str) -> None:
        if time.time() < self.open_until.get(host, 0):
            raise HttpError(f"{host} temporarily disabled after repeated failures")

    def ok(self, host: str) -> None:
        self.fails[host] = 0

    def fail(self, host: str) -> None:
        self.fails[host] = self.fails.get(host, 0) + 1
        if self.fails[host] >= self.threshold:
            self.open_until[host] = time.time() + self.cooldown
            self.fails[host] = 0


breaker = _Breaker()


class CachedHttp:
    def __init__(self, cache_ttl_s: float = 7 * 86400, timeout: float = 30.0):
        self.dir = settings.cache_dir / "http"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl = cache_ttl_s
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": _UA})

    def _key(self, url: str, params: dict | None, headers: dict | None) -> Path:
        h = hashlib.sha256(json.dumps([url, params, sorted((headers or {}).keys())], sort_keys=True, default=str).encode()).hexdigest()
        return self.dir / f"{h}.json"

    def get(self, url: str, params: dict | None = None, headers: dict | None = None, *, as_text: bool = False,
            cache: bool = True, retries: int = 4) -> Any:
        key = self._key(url, params, headers)
        if cache and key.exists() and time.time() - key.stat().st_mtime < self.ttl:
            return json.loads(key.read_text())["data"]
        host = requests.utils.urlparse(url).netloc
        breaker.check(host)
        last_err: Exception | None = None
        for attempt in range(retries):
            _wait(host)
            t0 = time.perf_counter()
            try:
                r = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                trace("http", host=host, status=r.status_code, latency_s=round(time.perf_counter() - t0, 2))
                if r.status_code == 429 or r.status_code >= 500:
                    retry_after = float(r.headers.get("Retry-After", 0) or 0)
                    time.sleep(min(60.0, max(retry_after, 3.0 * (2 ** attempt))))
                    last_err = HttpError(f"{r.status_code} from {host}")
                    continue
                if r.status_code >= 400:
                    raise HttpError(f"{r.status_code} from {url}")
                data = r.text if as_text else r.json()
                if cache:
                    key.write_text(json.dumps({"data": data}))
                breaker.ok(host)
                return data
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        breaker.fail(host)
        raise HttpError(f"GET {url} failed after {retries} attempts: {last_err!r}")

    def download(self, url: str, dest: Path, max_mb: int = 80) -> Path:
        host = requests.utils.urlparse(url).netloc
        _wait(host)
        with self.session.get(url, stream=True, timeout=60, allow_redirects=True) as r:
            if r.status_code >= 400:
                raise HttpError(f"{r.status_code} downloading {url}")
            tmp = dest.with_suffix(".part")
            size = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    size += len(chunk)
                    if size > max_mb * 1e6:
                        tmp.unlink(missing_ok=True)
                        raise HttpError("PDF too large")
                    f.write(chunk)
            with open(tmp, "rb") as f:
                if f.read(5) != b"%PDF-":
                    tmp.unlink(missing_ok=True)
                    raise HttpError(f"{url} did not return a PDF")
            tmp.replace(dest)
        return dest


http = CachedHttp()
