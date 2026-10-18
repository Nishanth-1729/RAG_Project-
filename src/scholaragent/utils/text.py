"""Small text helpers shared across modules."""
from __future__ import annotations

import hashlib
import re
import unicodedata

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def approx_tokens(text: str) -> int:
    """Cheap, tokenizer-free token estimate (~ word-piece count)."""
    if not text:
        return 0
    return int(len(_TOKEN_RE.findall(text)) * 1.15) + 1


def normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def normalize_title(title: str) -> str:
    t = unicodedata.normalize("NFKD", title or "").lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return _WS_RE.sub(" ", t).strip()


def normalize_for_match(text: str) -> str:
    """Aggressive normalisation for quote / evidence matching (ignores case, punctuation, hyphenation)."""
    t = unicodedata.normalize("NFKD", text or "").lower()
    t = t.replace("-\n", "").replace("­", "")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return _WS_RE.sub(" ", t).strip()


def stable_hash(*parts: str, n: int = 16) -> str:
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return h[:n]


def slugify(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return s[:maxlen] or "x"


def truncate(text: str, n: int, suffix: str = "...") -> str:
    return text if len(text) <= n else text[: n - len(suffix)].rstrip() + suffix


def sentences(text: str) -> list[str]:
    """Light sentence splitter that is tolerant of abbreviations, 'et al.', 'Fig.', 'Eq.' etc."""
    text = normalize_ws(text)
    if not text:
        return []
    protected = re.sub(r"\b(et al|Fig|Figs|Eq|Eqs|Sec|Tab|Ref|e\.g|i\.e|vs|cf|approx|No|Alg)\.", lambda m: m.group(0).replace(".", "<DOT>"), text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[\(\\$])", protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]
