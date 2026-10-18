"""Turn retrieval hits into a numbered, token-budgeted evidence context for the LLM."""
from __future__ import annotations

from ..retrieval.hybrid import Hit
from ..utils.text import approx_tokens
from .citations import Evidence, evidence_from_hits


def build_context(hits: list[Hit], max_tokens: int = 5500, start_id: int = 1) -> tuple[str, list[Evidence]]:
    evidence = evidence_from_hits(hits, start_id)
    blocks, used, kept = [], 0, []
    for e in evidence:
        body = e.text
        head = f"[{e.id}] {e.short_ref} | {e.title} | {e.location}"
        cost = approx_tokens(head) + approx_tokens(body)
        if used + cost > max_tokens:
            room = max_tokens - used - approx_tokens(head)
            if room < 80:
                break
            body = body[: int(room * 3.2)] + " ..."          # ~3.2 chars per token
            e.text = body
            cost = max_tokens - used
        blocks.append(f"{head}\n\"\"\"\n{body}\n\"\"\"")
        kept.append(e)
        used += cost
    return "\n\n".join(blocks), kept
