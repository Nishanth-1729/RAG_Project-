"""Intent-classification accuracy of the Supervisor's planner on hand-written labeled queries."""
from __future__ import annotations

import json
from collections import Counter

from ..agents.supervisor import PLAN_SYS, Plan
from ..config import settings
from ..llm.client import get_llm
from .harness import save_json


def main():
    llm = get_llm()
    data = json.loads((settings.eval_dir / "planner_queries.json").read_text())

    def one(item):
        gold, q = item
        out = llm.generate_or_none(Plan, PLAN_SYS, f"REQUEST: {q}\nUSER SUPPLIED OWN PAPERS: False (0 papers)", max_tokens=300, tag="plan_eval")
        pred = ("incoherent" if out and not out.coherent_research_request else out.intent) if out else "none"
        return gold, pred, q
    res = [r for r in llm.map(one, data, workers=12) if isinstance(r, tuple)]
    acc = sum(g == p for g, p, _ in res) / len(res)
    conf = Counter((g, p) for g, p, _ in res)
    out = {"n": len(res), "intent_accuracy": round(acc, 3), "confusions": {f"{g}->{p}": c for (g, p), c in conf.items() if g != p},
           "errors": [{"gold": g, "pred": p, "query": q} for g, p, q in res if g != p]}
    save_json(out, "planner_eval")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
