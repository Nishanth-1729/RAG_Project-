"""Agent-level metrics computed from a finished Supervisor state."""
from __future__ import annotations

EXPECTED = {
    # intent -> (required actions, allowed-but-optional actions)
    "survey": ({"search", "ingest", "analyze", "compare", "report"}, {"build_graph", "find_gaps"}),
    "gaps": ({"search", "ingest", "analyze", "find_gaps", "report"}, {"compare", "build_graph"}),
    "comparison": ({"search", "ingest", "analyze", "compare", "answer"}, set()),
    "qa": ({"search", "ingest", "answer"}, set()),
}


def expected_for(mode: str, intent: str) -> tuple[set[str], set[str]]:
    req, opt = EXPECTED.get(intent, EXPECTED["survey"])
    req, opt = set(req), set(opt)
    if mode == "my_papers":                       # user supplied papers: no search / no ingest needed
        req -= {"search", "ingest"}
        opt -= {"search", "ingest"}
    return req, opt


def score_run(state: dict, mode: str) -> dict:
    acts = [d["action"] for d in state.get("decisions", []) if d["action"] != "finish"]
    intent = state.get("intent", "survey")
    req, opt = expected_for(mode, intent)
    allowed = req | opt | {"search"}                 # a second search round is legitimate when too few papers were found
    executed = set(acts)
    unnecessary = [a for a in acts if a not in allowed]
    repeats = len(acts) - len(executed) - max(0, acts.count("search") - 1) - max(0, acts.count("ingest") - 1)
    deliverable = bool(state.get("report")) if intent in ("survey", "gaps") else bool(state.get("answer"))
    return {"intent": intent, "actions": acts, "required_covered": round(len(req & executed) / max(1, len(req)), 3),
            "tool_selection_accuracy": round(sum(a in allowed for a in acts) / max(1, len(acts)), 3),
            "unnecessary_calls": len(unnecessary) + max(0, repeats), "steps": len(acts), "deliverable": deliverable,
            "errors": len(state.get("errors", []))}
