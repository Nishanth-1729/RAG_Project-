"""Supervisor action rules + agent metrics (pure logic, no models)."""
from scholaragent.agents.supervisor import allowed_actions, default_policy
from scholaragent.evaluation.agent_metrics import score_run


def base(**kw):
    s = {"mode": "deep", "intent": "survey", "plan": {"needs_literature_search": True}, "params": {"max_papers": 10},
         "done": [], "paper_ids": [], "pending_records": [], "search_rounds": 0, "decisions": []}
    s.update(kw)
    return s


def test_must_search_first():
    assert allowed_actions(base()) == ["search"]


def test_pending_papers_must_be_ingested_before_anything_else():
    assert allowed_actions(base(done=["search"], pending_records=[{"x": 1}], search_rounds=1)) == ["ingest"]


def test_search_again_when_too_few_papers():
    s = base(done=["search", "ingest"], paper_ids=["a", "b"], search_rounds=1)
    assert "search" in allowed_actions(s) and "analyze" in allowed_actions(s)
    s["search_rounds"] = 3
    assert "search" not in allowed_actions(s)


def test_enough_papers_goes_to_analysis_then_report():
    s = base(done=["search", "ingest"], paper_ids=list("abcdefgh"))
    assert allowed_actions(s) == ["analyze"]
    s["done"] += ["analyze", "compare", "build_graph", "find_gaps"]
    assert allowed_actions(s) == ["report"]
    s["done"].append("report")
    assert allowed_actions(s) == ["finish"]


def test_my_papers_qa_skips_search_and_analysis():
    s = base(mode="my_papers", intent="qa", plan={"needs_literature_search": False}, paper_ids=["a"])
    assert allowed_actions(s) == ["answer"]
    assert default_policy(s, allowed_actions(s)) == "answer"


def test_nothing_indexed_finishes_gracefully():
    s = base(done=["search", "ingest"], paper_ids=[], search_rounds=1)
    assert allowed_actions(s) == ["search"]          # nothing found yet: try another round
    s["search_rounds"] = 3
    assert allowed_actions(s) == ["finish"]          # budget exhausted: finish with an honest empty result


def test_agent_metrics():
    st = {"intent": "survey", "decisions": [{"action": a} for a in ["search", "ingest", "analyze", "compare", "report", "finish"]],
          "report": {"x": 1}, "errors": []}
    r = score_run(st, "deep")
    assert r["required_covered"] == 1.0 and r["unnecessary_calls"] == 0 and r["deliverable"]
    st2 = {"intent": "qa", "decisions": [{"action": a} for a in ["search", "answer", "finish"]], "answer": {}, "errors": []}
    assert score_run(st2, "my_papers")["unnecessary_calls"] == 0     # a search round is always tolerated
