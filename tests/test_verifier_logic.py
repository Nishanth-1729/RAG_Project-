"""Verifier decision logic with a stub NLI model (no GPU / LLM needed)."""
import numpy as np

from scholaragent.agents.report_agent import _clean_section
from scholaragent.agents.schemas import ClaimVerdict
from scholaragent.agents.verifier_agent import VerifierAgent, _summary, VerifiedClaim
from scholaragent.evaluation.dataset import has_meta_reference
from scholaragent.rag.citations import Evidence
from scholaragent.training.nli import NLIVerifier


class StubNLI:
    """Returns fixed class probabilities [SUPPORTED, PARTIAL, UNSUPPORTED, CONTRADICTED] per passage."""
    def __init__(self, probs):
        self.p = np.array(probs, dtype=float)

    def proba(self, claims, passages, batch_size=8):
        return self.p[: len(passages)]

    strict_veto = NLIVerifier.strict_veto


class StubLLM:
    def generate_or_none(self, schema, system, user, **kw):
        return ClaimVerdict(label="SUPPORTED", best_passage=1, rationale="llm says supported")


def ev(i):
    return Evidence(i, f"c{i}", "p", "Title", ["A B"], 2020, "3 Method", 1, 1, f"passage {i}")


def agent(probs, mode="primary"):
    return VerifierAgent(None, StubLLM(), nli=StubNLI(probs), nli_mode=mode)


def test_nli_primary_confident_supported():
    v = agent([[0.1, 0.0, 0.0, 0.0], [0.95, 0.02, 0.02, 0.01]]).judge("claim", [ev(1), ev(2)])
    assert v.label == "SUPPORTED" and v.best_passage == 2 and "entailment model" in v.rationale


def test_nli_primary_contradicted_and_unsupported():
    assert agent([[0.02, 0.02, 0.06, 0.9]]).judge("c", [ev(1)]).label == "CONTRADICTED"
    assert agent([[0.01, 0.01, 0.97, 0.01], [0.0, 0.02, 0.97, 0.01]]).judge("c", [ev(1), ev(2)]).label == "UNSUPPORTED"


def test_nli_uncertain_falls_back_to_llm():
    v = agent([[0.4, 0.2, 0.3, 0.1]]).judge("c", [ev(1)])
    assert v.rationale == "llm says supported"


def test_veto_only_downgrades_supported():
    sure_unsupported = [[0.01, 0.01, 0.97, 0.01]]
    assert agent(sure_unsupported, "veto").judge("c", [ev(1)]).label == "UNSUPPORTED"
    assert agent([[0.9, 0.05, 0.03, 0.02]], "veto").judge("c", [ev(1)]).label == "SUPPORTED"


def test_summary_counts():
    a, b = VerifiedClaim("x", [], "SUPPORTED", "SUPPORTED"), VerifiedClaim("y", [], "UNSUPPORTED", "UNSUPPORTED", action="removed")
    s = _summary([a, b])
    assert s["claims_checked"] == 2 and s["removed"] == 1 and s["initial"]["SUPPORTED"] == 1


def test_clean_section():
    t = "### Limitations Overview. - **A** - **B** text [1]. **Heading:**\n- point [2]"
    out = _clean_section(t, "Limitations")
    assert not out.startswith("#") and "\n- **A**" in out and "**Heading:**\n\n- point" in out


def test_meta_reference_filter():
    assert has_meta_reference("What does the notation mean in this passage?")
    assert has_meta_reference("How do the approaches in Passages A and B differ?")
    assert has_meta_reference("What does the paper propose?")
    assert not has_meta_reference("How is the prolongation operator P built in the NSPOD preconditioner?")
