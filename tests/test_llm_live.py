"""Needs the running vLLM server (scripts/start_llm.sh); skipped otherwise."""
import pytest


@pytest.mark.llm
def test_llm_structured_output():
    from pydantic import BaseModel
    from scholaragent.llm.client import get_llm
    llm = get_llm()
    if not llm.is_up():
        pytest.skip("LLM server not running")

    class Out(BaseModel):
        method: str
        year: int
    r = llm.generate(Out, "Extract.", "The Fast Multipole Method was introduced in 1987 by Greengard and Rokhlin.")
    assert r.year == 1987
