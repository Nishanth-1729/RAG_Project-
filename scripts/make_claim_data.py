"""Synthetic claim-verification data with controlled labels (SUPPORTED / PARTIALLY_SUPPORTED / CONTRADICTED / UNSUPPORTED).

For a paragraph the LLM writes a faithful claim, a claim with an unsupported extra detail (partial) and a minimally
edited false claim (contradicted).  UNSUPPORTED = a faithful claim paired with a different, unrelated paragraph.
Train claims come from TRAIN papers, eval claims from the TEST papers (never mixed).
"""
import argparse
import json
import random

from pydantic import BaseModel, Field

from scholaragent.config import settings
from scholaragent.evaluation.dataset import evidence_units, save_jsonl
from scholaragent.evaluation.harness import load_docs
from scholaragent.evaluation.run_experiments import split_papers
from scholaragent.llm.client import get_llm


class Claims(BaseModel):
    supported: str = Field(description="one self-contained factual claim fully stated in the passage (paraphrased, names the method/quantity)")
    partial: str = Field(description="the same claim plus ONE extra specific detail that is NOT in the passage (a number, a broader scope like 'for all matrices', an additional property)")
    contradicted: str = Field(description="the supported claim minimally edited so that it is FALSE according to the passage (change a number, direction of effect, negate it, or swap an entity)")
    usable: bool = Field(description="false if the passage has no clear checkable fact")


SYS = """You create training data for a scientific claim verifier. Read the passage and write three claims about it:
1. supported: a faithful, self-contained claim stated in the passage.
2. partial: the same claim with one extra specific detail that the passage does NOT contain.
3. contradicted: the supported claim edited minimally so the passage shows it is false.
Claims must name the method/quantity explicitly (no 'the method', 'it'), be one sentence, and not copy the passage verbatim."""


def gen(units, llm):
    def one(u):
        r = llm.generate_or_none(Claims, SYS, f"PASSAGE:\n{u.text}", max_tokens=800, temperature=0.4, tag="gen_claims", retries=1)
        if not r or not r.usable or min(len(r.supported), len(r.partial), len(r.contradicted)) < 25:
            return None
        return (u, r)
    return [x for x in llm.map(one, units, workers=24) if isinstance(x, tuple)]


def build_rows(pairs, rng, all_units):
    rows = []
    by_paper = {}
    for u in all_units:
        by_paper.setdefault(u.paper_id, []).append(u)
    for u, r in pairs:
        rows += [{"claim": r.supported, "passage": u.text, "label": "SUPPORTED"},
                 {"claim": r.partial, "passage": u.text, "label": "PARTIALLY_SUPPORTED"},
                 {"claim": r.contradicted, "passage": u.text, "label": "CONTRADICTED"}]
        # unsupported: unrelated passage (50% same paper = hard, 50% other paper)
        pool = [x for x in by_paper[u.paper_id] if x.text != u.text] if rng.random() < 0.5 else [x for x in all_units if x.paper_id != u.paper_id]
        if pool:
            other = rng.choice(pool)
            rows.append({"claim": r.supported, "passage": other.text, "label": "UNSUPPORTED"})
    rng.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-units", type=int, default=3600)
    ap.add_argument("--test-units", type=int, default=450)
    a = ap.parse_args()
    rng = random.Random(5)
    llm = get_llm()
    assert llm.is_up()
    for split, n in (("train", a.train_units), ("test", a.test_units)):
        docs = load_docs(split_papers(split))
        units = [u for d in docs for u in evidence_units(d, 220, 1400) if u.canonical not in ("related_work",)]
        rng.shuffle(units)
        # spread over papers
        per, chosen, cnt = max(3, n // max(1, len(docs))) + 2, [], {}
        for u in units:
            if cnt.get(u.paper_id, 0) < per:
                chosen.append(u); cnt[u.paper_id] = cnt.get(u.paper_id, 0) + 1
            if len(chosen) >= n:
                break
        pairs = gen(chosen, llm)
        rows = build_rows(pairs, rng, units)
        out = settings.data_dir / "training" / f"claims_{split}.jsonl"
        save_jsonl(rows, out)
        from collections import Counter
        print(split, len(chosen), "units ->", len(rows), "rows", Counter(r["label"] for r in rows), flush=True)


if __name__ == "__main__":
    main()
