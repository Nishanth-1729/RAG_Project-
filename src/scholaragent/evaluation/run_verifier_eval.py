"""Verifier evaluation on held-out claims (test papers) - LLM judge vs fine-tuned NLI vs ensemble.
python -m scholaragent.evaluation.run_verifier_eval --nli models/scholar-verifier"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ..agents.verifier_agent import VerifierAgent
from ..config import settings
from ..llm.client import get_llm
from ..rag.citations import Evidence
from ..training.nli import LABELS, NLIVerifier
from .dataset import load_jsonl
from .harness import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli", default=str(settings.models_dir / "scholar-verifier"))
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    rows = load_jsonl(settings.data_dir / "training" / "claims_test.jsonl")
    if a.max:
        rows = rows[: a.max]
    llm = get_llm()
    ver = VerifierAgent(None, llm)       # judge() does not need the library

    def llm_label(r):
        ev = Evidence(1, "c", "p", "t", [], None, "", 1, 1, r["passage"])
        return ver.judge(r["claim"], [ev]).label
    llm_pred = [x if isinstance(x, str) else "UNSUPPORTED" for x in llm.map(llm_label, rows, workers=24)]
    gold = [r["label"] for r in rows]
    out = {"n": len(rows), "llm_judge": _m(gold, llm_pred)}
    try:
        nli = NLIVerifier(a.nli)
        P = nli.proba([r["claim"] for r in rows], [r["passage"] for r in rows])
        nli_pred = [LABELS[i] for i in P.argmax(1)]
        out["nli_finetuned"] = _m(gold, nli_pred)
        # ensemble: trust the LLM unless the NLI model is very confident about a different, stricter label
        ens = []
        for lp, r in zip(llm_pred, rows):
            veto = nli.strict_veto(r["claim"], [r["passage"]]) if lp == "SUPPORTED" else None
            ens.append(veto or lp)
        out["ensemble"] = _m(gold, ens)
    except Exception as e:  # noqa: BLE001
        out["nli_error"] = repr(e)[:200]
    # --- human-annotated SciFact validation (3-way: SUPPORTED / CONTRADICTED / NEI->UNSUPPORTED), PARTIAL merged into SUPPORTED
    from ..training.train_verifier import scifact_rows
    sf = scifact_rows("validation")
    sf_llm = [x if isinstance(x, str) else "UNSUPPORTED" for x in llm.map(llm_label, sf, workers=24)]
    sf_llm = ["SUPPORTED" if p == "PARTIALLY_SUPPORTED" else p for p in sf_llm]
    sf_gold = [r["label"] for r in sf]
    out["scifact_validation"] = {"n": len(sf), "llm_judge": _m3(sf_gold, sf_llm)}
    try:
        P2 = nli.proba([r["claim"] for r in sf], [r["passage"] for r in sf])
        sf_nli = [LABELS[i] for i in P2.argmax(1)]
        sf_nli = ["SUPPORTED" if p == "PARTIALLY_SUPPORTED" else p for p in sf_nli]
        out["scifact_validation"]["nli_finetuned"] = _m3(sf_gold, sf_nli)
    except Exception as e:  # noqa: BLE001
        out["scifact_validation"]["nli_error"] = repr(e)[:150]
    out["confusion_synthetic"] = {"labels": LABELS, "llm_judge": confusion_matrix(gold, llm_pred, labels=LABELS).tolist(),
                                  **({"nli_finetuned": confusion_matrix(gold, nli_pred, labels=LABELS).tolist()} if "nli_finetuned" in out else {})}
    save_json(out, "verifier_eval")
    print(json.dumps({k: v for k, v in out.items() if k not in ("confusion_synthetic",)}, indent=1, default=str)[:2500])


def _m3(gold, pred) -> dict:
    labs = ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]
    return {"accuracy": round(accuracy_score(gold, pred), 4), "macro_f1": round(f1_score(gold, pred, average="macro", labels=labs, zero_division=0), 4),
            "per_class": {l: round(float(f1_score([g == l for g in gold], [p == l for p in pred], zero_division=0)), 3) for l in labs}}


def _m(gold, pred) -> dict:
    bad = [g != "SUPPORTED" for g in gold]
    flagged = [p != "SUPPORTED" for p in pred]
    tp = sum(b and f for b, f in zip(bad, flagged))
    return {"accuracy": round(accuracy_score(gold, pred), 4), "macro_f1": round(f1_score(gold, pred, average="macro", labels=LABELS, zero_division=0), 4),
            "unsupported_claim_detection_rate": round(tp / max(1, sum(bad)), 4),
            "false_alarm_rate": round(sum((not b) and f for b, f in zip(bad, flagged)) / max(1, len(bad) - sum(bad)), 4),
            "per_class": {l: round(float(f1_score([g == l for g in gold], [p == l for p in pred], zero_division=0)), 3) for l in LABELS}}


if __name__ == "__main__":
    main()
