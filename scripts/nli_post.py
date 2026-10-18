"""Post-hoc independent faithfulness: score every saved claim with the fine-tuned NLI model against the evidence the
system had (run when the GPU is free).   python scripts/nli_post.py [answer_outputs_test_EF] ..."""
import json
import sys

from scholaragent.config import settings
from scholaragent.training.nli import NLIVerifier

nli = NLIVerifier(str(settings.models_dir / "scholar-verifier"))
for name in sys.argv[1:] or ["answer_outputs_test_EF"]:
    path = settings.results_dir / f"{name}.json"
    data = json.loads(path.read_text())
    for system, outs in data.items():
        tot = ok = 0
        for o in outs:
            cm = o["claims"]
            ev = cm.get("evidence_texts") or []
            for c in cm.get("detail", []):
                tot += 1
                ok += nli.predict_label(c["claim"], ev) in ("SUPPORTED", "PARTIALLY_SUPPORTED")
            cm["nli_faithful"] = sum(nli.predict_label(c["claim"], ev) in ("SUPPORTED", "PARTIALLY_SUPPORTED") for c in cm.get("detail", [])) if ev else 0
        print(system, "NLI faithfulness", round(ok / max(1, tot), 4), f"({ok}/{tot})")
    path.write_text(json.dumps(data, ensure_ascii=False, default=str))
    sysp = settings.results_dir / path.name.replace("outputs", "systems")
    res = json.loads(sysp.read_text())
    for system, outs in data.items():
        n = sum(o["claims"].get("n_claims", 0) for o in outs)
        res[system]["faithfulness_nli"] = round(sum(o["claims"].get("nli_faithful", 0) for o in outs) / max(1, n), 4)
    sysp.write_text(json.dumps(res, indent=1, ensure_ascii=False))
