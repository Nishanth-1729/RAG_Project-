"""Fine-tuned claim-verification cross-encoder (independent of the LLM) - inference wrapper."""
from __future__ import annotations

import threading
from typing import Sequence

import numpy as np

LABELS = ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]


class NLIVerifier:
    def __init__(self, path: str, device: str = "cuda", max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.max_length = max_length
        self.lock = threading.Lock()

    def proba(self, claims: Sequence[str], passages: Sequence[str], batch_size: int = 8) -> np.ndarray:
        from ..utils.gpu import on_gpu

        def run():
            out = []
            with self.torch.no_grad():
                for i in range(0, len(claims), batch_size):
                    enc = self.tok(list(passages[i:i + batch_size]), list(claims[i:i + batch_size]), truncation="only_first",
                                   max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
                    out.append(self.torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy())
            return np.concatenate(out) if out else np.zeros((0, len(LABELS)))
        with self.lock:
            return on_gpu(run)

    def predict_label(self, claim: str, passages: Sequence[str]) -> str:
        """Aggregate over passages: best supporting passage wins; contradiction needs no supporting passage."""
        if not passages:
            return "UNSUPPORTED"
        P = self.proba([claim] * len(passages), list(passages))
        if P[:, 0].max() > 0.5:
            return "SUPPORTED"
        if P[:, 1].max() > 0.5:
            return "PARTIALLY_SUPPORTED"
        if P[:, 3].max() > 0.5:
            return "CONTRADICTED"
        return "UNSUPPORTED"

    def strict_veto(self, claim: str, passages: Sequence[str], thr: float = 0.9) -> str | None:
        """Label that overrides an LLM 'SUPPORTED': only when NO passage supports the claim and the NLI model is
        highly confident that it is contradicted / unsupported. Returns None when the LLM verdict should stand."""
        if not passages:
            return None
        P = self.proba([claim] * len(passages), list(passages))
        mx = P.max(0)                               # best score per label over passages
        if mx[0] < 0.1 and mx[1] < 0.1:
            if mx[3] > thr:
                return "CONTRADICTED"
            if mx[2] > thr:
                return "UNSUPPORTED"
        return None
