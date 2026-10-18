"""Cross-encoder reranker (query and passage are scored jointly)."""
from __future__ import annotations

import threading
from typing import Sequence

import numpy as np

from ..config import settings
from ..utils.gpu import on_gpu, resolve_device
from ..utils.logging import get_logger

log = get_logger("rerank")


class Reranker:
    def __init__(self, model_name: str | None = None, device: str | None = None, max_length: int = 512):
        self.model_name = model_name or settings.rerank_model
        self.device = resolve_device(device or settings.device)
        self.max_length = max_length
        self._model = None
        self._lock = threading.RLock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import torch
                    from sentence_transformers import CrossEncoder
                    log.info("loading reranker %s on %s", self.model_name, self.device)
                    kw = {"torch_dtype": torch.float16} if self.device.startswith("cuda") else {}
                    self._model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length,
                                               model_kwargs=kw, trust_remote_code=True)
        return self._model

    def score(self, query: str, passages: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        if not len(passages):
            return np.zeros(0, dtype=np.float32)
        model = self.model
        with self._lock:
            s = on_gpu(lambda: model.predict([(query, p) for p in passages], batch_size=batch_size or settings.rerank_batch,
                                                  show_progress_bar=False, convert_to_numpy=True))
        return np.asarray(s, dtype=np.float32)

    def unload(self) -> None:
        from ..utils.gpu import free_gpu
        self._model = None
        free_gpu()
