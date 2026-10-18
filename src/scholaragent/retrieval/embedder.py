"""Dense embedding model wrapper (SentenceTransformers on the GPU)."""
from __future__ import annotations

import threading
from typing import Sequence

import numpy as np

from ..config import settings
from ..utils.gpu import on_gpu, resolve_device
from ..utils.logging import get_logger

log = get_logger("embed")

_QUERY_PREFIX = {
    "bge-": "Represent this sentence for searching relevant passages: ",
    "e5": "query: ",
}
_DOC_PREFIX = {"e5": "passage: "}


def _prefix(model_name: str, table: dict[str, str], is_query_model_en: bool = True) -> str:
    n = model_name.lower()
    if "bge-m3" in n:
        return ""
    for k, v in table.items():
        if k in n:
            return v
    return ""


class Embedder:
    def __init__(self, model_name: str | None = None, device: str | None = None, max_seq_length: int = 512,
                 query_prefix: str | None = None, doc_prefix: str | None = None):
        self.model_name = model_name or settings.embed_model
        self.device = resolve_device(device or settings.device)
        self.max_seq_length = max_seq_length
        base = self._base_name(self.model_name)          # fine-tuned checkpoints inherit the prompt format of their base model
        self.q_prefix = _prefix(base, _QUERY_PREFIX) if query_prefix is None else query_prefix
        self.d_prefix = _prefix(base, _DOC_PREFIX) if doc_prefix is None else doc_prefix
        self._model = None
        self._lock = threading.RLock()

    @staticmethod
    def _base_name(name: str) -> str:
        import json
        from pathlib import Path
        info = Path(name) / "train_info.json"
        if info.exists():
            try:
                return json.loads(info.read_text()).get("base", name)
            except Exception:  # noqa: BLE001
                pass
        return name

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import torch
                    from sentence_transformers import SentenceTransformer
                    log.info("loading embedding model %s on %s", self.model_name, self.device)
                    kw = {"torch_dtype": torch.float16} if self.device.startswith("cuda") else {}
                    m = SentenceTransformer(self.model_name, device=self.device, model_kwargs=kw, trust_remote_code=True)
                    m.max_seq_length = self.max_seq_length
                    self._model = m
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def _encode(self, texts: Sequence[str], batch_size: int | None = None, show: bool = False) -> np.ndarray:
        model = self.model            # resolve (lazy-load) on the caller thread, then run the forward pass on the GPU thread

        def run():
            return model.encode(list(texts), batch_size=batch_size or settings.embed_batch, convert_to_numpy=True,
                                     normalize_embeddings=True, show_progress_bar=show)
        with self._lock:     # one forward pass at a time, always from the GPU thread
            v = on_gpu(run)
        return v.astype(np.float32)

    def encode_docs(self, texts: Sequence[str], batch_size: int | None = None, show: bool = False) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode([self.d_prefix + t for t in texts], batch_size, show)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.q_prefix + t for t in texts])

    def __call__(self, texts: Sequence[str]) -> np.ndarray:        # symmetric use (chunking, ranking)
        return self._encode(list(texts))

    def unload(self) -> None:
        from ..utils.gpu import free_gpu
        self._model = None
        free_gpu()
