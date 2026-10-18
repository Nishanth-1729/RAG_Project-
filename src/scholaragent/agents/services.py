"""Service container: one place that wires library, models, LLM and all agents."""
from __future__ import annotations

import threading

from ..llm.client import LLM, get_llm
from ..rag.pipeline import RagPipeline
from ..retrieval.embedder import Embedder
from ..retrieval.library import Library
from ..retrieval.reranker import Reranker
from .comparison_agent import ComparisonAgent
from .gap_agent import GapAgent
from .graph_agent import GraphAgent
from .paper_agent import PaperAnalyst
from .report_agent import ReportAgent
from .search_agent import SearchAgent
from .verifier_agent import VerifierAgent


def load_nli():
    """Fine-tuned claim verifier, if trained (models/scholar-verifier or $VERIFIER_NLI)."""
    import os
    from pathlib import Path
    from ..config import settings
    path = Path(os.getenv("VERIFIER_NLI", settings.models_dir / "scholar-verifier"))
    if os.getenv("VERIFIER_NLI_MODE", "primary") == "off" or not (path / "config.json").exists():
        return None
    from ..training.nli import NLIVerifier
    return NLIVerifier(str(path), device=settings.device)


class Services:
    def __init__(self, library: Library | None = None, llm: LLM | None = None):
        self.llm = llm or get_llm()
        if library is None:
            library = Library(embedder=Embedder(), reranker=Reranker()).load()
        self.lib = library
        self.rag = RagPipeline(self.lib, self.llm)
        self.search_agent = SearchAgent(self.lib.embedder, self.llm)
        self.analyst = PaperAnalyst(self.lib, self.llm)
        self.comparison = ComparisonAgent(self.llm)
        self.gap = GapAgent(self.lib.embedder, self.llm)
        self.verifier = VerifierAgent(self.lib, self.llm, nli=load_nli())
        self.report = ReportAgent(self.lib, self.verifier, self.llm)
        self.graph = GraphAgent(self.lib.store)


_svc: Services | None = None
_lock = threading.Lock()


def get_services() -> Services:
    global _svc
    with _lock:
        if _svc is None:
            _svc = Services()
        return _svc
