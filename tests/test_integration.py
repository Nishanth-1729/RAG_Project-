"""Integration tests: need the GPU (embedder) and, for the LLM test, the running vLLM server."""
import numpy as np
import pytest

from tests.test_core import make_pdf  # noqa: E402

torch = pytest.importorskip("torch")
_free_gb = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0.0
# the embedder + reranker need ~2.5 GB; skip (instead of failing) while the LLM server + API occupy the card
pytestmark = pytest.mark.skipif(_free_gb < 3.0, reason=f"needs a CUDA GPU with >= 3 GB free (have {_free_gb:.1f} GB; stop the stack with scripts/stop_stack.sh)")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    from scholaragent.evaluation.harness import Experiment
    from scholaragent.ingestion.pipeline import parse_pdf
    pdf = tmp_path_factory.mktemp("pdf") / "paper.pdf"
    make_pdf(pdf)
    doc = parse_pdf(pdf, use_cache=False, paper_id="itest_paper")
    return Experiment([doc])


def test_pdf_to_retrieval(pipeline):
    from scholaragent.evaluation.harness import Variant
    from scholaragent.ingestion.chunker import ChunkerConfig
    rt = pipeline.retriever(Variant("t", ChunkerConfig("section", 100, 10)))
    for mode in ("dense", "bm25", "hybrid"):
        hits = rt.search("speed-up over GMRES", k=3, mode=mode)
        assert hits, mode
        assert hits[0].chunk.paper_id == "itest_paper" and hits[0].chunk.page_start >= 1


def test_hybrid_rerank(pipeline):
    from scholaragent.evaluation.harness import Variant
    from scholaragent.ingestion.chunker import ChunkerConfig
    rt = pipeline.retriever(Variant("t", ChunkerConfig("section", 100, 10)), use_rerank=True)
    hits = rt.search("28% speed-up over GMRES", k=3, mode="hybrid_rerank")
    assert hits and "28%" in hits[0].chunk.text
    assert all("rerank" in h.scores for h in hits)


def test_qdrant_store_roundtrip():
    from scholaragent.ingestion.models import Chunk
    from scholaragent.retrieval.vectorstore import Filter, QdrantStore
    import requests
    try:
        requests.get("http://127.0.0.1:6333/healthz", timeout=2).raise_for_status()
    except Exception:  # noqa: BLE001
        pytest.skip("Qdrant not running")
    vs = QdrantStore(collection="itest_collection")
    vs.drop()
    cs = [Chunk("c1", "p1", "a", section="1 Intro", year=2020), Chunk("c2", "p2", "b", section="2 Method", year=2024)]
    vs.upsert(cs, np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))
    assert vs.search(np.array([1, 0, 0, 0], dtype=np.float32), 2)[0][0] == "c1"
    assert [c for c, _ in vs.search(np.array([1, 0, 0, 0], dtype=np.float32), 2, Filter(year_from=2023))] == ["c2"]
    vs.delete_paper("p1")
    assert vs.count() == 1
    vs.drop()
