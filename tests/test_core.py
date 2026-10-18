"""Unit tests for the deterministic parts (no GPU / LLM / network needed)."""
import numpy as np
import pymupdf
import pytest

from scholaragent.evaluation.retrieval_metrics import aggregate, contains_span, evaluate_ranking, ndcg_at_k
from scholaragent.ingestion.chunker import Chunker, ChunkerConfig
from scholaragent.ingestion.metadata import find_ids
from scholaragent.ingestion.models import Chunk, Paragraph, ParsedDocument, Section
from scholaragent.ingestion.pdf_parser import classify_equation
from scholaragent.ingestion.pipeline import parse_pdf
from scholaragent.ingestion.section_parser import canonicalize
from scholaragent.rag.citations import extract_citation_ids, quote_support, strip_invalid_citations
from scholaragent.retrieval.fusion import max_per_group, rrf
from scholaragent.retrieval.sparse import BM25, tokenize
from scholaragent.search.dedup import deduplicate
from scholaragent.search.models import PaperRecord

LOREM = ("The far-field interactions are approximated using low-rank representations of the kernel matrix. "
         "This reduces the cost of the matrix-vector product from quadratic to near-linear in the number of points. ") * 6


def make_pdf(path):
    doc = pymupdf.open()
    def page(blocks):
        p = doc.new_page()
        y = 70
        for text, size, bold in blocks:
            font = "hebo" if bold else "helv"
            if len(text) < 80:                      # headings / short lines
                p.insert_text((50, y), text, fontsize=size, fontname=font)
                y += size * 1.8
            else:
                h = size * 1.5 * (2 + len(text) // 95)
                rc = p.insert_textbox(pymupdf.Rect(50, y - size, 545, y - size + h), text, fontsize=size, fontname=font)
                assert rc >= 0, "test pdf text overflow"
                y += h + 6
    page([("Fast Hierarchical Solvers for Kernel Matrices", 18, True), ("Alice Smith, Bob Jones", 11, False),
          ("Abstract", 12, True), ("We present a fast solver based on low-rank approximations.", 10, False),
          ("1 Introduction", 12, True), (LOREM, 10, False)])
    page([("2 Method", 12, True), (LOREM, 10, False), ("2.1 Hierarchical Approximation", 11, True), (LOREM, 10, False)])
    page([("3 Experiments", 12, True), (LOREM + " We obtain a 28% speed-up over GMRES.", 10, False),
          ("4 Conclusion", 12, True), ("We showed the method works. Future work includes adaptive rank selection.", 10, False),
          ("References", 12, True), ("[1] A. Author. Some paper. 2020.", 9, False)])
    doc.save(path)


def test_canonicalize():
    assert canonicalize("3.2 Hierarchical Approximation") == "other"
    assert canonicalize("1 Introduction") == "introduction"
    assert canonicalize("4 Experimental Setup") == "experiments"
    assert canonicalize("6 Limitations and Future Work") == "limitations"
    assert canonicalize("REFERENCES") == "references"
    assert canonicalize("2 Related Work") == "related_work"


def test_equation_detection():
    assert classify_equation("A x = b  ⇒  x = A⁻¹ b  (14)")
    assert not classify_equation("We evaluate the proposed approach on three standard benchmark problems and report timings.")


def test_ids():
    doi, arx = find_ids("see https://doi.org/10.1145/3394486.3403136. and arXiv:2406.00809v2")
    assert doi == "10.1145/3394486.3403136" and arx == "2406.00809"


def test_parse_and_chunk_roundtrip(tmp_path):
    pdf = tmp_path / "paper.pdf"
    make_pdf(pdf)
    from scholaragent.config import settings
    settings.parsed_dir.mkdir(parents=True, exist_ok=True)
    doc = parse_pdf(pdf, use_cache=False, paper_id="test_paper")
    assert "Fast Hierarchical" in doc.title
    canon = [s.canonical for s in doc.sections]
    assert "introduction" in canon and "experiments" in canon and "references" in canon
    chunks = Chunker(ChunkerConfig("section", 120, 20)).chunk(doc)
    assert chunks and all(c.section and c.page_start >= 1 for c in chunks)
    assert not any(c.section_canonical == "references" for c in chunks)
    assert any(c.page_start == 3 for c in chunks)
    fixed = Chunker(ChunkerConfig("fixed", 100, 10)).chunk(doc)
    assert len(fixed) > 3
    (settings.parsed_dir / "test_paper.json").unlink(missing_ok=True)


def test_section_chunks_never_cross_sections():
    s1 = Section("1 Introduction", canonical="introduction", paragraphs=[Paragraph(LOREM, 1, 1)])
    s2 = Section("2 Method", canonical="method", paragraphs=[Paragraph(LOREM, 2, 2)])
    d = ParsedDocument("p", "x", 2, title="T", sections=[s1, s2])
    for c in Chunker(ChunkerConfig("section", 150, 20)).chunk(d):
        assert c.section in ("1 Introduction", "2 Method")
        assert (c.page_start == 1) == (c.section == "1 Introduction")


def test_bm25_exact_terms():
    docs = {"a": "GMRES converges slowly for ill-conditioned systems without preconditioning",
            "b": "The fast multipole method has O(N) complexity for N-body problems",
            "c": "We train a neural network on random matrices"}
    bm = BM25().fit(list(docs), list(docs), list(docs.values()))
    assert bm.search("GMRES", 3)[0][0] == "a"
    assert bm.search("O(N) complexity", 3)[0][0] == "b"
    assert "low" in tokenize("low-rank") and "low-rank" in tokenize("low-rank")


def test_rrf():
    out = rrf([["A", "C", "D", "B"], ["B", "A", "E", "C"]])
    assert out[0][0] == "A"
    assert {d for d, _ in out} == set("ABCDE")


def test_max_per_group():
    assert max_per_group(list("aabab"), lambda x: x, 2) == list("aabb")


def test_dedup():
    a = PaperRecord("Fast Multipole Methods for N-body problems", year=2020, doi="10.1/abc", authors=["A. Smith"], sources=["crossref"])
    b = PaperRecord("Fast multipole methods for N-body problems.", year=2020, arxiv_id="2001.00001", authors=["A. Smith"], abstract="abs", sources=["arxiv"])
    c = PaperRecord("Something else entirely different", year=2021)
    out = deduplicate([a, b, c])
    assert len(out) == 2 and set(out[0].sources) == {"crossref", "arxiv"} and out[0].abstract == "abs"


def test_citations():
    assert extract_citation_ids("A [1][3]. B [2, 4]. C [5-6]") == [1, 3, 2, 4, 5, 6]
    txt, removed = strip_invalid_citations("x [1] y [9] z [2, 8]", {1, 2})
    assert "[9]" not in txt and "8" not in txt and removed == [9, 8]
    assert quote_support("reduces the cost from quadratic", "This reduces the cost from quadratic to near-linear.") == 100
    assert quote_support("completely unrelated statement about cats", "reduces the cost from quadratic") < 60


def _chunk(cid, pid, text, sec="3 Method"):
    return Chunk(cid, pid, text, section=sec, n_tokens=len(text.split()))


def test_retrieval_metrics():
    q = {"qid": "q", "type": "factual", "evidences": [{"paper_id": "p1", "span": "the preconditioner reduces iteration counts by forty percent on all test matrices",
                                                         "paragraph": "x", "section": "4 Results"}]}
    good = _chunk("c1", "p1", "In our experiments the preconditioner reduces iteration counts by forty percent on all test matrices of the suite.")
    bad = _chunk("c2", "p2", "unrelated text")
    r = evaluate_ranking(q, [bad, good], max_k=5)
    assert r.first_rank == 2
    agg = aggregate([r])
    assert agg["recall@1"] == 0 and agg["recall@3"] == 1 and agg["mrr"] == 0.5
    assert contains_span(good.text, q["evidences"][0]["span"])
    assert ndcg_at_k([0, 3], 2) < ndcg_at_k([3, 0], 2) == 1.0
    assert ndcg_at_k([3, 2], 2, ideal_grades=[3, 2, 1]) == 1.0 and ndcg_at_k([3, 2, 3], 3, ideal_grades=[3, 3, 2]) <= 1.0


def test_memory_store_filter():
    torch = pytest.importorskip("torch")
    from scholaragent.retrieval.vectorstore import Filter, MemoryStore
    vs = MemoryStore("cpu")
    cs = [_chunk("a", "p1", "x"), _chunk("b", "p2", "y")]
    vs.upsert(cs, np.array([[1, 0], [0, 1]], dtype=np.float32))
    assert vs.search(np.array([1, 0], dtype=np.float32), 2)[0][0] == "a"
    assert [c for c, _ in vs.search(np.array([1, 0], dtype=np.float32), 2, Filter(paper_ids={"p2"}))] == ["b"]
