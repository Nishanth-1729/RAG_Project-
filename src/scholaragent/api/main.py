"""ScholarAgent REST API."""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Literal

import markdown as md_lib
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..agents.graph_agent import to_cypher
from ..agents.services import get_services
from ..config import settings
from ..llm.client import get_llm, meter
from ..search.service import search_literature
from ..utils.gpu import gpu_info
from . import jobs

app = FastAPI(title="ScholarAgent API", version=__version__,
              description="Evidence-grounded agentic RAG for scientific literature discovery, analysis and research-gap exploration")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


app.mount("/results-files", StaticFiles(directory=str(settings.results_dir)), name="results-files")


def svc():
    return get_services()


# ------------------------------------------------------------------- schemas
class AskRequest(BaseModel):
    question: str
    collection_id: str | None = None
    paper_ids: list[str] | None = None
    k: int = 8
    mode: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    rewrite: bool = False
    decompose: bool = True       # split comparison / multi-part questions into per-entity retrieval queries
    verify: bool = False
    max_per_paper: int | None = None


class SearchRequest(BaseModel):
    query: str
    year_from: int | None = None
    year_to: int | None = None
    sources: list[str] = ["arxiv", "semantic_scholar", "crossref"]
    limit: int = 20


class ResearchRequest(BaseModel):
    question: str
    mode: Literal["deep", "my_papers"] = "deep"
    max_papers: int = Field(15, ge=1, le=60)
    year_from: int | None = None
    year_to: int | None = None
    sources: list[str] = ["arxiv", "semantic_scholar", "crossref"]
    collection_id: str | None = None
    paper_ids: list[str] | None = None
    verify: bool = True
    intent: Literal["survey", "comparison", "qa", "gaps"] | None = None


class CollectionRequest(BaseModel):
    name: str


# --------------------------------------------------------------------- system
@app.get("/health")
def health():
    llm_up = get_llm().is_up()
    out = {"status": "ok" if llm_up else "degraded", "version": __version__, "llm": {"up": llm_up, "model": settings.llm_model, "base_url": settings.llm_base_url},
           "gpu": gpu_info(), "embed_model": settings.embed_model, "rerank_model": settings.rerank_model}
    try:
        out["library"] = svc().lib.stats()
        out["qdrant"] = {"up": True}
    except Exception as e:  # noqa: BLE001
        out["qdrant"] = {"up": False, "error": repr(e)[:200]}
        out["status"] = "degraded"
    out["usage"] = meter.snapshot()
    return out


# --------------------------------------------------------------------- papers
@app.post("/papers/upload")
async def upload_papers(files: list[UploadFile] = File(...), collection_id: str | None = Form(None)):
    s = svc()
    cid = s.lib.store.create_collection("My Papers", "user", collection_id=collection_id or "my_papers")
    results = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            results.append({"ok": False, "title": f.filename, "error": "not a PDF"})
            continue
        dest = settings.upload_dir / f"{uuid.uuid4().hex[:8]}_{Path(f.filename).name}"
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        results.append(s.lib.ingest_pdf(dest, collection_id=cid))
    return {"collection_id": cid, "results": results}


@app.get("/papers")
def list_papers(collection_id: str | None = None):
    store = svc().lib.store
    ids = store.collection_papers(collection_id) if collection_id else None
    return store.list_papers(ids)


@app.delete("/papers/{paper_id}")
def delete_paper(paper_id: str):
    svc().lib.remove_paper(paper_id)
    return {"deleted": paper_id}


@app.get("/papers/{paper_id}/chunks")
def paper_chunks(paper_id: str):
    return [c.to_dict() for c in svc().lib.chunks_for(paper_id)]


@app.get("/papers/{paper_id}/analysis")
def paper_analysis(paper_id: str, refresh: bool = False):
    a = svc().analyst.analyze(paper_id, force=refresh)
    if not a:
        raise HTTPException(404, "analysis unavailable")
    return a


@app.get("/evidence/{chunk_id}")
def evidence(chunk_id: str):
    c = svc().lib.store.get_chunk_by_ids([chunk_id]).get(chunk_id)
    if not c:
        raise HTTPException(404, "chunk not found")
    return c.to_dict()


@app.get("/collections")
def collections():
    return svc().lib.store.list_collections()


@app.post("/collections")
def create_collection(req: CollectionRequest):
    return {"collection_id": svc().lib.store.create_collection(req.name)}


# ---------------------------------------------------------------------- QA
@app.post("/ask")
def ask(req: AskRequest):
    s = svc()
    pids = req.paper_ids
    if pids is None and req.collection_id:
        pids = s.lib.store.collection_papers(req.collection_id)
    ans, timing = s.rag.ask(req.question, pids, k=req.k, mode=req.mode, rewrite=req.rewrite, max_per_paper=req.max_per_paper, decompose=req.decompose)
    out = ans.to_dict() | {"timing": timing}
    if req.verify:
        rep = s.verifier.verify_answer(ans, pids)
        out["verification"] = rep
        out["final_answer"] = rep["final_answer"]
    else:
        out["final_answer"] = ans.text
    return out


@app.post("/search/papers")
def search_papers(req: SearchRequest):
    recs, stats = search_literature([req.query], sources=req.sources, per_source=req.limit, year_from=req.year_from,
                                    year_to=req.year_to, rank_query=req.query, embed_fn=svc().lib.embedder)
    return {"stats": stats, "papers": [r.to_dict() for r in recs[: req.limit]]}


# -------------------------------------------------------------------- research
@app.post("/research")
def start_research(req: ResearchRequest):
    store = svc().lib.store
    params = req.model_dump(exclude={"question", "mode"})
    if req.mode == "my_papers" and not (req.paper_ids or req.collection_id):
        params["collection_id"] = "my_papers"
    rid = store.create_run(req.question, req.mode, params)
    jobs.submit(rid)
    return {"run_id": rid}


@app.get("/runs")
def runs():
    return svc().lib.store.list_runs()


@app.get("/runs/{run_id}")
def run_status(run_id: str, with_result: bool = False):
    r = svc().lib.store.get_run(run_id)
    if not r:
        raise HTTPException(404, "run not found")
    if not with_result:
        r = {**r, "result": None, "has_result": r["result"] is not None}
    return r


def _result(run_id: str) -> dict:
    r = svc().lib.store.get_run(run_id)
    if not r or not r.get("result"):
        raise HTTPException(404, "result not ready")
    return r["result"]


@app.get("/runs/{run_id}/report.{fmt}")
def report(run_id: str, fmt: Literal["md", "html", "bib", "json", "tex"]):
    res = _result(run_id)
    rep = res.get("report")
    if not rep:
        raise HTTPException(404, "this run produced no report (it was a question-answering run)")
    if fmt == "md":
        return PlainTextResponse(rep["markdown"], media_type="text/markdown")
    if fmt == "bib":
        return PlainTextResponse(rep["bibtex"], media_type="text/plain")
    if fmt == "json":
        return JSONResponse(rep)
    if fmt == "tex":
        return PlainTextResponse(to_latex(rep["markdown"]), media_type="text/plain")
    body = md_lib.markdown(rep["markdown"], extensions=["tables", "fenced_code"])
    return HTMLResponse(f"<html><head><meta charset='utf-8'><title>{rep['title']}</title><style>body{{font-family:Georgia,serif;max-width:860px;margin:2rem auto;line-height:1.55}}"
                        "table{border-collapse:collapse;font-size:.85em}td,th{border:1px solid #bbb;padding:4px 6px;vertical-align:top}blockquote{color:#555}</style></head>"
                        f"<body>{body}</body></html>")


@app.get("/runs/{run_id}/graph")
def run_graph(run_id: str, fmt: Literal["json", "cypher"] = "json"):
    g = _result(run_id).get("graph")
    if not g:
        raise HTTPException(404, "no graph")
    return PlainTextResponse(to_cypher(g)) if fmt == "cypher" else g


def to_latex(md: str) -> str:
    import re
    t = md
    t = re.sub(r"^# (.+)$", r"\\section*{\1}", t, flags=re.M)
    t = re.sub(r"^## (.+)$", r"\\section{\1}", t, flags=re.M)
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"\*(.+?)\*", r"\\emph{\1}", t)
    t = t.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")
    return "\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n\\begin{document}\n" + t + "\n\\end{document}\n"


# ------------------------------------------------------------------ evaluation
@app.get("/evaluation/results")
def eval_results():
    out = {}
    for f in settings.results_dir.glob("*.json"):
        try:
            out[f.stem] = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            pass
    return out
