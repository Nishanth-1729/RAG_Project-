"""Citation / knowledge graph.

Edges between papers of the corpus are recovered from the *parsed reference lists* (arXiv ids, DOIs and
fuzzy title matching) - no external API needed - and optionally enriched from Semantic Scholar.
Entity nodes (Approach / Dataset) come from the evidence-linked paper analyses.
Exports: networkx graph, JSON for the UI, Cypher for Neo4j.
"""
from __future__ import annotations

import json
import re

import networkx as nx
from rapidfuzz import fuzz

from ..config import settings
from ..ingestion.models import ParsedDocument
from ..utils.logging import get_logger
from ..utils.text import normalize_title

log = get_logger("graph")


def _ref_entries(text: str) -> list[str]:
    """Split a raw reference list into entries (works for numbered and author-year styles)."""
    text = re.sub(r"\n\s*\n", "\n", text)
    parts = re.split(r"\n(?=\[\d+\]|\d+\.\s+[A-Z])", text)
    if len(parts) < 5:                      # author-year: a new entry starts after a line that ends with a year/page-number
        parts = re.split(r"(?<=\d{4}\.)\s*\n|(?<=\.)\n(?=[A-Z][a-z]+,? [A-Z]\.)", text)
    return [re.sub(r"\s+", " ", p).strip() for p in parts if len(p.strip()) > 25]


def match_references(doc: ParsedDocument, corpus: dict[str, dict], threshold: float = 88.0) -> list[str]:
    """paper_ids from `corpus` (paper_id -> {title, arxiv_id, doi}) cited by `doc`."""
    cited: set[str] = set()
    entries = _ref_entries(doc.references_text)
    blob = " ".join(entries)
    low = blob.lower()
    for pid, m in corpus.items():
        if pid == doc.paper_id:
            continue
        if m.get("arxiv_id") and m["arxiv_id"].lower() in low:
            cited.add(pid)
            continue
        if m.get("doi") and m["doi"].lower() in low:
            cited.add(pid)
            continue
        nt = normalize_title(m.get("title", ""))
        if len(nt) < 20:
            continue
        nb = normalize_title(blob)
        if nt in nb:
            cited.add(pid)
            continue
        # fuzzy: compare against each entry (titles are embedded inside references)
        for e in entries:
            if fuzz.partial_ratio(nt, normalize_title(e)) >= threshold and len(nt) >= 25:
                cited.add(pid)
                break
    return sorted(cited)


class GraphAgent:
    def __init__(self, store):
        self.store = store

    def build(self, paper_ids: list[str], comparison: dict | None, analyses: dict[str, dict]) -> dict:
        corpus = {p["paper_id"]: p for p in self.store.list_papers(paper_ids)}
        G = nx.DiGraph()
        for pid, p in corpus.items():
            G.add_node(pid, type="Paper", label=p["title"], year=p["year"], authors=p["authors"][:3])
        edges: list[dict] = []
        for pid in paper_ids:
            cache = settings.parsed_dir / f"{pid}.json"
            if not cache.exists():
                continue
            doc = ParsedDocument.from_dict(json.loads(cache.read_text()))
            for tgt in match_references(doc, corpus):
                G.add_edge(pid, tgt, relation="CITES")
                edges.append({"src": pid, "dst": tgt, "relation": "CITES"})
        if comparison:
            for r in comparison["rows"]:
                cat = f"approach:{r['category']}"
                G.add_node(cat, type="Approach", label=r["category"])
                G.add_edge(r["paper_id"], cat, relation="PROPOSES")
                edges.append({"src": r["paper_id"], "dst": cat, "relation": "PROPOSES", "dst_type": "Approach"})
        for pid, a in analyses.items():
            for it in a["fields"].get("datasets", [])[:4]:
                name = re.split(r"[,;(]| and ", it["text"])[0].strip()[:50]
                if len(name) < 3:
                    continue
                node = f"dataset:{normalize_title(name)}"
                G.add_node(node, type="Dataset", label=name)
                G.add_edge(pid, node, relation="EVALUATES_ON")
                edges.append({"src": pid, "dst": node, "relation": "EVALUATES_ON", "dst_type": "Dataset"})
        self.store.upsert_edges(edges)
        return self.to_json(G)

    @staticmethod
    def to_json(G: nx.DiGraph) -> dict:
        nodes = [{"id": n, **d} for n, d in G.nodes(data=True)]
        links = [{"source": u, "target": v, **d} for u, v, d in G.edges(data=True)]
        paper_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Paper"]
        cites = nx.DiGraph([(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "CITES"])
        most_cited = sorted(((n, cites.in_degree(n)) for n in cites.nodes), key=lambda x: -x[1])[:5]
        return {"nodes": nodes, "links": links,
                "stats": {"papers": len(paper_nodes), "cites_edges": cites.number_of_edges(),
                          "most_cited_in_corpus": [{"paper_id": n, "in_degree": k, "title": G.nodes[n].get("label")} for n, k in most_cited if k]}}


def to_cypher(graph: dict) -> str:
    """Cypher script to load the graph into Neo4j."""
    q = lambda s: json.dumps(str(s), ensure_ascii=False)  # noqa: E731
    lines = []
    for n in graph["nodes"]:
        lines.append(f"MERGE (:{n['type']} {{id: {q(n['id'])}, label: {q(n.get('label', ''))}}});")
    for l in graph["links"]:
        lines.append(f"MATCH (a {{id: {q(l['source'])}}}), (b {{id: {q(l['target'])}}}) MERGE (a)-[:{l['relation']}]->(b);")
    return "\n".join(lines)
