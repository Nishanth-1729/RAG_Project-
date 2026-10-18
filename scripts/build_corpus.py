"""Build the paper corpora used for evaluation (test) and training, entirely via ScholarAgent's own search layer.

  eval  : papers on 'machine learning for numerical linear algebra solvers'  (the project's headline topic)
  train : a larger, *disjoint* set of related scientific-ML / numerical-analysis papers

Split is by PAPER (never by chunk) so no test passage is ever seen in training.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from scholaragent.config import settings
from scholaragent.retrieval.embedder import Embedder
from scholaragent.search.fetcher import fetch_pdf
from scholaragent.search.service import search_literature

EVAL_QUERIES = [
    "learned preconditioner iterative linear solver",
    "graph neural network preconditioner sparse linear systems",
    "neural network accelerate conjugate gradient method",
    "machine learning algebraic multigrid",
    "learned sparse approximate inverse preconditioner",
    "neural incomplete factorization preconditioner",
    "deep learning Krylov subspace GMRES",
    "learning to solve linear systems neural",
    "machine learning low-rank approximation hierarchical matrices",
    "neural network fast multipole method",
    "reinforcement learning linear solver selection",
    "deep learning sparse direct solver pivoting ordering",
    "neural network eigenvalue solver",
    "learning multigrid smoother neural",
    "transformer numerical linear algebra",
]
TRAIN_QUERIES = [
    "neural operator partial differential equations",
    "physics-informed neural networks",
    "Fourier neural operator",
    "hierarchical matrices H2 matrix fast algorithms",
    "fast multipole method",
    "randomized low-rank approximation",
    "randomized numerical linear algebra sketching",
    "multigrid method convergence",
    "domain decomposition preconditioner",
    "Krylov subspace methods convergence analysis",
    "tensor decomposition low rank",
    "kernel matrix fast summation",
    "scientific machine learning surrogate model",
    "learned optimizer meta-learning",
    "deep learning inverse problems",
    "mixed precision iterative refinement",
    "GPU sparse matrix computations",
    "neural network PDE solver finite element",
    "operator learning DeepONet",
    "preconditioned conjugate gradient",
    "algebraic multigrid aggregation",
    "iterative methods for sparse linear systems",
    "Gaussian process scalable kernel inference",
    "differentiable programming numerical solvers",
    "graph neural network mesh simulation",
    "learning to optimize algorithm unrolling",
    "Cholesky factorization sparse",
    "matrix completion low rank",
    "neural network quantum many-body solver",
    "uncertainty quantification surrogate neural network",
]


def collect(queries, year_from, year_to, per_q, n_total, exclude, embed, rank_query):
    recs, stats = search_literature(queries, sources=("arxiv",), per_source=per_q, year_from=year_from, year_to=year_to,
                                    rank_query=rank_query, embed_fn=embed)
    recs = [r for r in recs if r.arxiv_id and r.abstract and r.paper_id not in exclude]
    return recs[:n_total], stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-n", type=int, default=40)
    ap.add_argument("--train-n", type=int, default=140)
    ap.add_argument("--year-from", type=int, default=2018)
    args = ap.parse_args()
    emb = Embedder()
    out_dir = settings.eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ev, st1 = collect(EVAL_QUERIES, args.year_from, 2026, 12, args.eval_n, set(), emb,
                      "machine learning methods to accelerate numerical linear algebra solvers and preconditioners")
    eval_ids = {r.paper_id for r in ev}
    tr, st2 = collect(TRAIN_QUERIES, args.year_from, 2026, 10, args.train_n, eval_ids, emb,
                      "numerical methods and machine learning for scientific computing")
    print("eval", len(ev), st1)
    print("train", len(tr), st2)

    def dl(r):
        p = fetch_pdf(r)
        return r, p

    manifest = {"eval": [], "train": []}
    for split, recs in (("eval", ev), ("train", tr)):
        with ThreadPoolExecutor(max_workers=2) as ex:
            for r, p in ex.map(dl, recs):
                if p:
                    d = r.to_dict()
                    d["pdf"] = str(p)
                    manifest[split].append(d)
                    print(f"[{split}] ok  {r.paper_id} {r.title[:70]}")
                else:
                    print(f"[{split}] FAIL {r.paper_id} {r.title[:70]}")
        (out_dir / "corpus.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print({k: len(v) for k, v in manifest.items()})


if __name__ == "__main__":
    main()
