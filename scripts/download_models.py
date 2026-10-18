"""Pre-download every model ScholarAgent uses (so the first run is not blocked by downloads)."""
import sys
from huggingface_hub import snapshot_download

MODELS = {
    "LLM (served by vLLM, FP8 on the fly)": ("Qwen/Qwen2.5-7B-Instruct", ["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"]),
    "embedder": ("BAAI/bge-base-en-v1.5", None),
    "reranker": ("BAAI/bge-reranker-v2-m3", None),
    "NLI base (claim verifier)": ("cross-encoder/nli-deberta-v3-base", None),
}
extra = sys.argv[1:]          # e.g. BAAI/bge-m3 BAAI/bge-large-en-v1.5 for the embedder experiments
for label, (name, pats) in MODELS.items():
    print("->", label, name, flush=True)
    snapshot_download(name, allow_patterns=pats)
for name in extra:
    print("-> extra", name, flush=True)
    snapshot_download(name)
print("models ready")
