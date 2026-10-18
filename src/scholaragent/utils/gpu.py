"""GPU helpers."""
from __future__ import annotations

import gc
import threading
from concurrent.futures import ThreadPoolExecutor


def resolve_device(pref: str = "cuda") -> str:
    try:
        import torch
        if pref.startswith("cuda") and torch.cuda.is_available():
            return pref
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def gpu_info() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cuda": False}
        free, total = torch.cuda.mem_get_info()
        return {"cuda": True, "name": torch.cuda.get_device_name(0),
                "free_gb": round(free / 1e9, 2), "total_gb": round(total / 1e9, 2)}
    except Exception as e:  # noqa: BLE001
        return {"cuda": False, "error": repr(e)}


_gpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")


def on_gpu(fn, *args, **kwargs):
    """Run `fn` on the single dedicated GPU thread. Every calling thread would otherwise create its own cuBLAS handle +
    workspace (~60 MB each) - with dozens of agent/worker threads that is >1 GB of hidden VRAM, and CUDA work is
    serialised by locks anyway."""
    if threading.current_thread().name.startswith("gpu"):
        return fn(*args, **kwargs)
    return _gpu_pool.submit(fn, *args, **kwargs).result()
