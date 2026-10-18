"""Background research jobs. State lives in the DB, so the UI can poll progress and reload finished runs."""
from __future__ import annotations

import datetime as dt
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

from ..agents.services import get_services
from ..agents.supervisor import Supervisor, summarize_state
from ..utils.logging import get_logger, set_run

log = get_logger("jobs")
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="research")
_sup: Supervisor | None = None
_lock = threading.Lock()


def supervisor() -> Supervisor:
    global _sup
    with _lock:
        if _sup is None:
            _sup = Supervisor(get_services())
        return _sup


def submit(run_id: str) -> None:
    _pool.submit(_run, run_id)


def _run(run_id: str) -> None:
    svc = get_services()
    store = svc.lib.store
    run = store.get_run(run_id)
    if not run:
        return
    set_run(run_id)
    store.update_run(run_id, status="running")
    p = run["params"]
    try:
        state = supervisor().run(
            run["question"], mode=run["mode"], run_id=run_id,
            params={k: v for k, v in p.items() if k != "paper_ids"},
            paper_ids=p.get("paper_ids") or ([] if not p.get("collection_id") else store.collection_papers(p["collection_id"])),
            emit=lambda ev: store.add_run_event(run_id, ev))
        result = summarize_state(state, svc)
        store.update_run(run_id, status="done", result=result, finished_at=dt.datetime.utcnow())
        store.add_run_event(run_id, {"stage": "done", "message": "Research complete"})
    except Exception as e:  # noqa: BLE001
        log.error("run %s failed: %s", run_id, traceback.format_exc())
        store.update_run(run_id, status="failed", error=repr(e)[:2000], finished_at=dt.datetime.utcnow())
        store.add_run_event(run_id, {"stage": "error", "message": f"Run failed: {e!r}"[:300]})
