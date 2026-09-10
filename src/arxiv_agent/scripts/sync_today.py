#!/usr/bin/env python3
"""Standalone entry point for the daily embedding sync.

Invoked as a subprocess by prelaunch.py. Uses sys.path manipulation to
import recommend_arxiv_papers directly (as a standalone module) without
triggering the arxiv_agent package tree (graph → nodes → tools) which
would cause a circular import deadlock.
"""

import importlib.util
import logging
import sys
from pathlib import Path

# Load .env before anything reads os.environ
from dotenv import load_dotenv

load_dotenv()

# Directly load recommend_arxiv_papers.py as a standalone module, bypassing
# the arxiv_agent package __init__.py which would pull in the full graph.
_script_dir = Path(__file__).resolve().parent.parent / "tools"
_spec = importlib.util.spec_from_file_location(
    "recommend_arxiv_papers",
    _script_dir / "recommend_arxiv_papers.py",
    submodule_search_locations=[],
)
if _spec is None or _spec.loader is None:
    sys.exit(f"Cannot load {_script_dir / 'recommend_arxiv_papers.py'}")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["recommend_arxiv_papers"] = _mod
_spec.loader.exec_module(_mod)

_ensure_db_ready = _mod._ensure_db_ready
_sync_date = _mod._sync_date

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if len(sys.argv) != 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} <YYYY-MM-DD>\n")
        sys.exit(1)

    target_date = sys.argv[1]
    _ensure_db_ready()
    count = _sync_date(target_date)
    logging.info("Synced %d chunks for %s.", count, target_date)
