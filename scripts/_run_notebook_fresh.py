"""
One-shot fresh Restart-&-Run-All of the submission notebook.
Runs nbconvert --execute against a separate _executed copy so the
original is never touched until we verify the result.

Windows fix: force the Selector event-loop policy so zmq/jupyter_client
does not die on the Proactor loop (the cause of the silent early exit).
"""
import asyncio
import sys
import time
import traceback
from pathlib import Path

# --- Windows zmq/jupyter event-loop fix (must run before nbclient imports) ---
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "notebooks" / "EPC_Oxford_Analysis.ipynb"
OUT = ROOT / "notebooks" / "EPC_Oxford_Analysis_executed.ipynb"

print(f"[fresh-run] source : {SRC}", flush=True)
print(f"[fresh-run] output : {OUT}", flush=True)
print(f"[fresh-run] python : {sys.version.split()[0]}", flush=True)

t0 = time.time()
nb = nbformat.read(SRC, as_version=4)
n_code = sum(1 for c in nb.cells if c.cell_type == "code")
print(f"[fresh-run] cells  : {len(nb.cells)} total, {n_code} code", flush=True)

client = NotebookClient(
    nb,
    timeout=3000,                 # per-cell seconds
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT / "notebooks")}},
    allow_errors=False,           # stop on first error so we catch problems
)

try:
    client.execute()
    nbformat.write(nb, OUT)
    dt = time.time() - t0
    n_err = sum(
        1
        for c in nb.cells
        if c.cell_type == "code"
        for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    )
    print(f"[fresh-run] DONE in {dt/60:.1f} min  | error-cells={n_err}", flush=True)
    print("[fresh-run] EXIT_CODE=0", flush=True)
except Exception as e:
    dt = time.time() - t0
    # persist whatever executed so far for diagnosis
    try:
        nbformat.write(nb, OUT)
    except Exception:
        pass
    print(f"[fresh-run] FAILED after {dt/60:.1f} min: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    print("[fresh-run] EXIT_CODE=1", flush=True)
    sys.exit(1)
