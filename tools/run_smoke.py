#!/usr/bin/env python3
"""Chạy notebook end-to-end ở chế độ smoke test (3 epoch, 2 fold) trên dữ liệu mô phỏng.

    LEAK_OUT=/tmp/leak_smoke python tools/run_smoke.py notebooks/leak-cnn-pvc-kaggle-v7.ipynb

Dừng ở cell lỗi đầu tiên và in chỉ số cell + traceback; lưu notebook đã chạy cạnh LEAK_OUT.
"""
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

src = Path(sys.argv[1])
out_dir = Path(os.environ.setdefault("LEAK_OUT", "/tmp/leak_smoke")).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("LEAK_SMOKE", "1")
os.environ.setdefault("LEAK_DATA", "")          # rỗng -> không có dữ liệu thật -> SYNTHETIC
os.environ.setdefault("MPLBACKEND", "Agg")

nb = nbformat.read(src, as_version=4)
client = NotebookClient(nb, timeout=3600, kernel_name="python3", allow_errors=False,
                        resources={"metadata": {"path": str(out_dir)}})
t0 = time.time()
ok = True
try:
    client.execute()
except CellExecutionError as e:
    ok = False
    print(str(e)[-6000:])
finally:
    dst = out_dir / (src.stem + ".executed.ipynb")
    nbformat.write(nb, dst)
    n_run = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("execution_count"))
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"\n{'OK' if ok else 'FAILED'}: {n_run}/{n_code} code cells executed in {time.time()-t0:.0f}s -> {dst}")
sys.exit(0 if ok else 1)
