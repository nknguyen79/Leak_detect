#!/usr/bin/env python3
"""Dựng notebook Adversarial Domain Adaptation (ADA v2) từ các tệp nội dung cell.

Bản v1 (notebooks/leak-ada-dann-kaggle-v1.ipynb) được dựng từ commit ffda6c3 của các tệp này.

    python tools/build_ada.py notebooks/leak-ada-dann-kaggle-v2.ipynb

Nội dung cell nằm ở tools/ada_cells_1.py (giới thiệu, dữ liệu đa miền, EDA dịch chuyển miền),
tools/ada_cells_2.py (kiến trúc, giao thức, huấn luyện) và tools/ada_cells_3.py (kết quả, phân tích,
xuất tư liệu). Mọi cell code được biên dịch thử trước khi ghi để bắt lỗi cú pháp sớm.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ada_cells_1 as A  # noqa: E402
import ada_cells_2 as B  # noqa: E402
import ada_cells_3 as C  # noqa: E402

DST = Path(sys.argv[1] if len(sys.argv) > 1 else "notebooks/leak-ada-dann-kaggle-v2.ipynb")

cells = []
for kind, text in A.CELLS + B.CELLS + C.CELLS:
    text = text.strip("\n")
    if kind == "code":
        compile(text, f"<cell {len(cells)}>", "exec")
        cells.append({"cell_type": "code", "id": f"ada-{len(cells):03d}", "metadata": {},
                      "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)})
    else:
        cells.append({"cell_type": "markdown", "id": f"ada-{len(cells):03d}", "metadata": {},
                      "source": text.splitlines(keepends=True)})

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12", "mimetype": "text/x-python",
                          "file_extension": ".py", "pygments_lexer": "ipython3",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "nbconvert_exporter": "python"},
        "accelerator": "GPU",
        "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isGpuEnabled": True,
                   "isInternetEnabled": False, "language": "python", "sourceType": "notebook"},
        "leak_ada_version": "v2",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
n_code = sum(c["cell_type"] == "code" for c in cells)
print(f"Wrote {DST}: {len(cells)} cells ({n_code} code)")
