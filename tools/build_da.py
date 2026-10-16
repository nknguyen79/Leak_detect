#!/usr/bin/env python3
"""Dựng notebook thích nghi miền PVC -> HDPE (1D/2D-CNN + DANN/GRL + Deep CORAL) từ tools/da_cells.py.

    python tools/build_da.py notebooks/leak-da-pvc-hdpe-dann-coral-kaggle.ipynb

Mỗi cell code được biên dịch thử (compile) trước khi ghi, nên lỗi cú pháp dừng ngay ở đây
thay vì lộ ra giữa lần chạy trên Kaggle.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import da_cells as C  # noqa: E402

DST = Path(sys.argv[1] if len(sys.argv) > 1 else "notebooks/leak-da-pvc-hdpe-dann-coral-kaggle.ipynb")

cells = []
for i, (kind, text) in enumerate(C.CELLS):
    if kind == "code":
        compile(text, f"<cell {i}>", "exec")
        cells.append({"id": f"da-{i:03d}", "cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": text.splitlines(keepends=True)})
    else:
        cells.append({"id": f"da-{i:03d}", "cell_type": "markdown", "metadata": {},
                      "source": text.splitlines(keepends=True)})

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "mimetype": "text/x-python", "file_extension": ".py",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "pygments_lexer": "ipython3", "nbconvert_exporter": "python"},
        "accelerator": "GPU",
        "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isGpuEnabled": True,
                   "isInternetEnabled": True, "language": "python", "sourceType": "notebook"},
        "leak_da": {"source_domain": "PVC", "target_domain": "HDPE",
                    "methods": "Source-only, AdaBN, Deep CORAL, DANN (GRL), DANN+CORAL, DANN+CORAL+PhysAug, "
                               "Few-shot FT, Oracle",
                    "built_by": "tools/build_da.py"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
n_code = sum(c["cell_type"] == "code" for c in cells)
print(f"{DST}: {len(cells)} cell ({n_code} code, {len(cells) - n_code} markdown)")
