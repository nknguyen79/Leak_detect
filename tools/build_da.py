#!/usr/bin/env python3
"""Dựng notebook thích nghi miền (1D/2D-CNN + DANN/CDAN + Deep CORAL) từ tools/da_cells.py.

    python tools/build_da.py                                   # dựng cả hai kịch bản
    python tools/build_da.py --scenario sim2real <out.ipynb>   # chỉ một kịch bản

Kịch bản:
    sim2real  Sx (âm thanh mô phỏng tuyến ống rò rỉ) -> Rx (thu thực địa)
    pvc2hdpe  PVC -> HDPE

Mỗi cell code được biên dịch thử (compile) trước khi ghi, nên lỗi cú pháp dừng ngay ở đây.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import da_cells as C  # noqa: E402

DEFAULT_OUT = {
    "sim2real": "notebooks/leak-da-sim2real-sx-rx-kaggle.ipynb",
    "pvc2hdpe": "notebooks/leak-da-pvc-hdpe-dann-coral-kaggle.ipynb",
}


def build(scenario, dst):
    cells = []
    for i, (kind, text) in enumerate(C.render(scenario)):
        cid = f"da-{scenario}-{i:03d}"
        if kind == "code":
            compile(text, f"<{scenario} cell {i}>", "exec")
            cells.append({"id": cid, "cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": text.splitlines(keepends=True)})
        else:
            cells.append({"id": cid, "cell_type": "markdown", "metadata": {},
                          "source": text.splitlines(keepends=True)})
    src, tgt = C.NAMES[scenario]
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
            "leak_da": {"scenario": scenario, "source_domain": src, "target_domain": tgt,
                        "methods": "Source-only, AdaBN, Deep CORAL, DANN (GRL), DANN+CORAL, CDAN+E+CORAL, "
                                   "DANN+CORAL+DAug, Few-shot (target only), Few-shot FT, Oracle",
                        "built_by": "tools/build_da.py"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    n_code = sum(c["cell_type"] == "code" for c in cells)
    print(f"[{scenario}] {dst}: {len(cells)} cell ({n_code} code, {len(cells) - n_code} markdown)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=sorted(C.NAMES), default=None)
    ap.add_argument("out", nargs="?", default=None)
    a = ap.parse_args()
    if a.scenario:
        build(a.scenario, a.out or DEFAULT_OUT[a.scenario])
    else:
        for sc, out in DEFAULT_OUT.items():
            build(sc, out)
