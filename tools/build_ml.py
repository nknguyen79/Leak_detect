#!/usr/bin/env python3
"""Dựng notebook đặc trưng thủ công + ML từ các tệp nguồn dạng "percent" trong tools/ml_nb/.

    python tools/build_ml.py [notebooks/leak-ml-handcrafted-kaggle-v1.ipynb]

Mỗi tệp tools/ml_nb/NN_*.py được ghép theo thứ tự tên. Ranh giới cell là dòng bắt đầu bằng "# %%";
"# %% [markdown]" mở một cell markdown (mỗi dòng bỏ tiền tố "# "). Sửa mã ở tools/ml_nb/ rồi chạy lại
script này, không sửa trực tiếp .ipynb.
"""
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "ml_nb"
DST = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "notebooks" / "leak-ml-handcrafted-kaggle-v1.ipynb"


def parse(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = list(buf)
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            return
        if kind == "markdown":
            md = [ln[2:] if ln.startswith("# ") else ln[1:] if ln.startswith("#") else ln for ln in lines]
            cells.append(nbformat.v4.new_markdown_cell("\n".join(md)))
        else:
            cells.append(nbformat.v4.new_code_cell("\n".join(lines)))

    for ln in text.splitlines():
        if ln.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in ln else "code"), []
        else:
            buf.append(ln)
    flush()
    return cells


def main():
    cells = []
    for f in sorted(SRC.glob("[0-9][0-9]_*.py")):
        cells += parse(f.read_text(encoding="utf-8"))
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"},
                   "kaggle": {"accelerator": "none", "isInternetEnabled": True, "isGpuEnabled": False}}
    for c in nb.cells:                       # id ổn định -> diff gọn
        c["id"] = f"c{nb.cells.index(c):03d}"
    DST.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, DST)
    n_code = sum(c.cell_type == "code" for c in cells)
    print(f"{DST}: {len(cells)} cells ({n_code} code, {len(cells) - n_code} markdown)")


if __name__ == "__main__":
    main()
