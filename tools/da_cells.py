# -*- coding: utf-8 -*-
"""Nguồn của notebook thích nghi miền cho nhận dạng rò rỉ nước bằng âm thanh
(1D/2D-CNN + GRL/DANN + CDAN + Deep CORAL), dùng chung cho hai kịch bản:

    sim2real : Sx (âm thanh leak/noleak MÔ PHỎNG tuyến ống rò rỉ)  ->  Rx (thu THỰC ĐỊA)   [mặc định]
    pvc2hdpe : ống PVC  ->  ống HDPE (lệch miền do vật liệu, theo tài liệu "domain_shift")

Mỗi phần tử của CELLS là ("markdown" | "code", nguồn). Markdown có thể là dict {kịch_bản: văn bản}
(thiếu khoá = bỏ cell ở kịch bản đó); «S» / «T» được thay bằng tên miền nguồn / đích; trong code,
"__SCENARIO__" được thay bằng tên kịch bản. Dựng notebook:

    python tools/build_da.py                     # dựng cả hai notebook
    python tools/build_da.py --scenario sim2real notebooks/leak-da-sim2real-sx-rx-kaggle.ipynb
"""

NAMES = {"sim2real": ("Sx", "Rx"), "pvc2hdpe": ("PVC", "HDPE")}
CELLS = []


def md(s):
    s = {k: v.strip("\n") for k, v in s.items()} if isinstance(s, dict) else s.strip("\n")
    CELLS.append(("markdown", s))


def code(s):
    CELLS.append(("code", s.strip("\n")))


def render(scenario):
    S, T = NAMES[scenario]
    out = []
    for kind, s in CELLS:
        if isinstance(s, dict):
            s = s.get(scenario, "")
        if not s:
            continue
        s = s.replace("«S»", S).replace("«T»", T)
        if kind == "code":
            s = s.replace("__SCENARIO__", scenario)
        out.append((kind, s))
    return out


# =============================================================================
# 0. Tiêu đề
# =============================================================================
md({"sim2real": r'''
# Thích nghi miền mô phỏng → thực địa (Sx → Rx) cho nhận dạng rò rỉ nước bằng âm thanh: 1D/2D-CNN + DANN/CDAN (GRL) + Deep CORAL

**Sim-to-Real Domain Adaptation for Acoustic Leak Detection — from simulated pipeline-leak recordings (Sx) to field recordings (Rx) with 1D/2D-CNNs, a Gradient Reversal Layer (DANN, CDAN) and Deep CORAL**

---

### Bài toán / Problem

| | **Sx — miền nguồn (mô phỏng)** | **Rx — miền đích (thực địa)** |
|---|---|---|
| Nguồn gốc | testbed/bàn thí nghiệm hoặc mô phỏng số tuyến ống rò rỉ | thu thực tế ngoài hiện trường các tuyến ống |
| Nhãn | đầy đủ, sạch, điều kiện kiểm soát (áp suất, kích thước lỗ, khoảng cách) | ít hoặc không có; nhãn **yếu** (bản ghi "leak" không nghe rò suốt thời gian) |
| Nhiễu nền | thấp, dừng | xe cộ, bơm, người, gió, xung va đập, hài 50 Hz; **không dừng** |
| Kênh đo | một cảm biến, gắn cố định | cảm biến/ghép nối/đất khác nhau theo điểm đo |
| Tỉ lệ lớp | thường cân bằng | thường lệch (ít bản ghi rò) |

Mục tiêu: học bộ phân loại rò rỉ từ **Sx có nhãn**, dùng **Rx không nhãn** để thích nghi, đánh giá trên **Rx
kiểm thử tách rời** (nhãn Rx chỉ để đánh giá). Hàm mục tiêu (minimax, hiện thực bằng GRL):

$$
\min_{\theta_f,\theta_y}\max_{\theta_d}\;
\underbrace{\frac{1}{N_S}\sum_{i}\mathcal{L}_y\big(G_y(G_f(x_i^S)),y_i^S\big)}_{\text{chỉ Sx có nhãn}}
-\lambda_p\,\underbrace{\mathcal{L}_d\big(G_d(\,\cdot\,)\big)}_{\text{DANN: } f \;|\; \text{CDAN: } f\otimes \hat y}
+\beta_p\,\underbrace{\frac{1}{4d^2}\lVert C_S-C_T\rVert_F^2}_{\text{Deep CORAL}}
$$

với $\lambda_p=\lambda_{\max}\big(\frac{2}{1+e^{-\gamma p}}-1\big)$ (Ganin & Lempitsky, 2015).

### Năm câu hỏi nghiên cứu mà notebook trả lời bằng số liệu

| # | Câu hỏi | Bằng chứng |
|---|---|---|
| Q1 | Khoảng cách mô phỏng → thực địa lớn đến đâu? | Source-only so với Oracle (Rx có nhãn) |
| Q2 | Thích nghi KHÔNG nhãn có thu hẹp khoảng cách? | AdaBN / CORAL / DANN / **DANN+CORAL** / CDAN+CORAL / +DAug, ΔAUC ghép cặp + Holm |
| Q3 | Dữ liệu mô phỏng có **giá trị** khi đã có vài nhãn thực địa? | Few-shot FT (từ Sx) so với Few-shot **chỉ Rx** (cùng bản ghi) |
| Q4 | Dải tần nào **chuyển giao** được từ Sx sang Rx? | hồ sơ dải: phân biệt trên Sx × (1 − độ tách miền), không dùng nhãn Rx |
| Q5 | Lệch miền thể hiện ở đâu? | chỉ số chất lượng tín hiệu theo miền, phổ nền, không gian ẩn (A-distance, CORAL, MMD) |

### Bố trí dữ liệu / Data layout

```
/kaggle/input/<dataset>/
├── Sx/            (hoặc sim/, simulated/, lab/, testbed/, mo_phong/  — hoặc chỉ định cfg.source_root)
│   ├── leak/      sim_P2_D5_001.wav ...
│   └── noleak/    ...
└── Rx/            (hoặc real/, field/, hien_truong/, thuc_te/      — hoặc chỉ định cfg.target_root)
    ├── leak/      R001_seg01.wav ...   <- nhãn chỉ để ĐÁNH GIÁ
    ├── noleak/    ...
    └── unlabeled/ ...                  <- tuỳ chọn: bản ghi chưa biết nhãn (chỉ để thích nghi)
```

Chế độ tự nhận diện: **`real`** (có cả Sx và Rx) · **`simulated_source`** (chỉ có Rx, ví dụ bộ hiện trường
`field_rec_8k_V7.2` của v7 — Sx được sinh bằng **bộ mô phỏng tích hợp**) · **`pseudo_target`** (chỉ có Sx — Rx
giả lập bằng biến đổi "hiện trường hoá" các bản ghi Sx **tách rời**; chỉ kiểm chứng pipeline) · **`synthetic`**.

> **Cách dùng nhanh**
> 1. `Add Data`: gắn dataset Sx và Rx; sửa `cfg.source_root` / `cfg.target_root` (mục 1.2) nếu tên thư mục không chứa từ khoá miền.
> 2. Nếu Sx đặt tên tệp khác `Rxxx`: sửa `cfg.record_regex_source` (mỗi lần mô phỏng/kịch bản = một nhóm).
> 3. Nếu biết điểm khảo sát của Rx: điền `cfg.target_group_csv` (`record,group`) để chia fold theo **điểm**.
> 4. `GPU T4 x2` → `Run All` (~40–90 phút tuỳ kích thước Sx). Chạy thử: `LEAK_SMOKE=1`.
> 5. Đổi `SCENARIO = "pvc2hdpe"` ở mục 1.2 để quay lại bài toán PVC → HDPE.
''', "pvc2hdpe": r'''
# Thích nghi miền PVC → HDPE cho nhận dạng rò rỉ nước bằng âm thanh: 1D/2D-CNN + DANN (GRL) + Deep CORAL

**Cross-material Unsupervised Domain Adaptation for Acoustic Leak Detection in Low-Pressure Plastic Pipes — 1D/2D-CNN with a Gradient Reversal Layer (DANN) and Deep CORAL**

---

### Bài toán / Problem

Mô hình học trên ống **PVC** (miền nguồn, có nhãn) mất mạnh độ chính xác khi sang ống **HDPE** (miền đích). HDPE mềm
hơn nhiều ($E \approx 0{,}7$–$1{,}1$ GPa so với $2{,}8$–$3{,}4$ GPa), nên theo Korteweg
$c = c_0\left[1 + \frac{B\,D}{E\,e}\right]^{-1/2}$ vận tốc sóng giảm (≈ 380–480 → 180–260 m/s): phổ rò rỉ **dịch xuống**
(≈ 80–280 Hz → 20–120 Hz) và suy hao dốc hơn. Hệ quả là *covariate shift* $P_S(X) \neq P_T(X)$.

| Giai đoạn (sơ đồ) | Nội dung | Mục |
|---|---|---|
| **1. Lệch miền vật lý** | Korteweg; PSD, hồ sơ dải, tâm mode VMD theo miền | 3, 4, 5 |
| **2. Tiền xử lý & CNN** | Lọc dải 15–350 Hz, notch 50 Hz, VMD, CWT Morlet tần số log, CMVN theo miền; 1D-CNN và 2D-CNN + CBAM dùng chung $G_f$ | 3, 5, 6 |
| **3. Căn chỉnh miền** | GRL/DANN (và CDAN), Deep CORAL $\frac{1}{4d^2}\lVert C_S - C_T\rVert_F^2$ | 6, 8 |
| **4. Hội tụ bất biến** | t-SNE, proxy $\mathcal{A}$-distance, CORAL/MMD trong không gian ẩn | 11 |

$$
\min_{\theta_f,\theta_y}\max_{\theta_d}\;
\frac{1}{N_S}\sum_{i}\mathcal{L}_y\big(G_y(G_f(x_i^S)),y_i^S\big)
-\lambda_p\Big[\frac{1}{N_S}\sum_i\mathcal{L}_d\big(G_d(f_i^S),0\big)+\frac{1}{N_T}\sum_j\mathcal{L}_d\big(G_d(f_j^T),1\big)\Big]
+\beta_p\frac{1}{4d^2}\lVert C_S-C_T\rVert_F^2
$$

Thiết kế thực nghiệm (mục 5 tài liệu): Source-only, AdaBN, Deep CORAL, DANN, **DANN + CORAL**, CDAN + CORAL,
DANN + CORAL + DAug (giả lập vật lý PVC→HDPE: dịch tần, suy hao $e^{-\alpha f x}$, nhiễu nền), Few-shot chỉ HDPE,
Few-shot FT (LR 1e-4 / 1e-5 / 1e-3), Oracle. Mọi phân chia theo **bản ghi**, nhãn HDPE chỉ để đánh giá, mô hình lấy ở
**epoch cuối**, ngưỡng 0,5 (đăng ký trước).

```
/kaggle/input/<dataset>/PVC/{leak,noleak}   HDPE/{leak,noleak,unlabeled}
```

Chế độ: **`real`** · **`pseudo_target`** (chỉ có PVC thật, ví dụ bộ v7: một phần bản ghi PVC tách rời được biến đổi vật
lý thành "giả lập HDPE" — chỉ kiểm chứng pipeline) · **`synthetic`**. Đổi `SCENARIO = "sim2real"` ở mục 1.2 cho bài toán
mô phỏng → thực địa.
'''})

md({"sim2real": r'''
## 0. Phân tích và đánh giá chương trình khi chuyển sang Sx → Rx / Program review for the sim-to-real setting

Chương trình gốc được thiết kế cho PVC → HDPE, nơi lệch miền có **một cơ chế vật lý trội** (dịch tần theo vận tốc
sóng). Lệch mô phỏng → thực địa có **nhiều nguồn chồng lên nhau** và phần lớn KHÔNG phải dịch tần. Bảng dưới đây đánh
giá từng thành phần và điều chỉnh tương ứng trong notebook này.

| Thành phần | Giả định trong bản PVC→HDPE | Còn đúng cho Sx→Rx? | Điều chỉnh |
|---|---|---|---|
| **Cơ chế lệch miền** | dịch tần theo Korteweg + suy hao nhớt đàn hồi | **Phần lớn không** (thường cùng vật liệu). Lệch do nhiễu nền, kênh đo, SNR, tính không dừng, nhãn yếu, tỉ lệ lớp, sai số mô hình mô phỏng | Mục 4: **phân tích lệch dựa trên dữ liệu** (chỉ số chất lượng + độ tách miền từng chỉ số, phổ nền, hồ sơ dải); Korteweg chỉ dùng khi vật liệu Sx ≠ Rx |
| **Dải tần 15–350 Hz cố định** | hợp lý cho HDPE (20–120 Hz) | **Không**: dữ liệu PVC hiện trường (v7) có thông tin tới ~2,5 kHz | **Chọn dải tự động, không dùng nhãn Rx**: điểm = phân biệt trên Sx × (1 − độ tách Sx/Rx); `target_sr` suy ra từ dải |
| **Tần số lấy mẫu** | hai miền như nhau | Sx có thể 44,1/48 kHz, Rx 8 kHz → thành phần trên Nyquist của Rx **chỉ có ở Sx** = lối tắt | dải luôn ≤ 0,45 × sr gốc **nhỏ nhất của cả hai miền** |
| **Tăng cường PhysAug** | dịch tần + suy hao | sai loại nhiễu | **DAug** "hiện trường hoá" Sx: nhiễu theo **phổ nền của Rx** (không nhãn), xung va đập, EQ kênh ngẫu nhiên; dịch tần chỉ bật khi đo được ($r_{emp}\neq 1$) |
| **CMVN theo miền** | loại hàm truyền ống/đất | **Đúng, còn quan trọng hơn** (cảm biến/ghép nối khác hẳn) | giữ |
| **DANN (căn chỉnh biên)** | tỉ lệ lớp hai miền tương đương | **Rủi ro**: Sx cân bằng, Rx lệch → căn chỉnh biên đẩy leak Rx về phía noleak Sx (negative transfer) | thêm **CDAN+E** (căn chỉnh có điều kiện theo dự đoán), **lấy mẫu cân bằng lớp** ở Sx, ước lượng tỉ lệ rò của Rx bằng **BBSE** |
| **Đối chứng few-shot** | chỉ so với Source-only | thiếu "vài nhãn Rx, không dùng Sx" → **không trả lời được Sx có giá trị không** | thêm **Few-shot chỉ Rx** + so sánh ghép cặp (Q3) |
| **Chia fold Rx theo bản ghi** | đủ | thực địa: nhiều bản ghi tại một điểm → rò rỉ theo vị trí | tuỳ chọn chia theo **điểm khảo sát** (`target_group_csv`) |
| **Mã bản ghi** | cả hai miền `Rxxx` | Sx có quy ước tên riêng | regex **riêng từng miền** |
| **Kích thước Sx** | tương đương Rx | Sx thường lớn hơn nhiều | lấy mẫu phân tầng bản ghi Sx, giới hạn frame/bản ghi, giới hạn bước/epoch |
| **Chế độ dự phòng** | chỉ có PVC → "giả lập HDPE" | chỉ có Rx là tình huống thực tế nhất | chỉ có Rx → **bộ mô phỏng tích hợp** sinh Sx; chỉ có Sx → Rx giả lập "hiện trường hoá" |

**Giữ nguyên** (vẫn đúng): giao thức chia theo bản ghi + assertion, nhãn Rx chỉ để đánh giá, epoch cuối + ngưỡng 0,5
đăng ký trước, khởi tạo và lô Sx giống nhau giữa các phương pháp (so sánh ghép cặp), CI bootstrap theo bản ghi + Holm,
VMD, CWT, CBAM, AdaBN, few-shot chọn lọc theo tầng, phân tích không gian ẩn, xuất TorchScript.

**Hạn chế còn lại cần lưu ý khi viết bài:** (i) chọn mô hình không nhãn vẫn là bài toán mở — notebook dùng quy tắc
đăng ký trước; (ii) nhãn Rx yếu (MIL) — đánh giá chính ở **mức bản ghi**; (iii) căn chỉnh không thể sửa *concept shift*
(cơ chế âm học của rò rỉ thật khác mô phỏng) — mục 4.2 đo mức trùng khớp dải phân biệt Sx/Rx để phát hiện điều này.
'''})

# =============================================================================
# 1. Môi trường
# =============================================================================
md(r'''
## 1. Thiết lập môi trường / Environment setup
''')

code(r'''
# =========================================================
# 1.1 Imports & phụ thuộc / Imports and dependencies
#     Không phụ thuộc torchaudio: STFT/Mel/CWT viết bằng torch thuần.
# =========================================================
import os, sys, gc, json, math, re, time, random, shutil, hashlib, warnings, subprocess, threading, queue
import unicodedata
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce

warnings.filterwarnings("ignore")

def _pip_install(pkgs):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)
    if r.returncode != 0:
        print(f"  !! pip install thất bại cho {pkgs} (mã {r.returncode}).")
    return r.returncode == 0

try:
    import soundfile as sf
except Exception:
    _pip_install(["soundfile"]); import soundfile as sf
try:
    import librosa
except Exception:
    _pip_install(["librosa"]); import librosa

import numpy as np
import pandas as pd
import scipy.signal as sps
from scipy.stats import rankdata, kurtosis, spearmanr
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from IPython.display import display, Markdown

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_curve, average_precision_score, balanced_accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def amp_autocast(device, enabled):
    dev_type = torch.device(device).type
    try:
        return torch.amp.autocast(device_type=dev_type, enabled=bool(enabled))
    except Exception:
        return torch.cuda.amp.autocast(enabled=bool(enabled))

def make_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=bool(enabled))
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=bool(enabled))

print("Python      :", sys.version.split()[0])
print("NumPy       :", np.__version__, "| pandas", pd.__version__)
print("PyTorch     :", torch.__version__)
print("librosa     :", librosa.__version__)
N_GPU = torch.cuda.device_count()
USE_CUDA = N_GPU > 0
print("GPU count   :", N_GPU)
for i in range(N_GPU):
    p = torch.cuda.get_device_properties(i)
    print(f"  [GPU {i}] {p.name} | {p.total_memory/1024**3:.1f} GB")
''')

code(r'''
# =========================================================
# 1.2 Cấu hình trung tâm / Central configuration
#     SCENARIO chọn bộ giá trị mặc định (PRESETS) - mọi trường vẫn sửa được sau đó.
#     Miền 0 = nguồn (CÓ nhãn) | miền 1 = đích (nhãn CHỈ dùng để đánh giá)
# =========================================================
SCENARIO = "__SCENARIO__"     # "sim2real" (Sx mô phỏng -> Rx thực địa) | "pvc2hdpe" (PVC -> HDPE)
V7_FIELD = "/kaggle/input/datasets/nguyennguyenkhac/tapdulieutt-8k-v7-2/field_rec_8k_V7.2"   # dữ liệu hiện trường v7
MATERIAL_E = {"PVC": 3.0e9, "UPVC": 3.0e9, "HDPE": 0.9e9, "PE100": 1.0e9, "PE80": 0.8e9, "PP": 1.3e9}

@dataclass
class CFG:
    # ---- Dữ liệu hai miền ----------------------------------------------------------
    data_root: str = ""                    # thư mục chứa cả hai miền ("" = tự dò trong /kaggle/input)
    source_root: str = ""
    target_root: str = ""
    source_name: str = "Sx"
    target_name: str = "Rx"
    source_keys: tuple = ()
    target_keys: tuple = ()
    unnamed_dataset_domain: int = 1        # bộ leak/noleak KHÔNG mang tên miền thuộc miền nào (0/1)
    leak_keys: tuple = ("leak", "ro_ri", "rori")
    noleak_keys: tuple = ("noleak", "no_leak", "non_leak", "nonleak", "normal", "background",
                          "khong_ro_ri", "khongroli")
    unlabeled_keys: tuple = ("unlabeled", "unlabelled", "unknown", "chua_gan_nhan", "khong_nhan")
    audio_ext: tuple = (".wav", ".flac", ".ogg", ".mp3")
    record_regex_source: str = r"(R\d{1,4})"   # nhóm của miền nguồn (Sx: mỗi lần mô phỏng/kịch bản)
    record_regex_target: str = r"(R\d{1,4})"   # nhóm của miền đích (Rx: mỗi bản ghi hiện trường)
    target_group_csv: str = ""             # CSV 2 cột record,group (điểm khảo sát) -> chia fold đích theo NHÓM
    source_max_records: int = 0            # 0 = dùng hết; >0 = lấy mẫu phân tầng (Sx thường rất lớn)
    max_frames_per_record: int = 0         # 0 = không giới hạn
    strict_checks: bool = True
    # ---- Chế độ dự phòng ---------------------------------------------------------------
    target_fallback: str = "transform"     # chỉ có nguồn -> đích giả lập từ bản ghi nguồn TÁCH RỜI | "stop"
    source_fallback: str = "simulator"     # chỉ có đích -> sinh nguồn bằng bộ mô phỏng tích hợp | "stop"
    sim_target_frac: float = 0.5
    sim_snr_db: tuple = (0.0, 10.0)
    sim_transients: bool = True
    sim_eq: bool = True
    simulator_n_rec_per_class: int = 30
    hide_target_labels: bool = False       # True = kịch bản triển khai: miền đích hoàn toàn KHÔNG nhãn

    # ---- Tín hiệu & chọn dải -------------------------------------------------------------
    analysis_sr: int = 8000                # tần số lấy mẫu cho phân tích lệch miền (mục 4)
    band_mode: str = "auto"                # "auto" (chọn theo hồ sơ dải, không dùng nhãn đích) | "fixed"
    band_hz: tuple = (20.0, 2000.0)        # dải cố định / dải khởi tạo
    band_search_hz: tuple = (20.0, 3600.0)
    band_n: int = 18
    band_keep_frac: float = 0.5            # giữ các dải có điểm >= band_keep_frac * điểm cao nhất
    band_min_octaves: float = 1.5          # bề rộng tối thiểu (quãng tám) - chống dải quá hẹp khi ít bản ghi
    target_sr: int = 0                     # 0 = tự suy ra từ dải (>= 2.5 x cận trên, <= sr gốc nhỏ nhất)
    filter_order: int = 4
    notch_mains: bool = True               # lọc răng lược 50 Hz x mọi hài
    mains_hz: float = 50.0
    notch_q: float = 30.0
    frame_sec: float = 2.0
    hop_sec: float = 1.0
    eda_max_records: int = 40              # số bản ghi tối đa / (miền, lớp) cho phân tích mục 4

    # ---- VMD ---------------------------------------------------------------------------------
    use_vmd: bool = True
    vmd_K: int = 4
    vmd_alpha: float = 1500.0
    vmd_tau: float = 0.0
    vmd_iters: int = 150
    vmd_tol: float = 1e-6
    vmd_batch: int = 256                   # cho frame 4000 mẫu; tự co theo độ dài frame
    vmd_mains_tol_hz: float = 2.0

    # ---- Biểu diễn thời gian-tần số -------------------------------------------------------------
    tf_repr: str = "cwt"                   # "cwt" (Morlet, tần số log) | "logmel"
    n_freqs: int = 64
    tf_hop: int = 0                        # 0 = 10 ms
    cwt_w0: float = 6.0
    n_fft: int = 0                         # 0 = ~0,5 s (logmel)
    input_norm: str = "domain_cmvn"        # "domain_cmvn" | "frame"
    specaug_freq_mask: int = 6
    specaug_time_mask: int = 12

    # ---- Vật lý (Korteweg) --------------------------------------------------------------------
    c0: float = 1480.0
    K_water: float = 2.2e9
    source_material: str = "PVC"
    target_material: str = "PVC"
    sdr_source: float = 17.0
    sdr_target: float = 17.0

    # ---- DAug: tăng cường miền cho lô nguồn (phương pháp *_aug) ---------------------------------
    daug_p: float = 0.7
    daug_warp: str = "auto"                # "auto" (bật khi vật liệu khác hoặc r_emp != 1) | "on" | "off"
    daug_warp_range: tuple = (0.85, 1.15)
    daug_alpha: float = 4.6e-4             # Np/(m.Hz) suy hao nhớt đàn hồi
    daug_dist_m: tuple = (0.0, 15.0)
    daug_floor_p: float = 0.6              # nhiễu theo PHỔ NỀN miền đích (ước lượng không nhãn)
    daug_floor_snr_db: tuple = (-5.0, 10.0)
    daug_floor_pct: float = 20.0           # phân vị phổ dùng làm "phổ nền"
    daug_transient_p: float = 0.4          # xung va đập (xe cộ, bước chân, van)
    daug_transient_snr_db: tuple = (-5.0, 10.0)
    daug_eq_p: float = 0.5                 # đáp ứng kênh ngẫu nhiên (cảm biến/ghép nối)
    daug_eq_db: float = 6.0

    # ---- Mô hình ----------------------------------------------------------------------------------
    backbones: tuple = ("CNN2D", "CNN1D")
    main_backbone: str = "CNN2D"
    emb_dim: int = 128
    use_cbam: bool = True
    disc_hidden: int = 128
    dropout: float = 0.3

    # ---- Thích nghi miền -----------------------------------------------------------------------------
    methods: tuple = ("source_only", "adabn", "coral", "dann", "dann_coral", "cdan_coral", "dann_coral_aug",
                      "fewshot_scratch", "fewshot_src", "fewshot_da", "oracle")   # thêm "mmd" nếu muốn
    lambda_dann: float = 1.0
    grl_gamma: float = 10.0
    coral_weight: float = 1.0
    mmd_weight: float = 1.0
    ramp_alignment: bool = True
    cdan_entropy: bool = True              # CDAN+E: ưu tiên mẫu dự đoán chắc chắn
    final_method: str = "dann_coral"       # mô hình xuất - ĐĂNG KÝ TRƯỚC

    # ---- Huấn luyện ------------------------------------------------------------------------------------
    epochs: int = 30
    batch_size: int = 64                   # mỗi miền / bước
    eval_batch: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.1
    label_smoothing: float = 0.05
    grad_clip: float = 5.0
    amp: bool = True
    balanced_source_batches: bool = True   # mỗi lô nguồn 50/50 leak/noleak
    max_steps_per_epoch: int = 0           # 0 = không giới hạn
    min_steps_per_epoch: int = 4           # cho tập nhãn đích rất nhỏ (few-shot)
    aug_noise_p: float = 0.5
    aug_noise_snr_db: tuple = (10.0, 30.0)
    aug_shift_p: float = 0.5
    ft_epochs: int = 15
    ft_lr_early: float = 1e-4
    ft_lr_mid: float = 1e-5
    ft_lr_head: float = 1e-3
    fewshot_frac: float = 0.10
    fewshot_min_records_per_class: int = 1

    # ---- Chia dữ liệu, đánh giá, hạ tầng ---------------------------------------------------------------
    n_target_folds: int = 3
    source_val_frac: float = 0.2
    seed: int = 42
    seeds: tuple = (42,)
    decision_threshold: float = 0.5
    bootstrap_n: int = 2000
    tsne_max_points: int = 2000
    parallel_jobs: bool = True             # song song nhiều GPU (dropout dùng RNG toàn cục -> không bit-exact)
    max_parallel: int = 2
    gpu_resident_max_gb: float = 4.0
    verbose_epochs: bool = False
    train_final_model: bool = True
    out_dir: str = "/kaggle/working"
    fig_dpi: int = 300
    smoke_test: bool = False

PRESETS = {
    "sim2real": dict(
        source_name="Sx", target_name="Rx",
        source_keys=("sx", "sim", "simulated", "simulation", "mo_phong", "lab", "testbed"),
        target_keys=("rx", "real", "field", "hien_truong", "thuc_te", "thuc_dia"),
        unnamed_dataset_domain=1, source_root="", target_root=V7_FIELD,
        source_material="PVC", target_material="PVC",
        band_mode="auto", band_hz=(20.0, 2000.0), band_search_hz=(20.0, 3600.0), target_sr=0,
        daug_warp="auto", daug_warp_range=(0.85, 1.15), daug_dist_m=(0.0, 15.0),
        daug_floor_p=0.6, daug_floor_snr_db=(-5.0, 10.0), daug_transient_p=0.4, daug_eq_p=0.5,
        target_fallback="transform", source_fallback="simulator",
        sim_snr_db=(0.0, 10.0), sim_transients=True, sim_eq=True, max_steps_per_epoch=300),
    "pvc2hdpe": dict(
        source_name="PVC", target_name="HDPE",
        source_keys=("pvc", "upvc", "u_pvc"),
        target_keys=("hdpe", "pe100", "pe80", "pe_100", "pe_80", "polyethylene"),
        unnamed_dataset_domain=0, source_root=V7_FIELD, target_root="",
        source_material="PVC", target_material="HDPE",
        band_mode="fixed", band_hz=(15.0, 350.0), band_search_hz=(15.0, 1000.0), target_sr=2000,
        daug_warp="on", daug_warp_range=(0.50, 0.65), daug_dist_m=(5.0, 25.0),
        daug_floor_p=0.5, daug_floor_snr_db=(-5.0, 5.0), daug_transient_p=0.0, daug_eq_p=0.0,
        target_fallback="transform", source_fallback="stop",
        sim_snr_db=(5.0, 15.0), sim_transients=False, sim_eq=False, max_steps_per_epoch=0),
}
cfg = CFG()
for _k, _v in PRESETS[SCENARIO].items():
    setattr(cfg, _k, _v)

# ---- Ghi đè bằng biến môi trường (chạy ngoài Kaggle / smoke test) ----
if os.environ.get("LEAK_SMOKE") == "1":
    cfg.smoke_test = True
if os.environ.get("LEAK_DATA") is not None:          # thư mục chứa cả hai miền ("" = tự dò)
    cfg.data_root, cfg.source_root, cfg.target_root = os.environ["LEAK_DATA"], "", ""
if os.environ.get("DA_SOURCE") is not None:
    cfg.source_root = os.environ["DA_SOURCE"]
if os.environ.get("DA_TARGET") is not None:
    cfg.target_root = os.environ["DA_TARGET"]
if os.environ.get("DA_HIDE_TARGET_LABELS") == "1":
    cfg.hide_target_labels = True
if os.environ.get("LEAK_OUT"):
    cfg.out_dir = os.environ["LEAK_OUT"]
elif not Path("/kaggle").exists():
    cfg.out_dir = str(Path("./da_outputs").resolve())

if cfg.smoke_test:
    print("!! SMOKE TEST: 3 epoch, 2 fold đích, VMD 30 vòng - CHỈ để kiểm tra pipeline.")
    cfg.epochs, cfg.ft_epochs, cfg.n_target_folds = 3, 2, 2
    cfg.batch_size, cfg.eval_batch = 32, 128
    cfg.vmd_iters, cfg.bootstrap_n, cfg.tsne_max_points = 30, 200, 400
    cfg.seeds, cfg.fig_dpi = (cfg.seed,), 110
    cfg.analysis_sr, cfg.simulator_n_rec_per_class = 4000, 6
    cfg.band_search_hz = (cfg.band_search_hz[0], min(cfg.band_search_hz[1], 1500.0))

if os.environ.get("DA_METHODS"):          # ví dụ DA_METHODS=source_only,dann_coral,oracle (chạy nhanh)
    cfg.methods = tuple(m.strip() for m in os.environ["DA_METHODS"].split(",") if m.strip())
if os.environ.get("DA_BACKBONES"):        # ví dụ DA_BACKBONES=CNN1D
    cfg.backbones = tuple(b.strip() for b in os.environ["DA_BACKBONES"].split(",") if b.strip())
    if cfg.main_backbone not in cfg.backbones:
        cfg.main_backbone = cfg.backbones[0]

# Tên miền luôn là một từ khoá của chính miền đó
cfg.source_keys = tuple(dict.fromkeys(tuple(cfg.source_keys) + (cfg.source_name,)))
cfg.target_keys = tuple(dict.fromkeys(tuple(cfg.target_keys) + (cfg.target_name,)))

# ---- Đặc tả các phương pháp ----
#   kind="train": huấn luyện đầy đủ | kind="post": dựng trên mô hình nền (base)
#   src/tgt: dùng nguồn có nhãn / đích KHÔNG nhãn khi huấn luyện; label_dom/label_set: tập có nhãn
def _mi(kind="train", src=True, tgt=False, dann=False, cdan=False, coral=False, mmd=False, daug=False,
        label_dom=0, label_set=None, base=None, finetune=False, needs_target_labels=False):
    return dict(kind=kind, src=src, tgt=tgt, dann=dann, cdan=cdan, coral=coral, mmd=mmd, daug=daug,
                label_dom=label_dom, label_set=label_set, base=base, finetune=finetune,
                needs_target_labels=needs_target_labels)

METHOD_INFO = {
    "source_only":     _mi(),
    "coral":           _mi(tgt=True, coral=True),
    "mmd":             _mi(tgt=True, mmd=True),
    "dann":            _mi(tgt=True, dann=True),
    "dann_coral":      _mi(tgt=True, dann=True, coral=True),
    "cdan_coral":      _mi(tgt=True, cdan=True, coral=True),
    "dann_coral_aug":  _mi(tgt=True, dann=True, coral=True, daug=True),
    "fewshot_scratch": _mi(src=False, label_dom=1, label_set="fewshot", needs_target_labels=True),
    "oracle":          _mi(src=False, label_dom=1, label_set="adapt", needs_target_labels=True),
    "adabn":           _mi(kind="post", base="source_only"),
    "fewshot_src":     _mi(kind="post", base="source_only", finetune=True, needs_target_labels=True),
    "fewshot_da":      _mi(kind="post", base="dann_coral", finetune=True, needs_target_labels=True),
}
S_, T_ = cfg.source_name, cfg.target_name
METHOD_LABEL = {"source_only": "Source-only", "adabn": "AdaBN", "coral": "Deep CORAL", "mmd": "Deep MMD",
                "dann": "DANN (GRL)", "dann_coral": "DANN + CORAL", "cdan_coral": "CDAN+E + CORAL",
                "dann_coral_aug": "DANN + CORAL + DAug",
                "fewshot_scratch": f"Few-shot chỉ {T_} (không {S_})",
                "fewshot_src": f"Few-shot FT (từ Source-only)", "fewshot_da": "Few-shot FT (từ DANN+CORAL)",
                "oracle": f"Oracle ({T_} có nhãn)"}
UDA_METHODS = ("adabn", "coral", "mmd", "dann", "dann_coral", "cdan_coral", "dann_coral_aug")

_unknown = [m for m in cfg.methods if m not in METHOD_INFO]
assert not _unknown, f"Phương pháp không xác định: {_unknown}"
_m = list(cfg.methods)
for m in list(_m):
    base = METHOD_INFO[m].get("base")
    if base and base not in _m:
        _m.insert(0, base)
        print(f"  [cfg] '{m}' cần '{base}' -> tự thêm '{base}' vào cfg.methods")
cfg.methods = tuple(dict.fromkeys(_m))
TRAIN_METHODS = [m for m in cfg.methods if METHOD_INFO[m]["kind"] == "train"]
POST_METHODS = [m for m in cfg.methods if METHOD_INFO[m]["kind"] == "post"]
assert METHOD_INFO[cfg.final_method]["kind"] == "train" and METHOD_INFO[cfg.final_method]["src"]
assert cfg.main_backbone in cfg.backbones

OUT = Path(cfg.out_dir)
FIG_DIR, TAB_DIR = OUT / "figures", OUT / "tables"
CACHE_DIR, EXPORT_DIR = OUT / "cache", OUT / "export"
for _d in (FIG_DIR, TAB_DIR, CACHE_DIR, EXPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Tham số tín hiệu suy ra (gọi lại sau khi chọn dải ở mục 4.2) ----
_AUTO = dict(sr=(cfg.target_sr == 0), hop=(cfg.tf_hop == 0), nfft=(cfg.n_fft == 0))
def finalize_signal_config(min_native_sr=None, verbose=True):
    global FRAME_LEN, HOP_LEN
    lo, hi = cfg.band_hz
    if _AUTO["sr"]:
        need = 2.2 * hi                                  # Nyquist + ~10 % dự trữ cho bộ lọc
        sr = next((c for c in (1000, 2000, 4000, 8000, 16000, 32000, 48000) if c >= need), 48000)
        if min_native_sr:
            sr = min(sr, int(min_native_sr))
        cfg.target_sr = int(sr)
    if hi > 0.45 * cfg.target_sr:
        print(f"  !! cận trên dải {hi:.0f} Hz vượt 0,45 x {cfg.target_sr} Hz -> hạ về {0.45 * cfg.target_sr:.0f} Hz")
        cfg.band_hz = (lo, 0.45 * cfg.target_sr)
    if _AUTO["hop"]:
        cfg.tf_hop = max(1, int(round(0.01 * cfg.target_sr)))
    if _AUTO["nfft"]:
        cfg.n_fft = int(2 ** round(math.log2(0.5 * cfg.target_sr)))
    FRAME_LEN = int(round(cfg.frame_sec * cfg.target_sr))
    HOP_LEN = int(round(cfg.hop_sec * cfg.target_sr))
    if verbose:
        print(f"Tín hiệu: dải {cfg.band_hz[0]:.0f}-{cfg.band_hz[1]:.0f} Hz | target_sr {cfg.target_sr} Hz | "
              f"frame {FRAME_LEN} mẫu, hop {HOP_LEN} | tf_hop {cfg.tf_hop} | n_fft {cfg.n_fft}")

print(f"KỊCH BẢN: {SCENARIO}  ({cfg.source_name} -> {cfg.target_name})")
print(json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
                 indent=1, ensure_ascii=False))
finalize_signal_config()
print(f"Phương pháp huấn luyện: {TRAIN_METHODS}\nPhương pháp dựng sau   : {POST_METHODS}")
print(f"Thư mục ra: {OUT}")
''')

code(r'''
# =========================================================
# 1.3 Tái lập & phong cách hình vẽ / Reproducibility and plot style
# =========================================================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed % (2 ** 32)); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def make_generator(device, seed):
    g = torch.Generator(device=torch.device(device))
    g.manual_seed(int(seed) % (2 ** 63))
    return g

set_seed(cfg.seed)
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": cfg.fig_dpi, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white",
})
DOM_NAMES = {0: cfg.source_name, 1: cfg.target_name}
PAL = {"src": "#2E6F95", "tgt": "#C8553D", "acc": "#3F7D20", "warn": "#E0A458", "aux": "#8C6BB1", "grey": "#6E7B8B"}
GROUP_STYLE = {
    (0, 1): dict(label=f"{cfg.source_name} Leak", color="#B2182B", marker="o"),
    (0, 0): dict(label=f"{cfg.source_name} Safe", color="#2166AC", marker="o"),
    (1, 1): dict(label=f"{cfg.target_name} Leak", color="#E08214", marker="^"),
    (1, 0): dict(label=f"{cfg.target_name} Safe", color="#35978F", marker="^"),
    (1, -1): dict(label=f"{cfg.target_name} (không nhãn)", color="#8C8C8C", marker="^"),
}
METHOD_COLORS = {"source_only": "#6E7B8B", "adabn": "#A6761D", "coral": "#1B9E77", "mmd": "#66A61E",
                 "dann": "#7570B3", "dann_coral": "#D95F02", "cdan_coral": "#E6AB02", "dann_coral_aug": "#E7298A",
                 "fewshot_scratch": "#A6CEE3", "fewshot_src": "#8DA0CB", "fewshot_da": "#FC8D62", "oracle": "#222222"}

FIG_INDEX = {}
def savefig(fig, name, caption=""):
    n = len(FIG_INDEX) + 1
    stem = f"Fig{n:02d}_{name}"
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=cfg.fig_dpi, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    FIG_INDEX[stem] = caption
    plt.show()
    if caption:
        display(Markdown(f"**Figure {n}.** {caption}"))
    plt.close(fig)
    return stem

print("Device:", DEVICE, "| seed:", cfg.seed)
''')

# =============================================================================
# 2. Dữ liệu
# =============================================================================
md(r'''
## 2. Dữ liệu hai miền / Two-domain data discovery and inventory

Nhãn: `leak` = **1**, `noleak` / `normal` / `background` = **0**, `unlabeled` = **−1** (chỉ miền đích).
Mã bản ghi được **gắn tiền tố miền** (`«S»:R001`, `«T»:R001`) và trích bằng regex **riêng của từng miền**.
Bản ghi (hoặc nhóm, nếu có `target_group_csv`) là đơn vị chia dữ liệu và đơn vị thống kê.
''')

code(r'''
# =========================================================
# 2.1 Dò thư mục hai miền / Locate the source and target roots
# =========================================================
def _norm_token(s):
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.lower().replace("-", "_").replace(" ", "_")

def _key_regex(keys):
    ks = sorted({_norm_token(k) for k in keys}, key=len, reverse=True)
    return re.compile(r"(?:^|[_.])(?:" + "|".join(map(re.escape, ks)) + r")(?=$|[_.\d])")

_RX = dict(noleak=_key_regex(cfg.noleak_keys), leak=_key_regex(cfg.leak_keys),
           unlab=_key_regex(cfg.unlabeled_keys),
           src=_key_regex(cfg.source_keys), tgt=_key_regex(cfg.target_keys))

def label_of_token(s):
    s = _norm_token(s)
    if _RX["noleak"].search(s):
        return 0
    if _RX["leak"].search(s):
        return 1
    if _RX["unlab"].search(s):
        return -1
    return None

def domain_of_token(s):
    s = _norm_token(s)
    a, b = bool(_RX["src"].search(s)), bool(_RX["tgt"].search(s))
    return 0 if (a and not b) else 1 if (b and not a) else None

for _t, _want in [("leak", 1), ("noleak", 0), ("No-Leak", 0), ("R001_leak01", 1), ("R12_noleak_03", 0),
                  ("rò_rỉ", 1), ("unlabeled", -1), ("leakage", None), ("field_rec_8k_V7.2", None)]:
    assert label_of_token(_t) == _want, f"bộ gán nhãn sai với '{_t}'"
assert domain_of_token(cfg.source_name) == 0 and domain_of_token(cfg.target_name) == 1
for _t in ("pipe", "leak", "data", "R001"):
    assert domain_of_token(_t) is None, f"từ '{_t}' không được mang nghĩa miền"
assert all(domain_of_token(k) != 1 for k in cfg.source_keys) and all(domain_of_token(k) != 0 for k in cfg.target_keys)

def _is_within(p, root):
    if p is None or root is None:
        return False
    p, root = Path(p).resolve(), Path(root).resolve()
    return p == root or root in p.parents

def audio_files_in(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in cfg.audio_ext)

def _has_audio(d):
    return any(p.is_file() and p.suffix.lower() in cfg.audio_ext for p in Path(d).rglob("*"))

def find_domain_dirs(bases, dom, max_depth=6):
    out = []
    for base in bases:
        base = Path(base)
        if not base.exists():
            continue
        stack = [(base, 0)]
        while stack:
            d, depth = stack.pop()
            try:
                kids = sorted(k for k in d.iterdir() if k.is_dir())
            except Exception:
                continue
            for k in kids:
                if domain_of_token(k.name) == dom:
                    out.append(k)
                elif depth < max_depth:
                    stack.append((k, depth + 1))
    out = [d for d in dict.fromkeys(out) if _has_audio(d)]
    return sorted(out, key=lambda p: (len(p.parts), str(p)))

def dirs_with_leak_noleak(base, max_depth=6):
    out, base = [], Path(base)
    if not base.exists():
        return out
    stack = [(base, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            kids = [k for k in d.iterdir() if k.is_dir()]
        except Exception:
            continue
        if {0, 1}.issubset({label_of_token(k.name) for k in kids}):
            out.append(d)
        if depth < max_depth:
            stack += [(k, depth + 1) for k in kids]
    return sorted(out, key=lambda p: (-len(p.parts), str(p)))

def resolve_domain_roots():
    roots, how = {0: None, 1: None}, {}
    for dom, r in ((0, cfg.source_root), (1, cfg.target_root)):
        if r:
            if Path(r).exists():
                roots[dom], how[dom] = Path(r), "cfg"
            else:
                print(f"!! thư mục miền {DOM_NAMES[dom]} không tồn tại: {r} -> tự dò.")
    if cfg.data_root and not Path(cfg.data_root).exists():
        print(f"!! cfg.data_root không tồn tại: {cfg.data_root} -> tự dò trong /kaggle/input.")
    bases = ([Path(cfg.data_root)] if (cfg.data_root and Path(cfg.data_root).exists())
             else [Path("/kaggle/input"), Path("./data")])
    for dom in (1, 0):
        if roots[dom] is None:
            c = [d for d in find_domain_dirs(bases, dom) if not _is_within(d, roots[1 - dom])]
            if c:
                roots[dom], how[dom] = c[0], "tự dò theo tên thư mục"
                if len(c) > 1:
                    print(f"!! {len(c)} thư mục miền {DOM_NAMES[dom]}; dùng {c[0]}. Khác: {c[1:6]}")
    u = int(cfg.unnamed_dataset_domain)
    if roots[u] is None:              # bộ leak/noleak KHÔNG mang tên miền
        o = roots[1 - u]
        c = [d for b in bases for d in dirs_with_leak_noleak(b) if not (_is_within(d, o) or _is_within(o, d))]
        if c:
            roots[u], how[u] = c[0], f"bộ leak/noleak không tên miền -> coi là {DOM_NAMES[u]}"
    if roots[0] is not None and roots[1] is not None and (_is_within(roots[0], roots[1]) or _is_within(roots[1], roots[0])):
        raise RuntimeError(f"Thư mục hai miền lồng nhau: {roots[0]} / {roots[1]} - hãy chỉ định riêng từng miền.")
    return roots[0], roots[1], how

SRC_ROOT, TGT_ROOT, ROOT_HOW = resolve_domain_roots()
if SRC_ROOT is None and TGT_ROOT is None:
    DATA_MODE = "synthetic"
elif TGT_ROOT is None:
    if cfg.target_fallback != "transform":
        raise RuntimeError(f"Không tìm thấy dữ liệu {cfg.target_name} (cfg.target_fallback='stop').")
    DATA_MODE = "pseudo_target"
elif SRC_ROOT is None:
    if cfg.source_fallback != "simulator":
        raise RuntimeError(f"Không tìm thấy dữ liệu {cfg.source_name} (cfg.source_fallback='stop').")
    DATA_MODE = "simulated_source"
else:
    DATA_MODE = "real"

print(f"Nguồn ({cfg.source_name}) : {SRC_ROOT}  [{ROOT_HOW.get(0, '-')}]")
print(f"Đích  ({cfg.target_name}) : {TGT_ROOT}  [{ROOT_HOW.get(1, '-')}]")
print(f"CHẾ ĐỘ DỮ LIỆU : {DATA_MODE}")
if DATA_MODE == "pseudo_target":
    print(f"!! Chỉ có {cfg.source_name} -> {cfg.target_name} sẽ được GIẢ LẬP từ bản ghi {cfg.source_name} tách rời.")
    print("   Mọi con số trên miền đích chỉ kiểm chứng PIPELINE.")
if DATA_MODE == "simulated_source":
    print(f"!! Chỉ có {cfg.target_name} -> {cfg.source_name} sẽ được SINH bằng bộ mô phỏng tích hợp (mục 2.2).")
    print("   Kết quả trả lời: 'bộ mô phỏng ĐƠN GIẢN này chuyển giao sang thực địa tốt đến đâu'.")
''')

code(r'''
# =========================================================
# 2.2 Bộ mô phỏng tích hợp / Built-in simulator (synthetic & simulated_source)
#     Mỗi miền có một "phong cách": Sx = mô phỏng sạch, dừng, một cảm biến;
#     Rx = thực địa: nền mạnh, xung, hài 50 Hz, bơm, EQ kênh theo điểm, tiếng rò gián đoạn, lệch lớp.
# =========================================================
SYN_STYLES = {
    "sim2real": {
        0: dict(f0=(350, 600), bw=(150, 300), bg_slope=1.0, bg_gain=0.4, res_gain=0.0, mains=0.0, pump=0.0,
                transients=(0, 0), leak_db=(0, 8), witness=(1.0, 1.0), warp=(1.0, 1.0), dist=(0.0, 0.0),
                eq_db=0.0, rec_factor=(1.0, 1.0)),
        1: dict(f0=(350, 600), bw=(150, 300), bg_slope=1.5, bg_gain=1.0, res_gain=0.6, mains=0.08, pump=0.2,
                transients=(2, 7), leak_db=(-4, 6), witness=(0.4, 0.85), warp=(0.85, 1.05), dist=(2.0, 15.0),
                eq_db=6.0, rec_factor=(1.0, 0.7)),
    },
    "pvc2hdpe": {
        0: dict(f0=(140, 220), bw=(50, 70), bg_slope=1.0, bg_gain=1.0, res_gain=0.5, mains=0.08, pump=0.15,
                transients=(2, 6), leak_db=(2, 10), witness=(0.7, 0.7), warp=(1.0, 1.0), dist=(0.0, 0.0),
                eq_db=0.0, rec_factor=(1.0, 1.0)),
        1: dict(f0=(140, 220), bw=(50, 70), bg_slope=1.6, bg_gain=1.0, res_gain=0.5, mains=0.08, pump=0.15,
                transients=(2, 6), leak_db=(-2, 6), witness=(0.7, 0.7), warp=(0.50, 0.65), dist=(5.0, 25.0),
                eq_db=0.0, rec_factor=(1.0, 1.0)),
    },
}

def _pink_np(rng, n, sr, slope=1.0):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / sr); f[0] = f[1]
    y = np.fft.irfft(X / f ** (slope / 2.0), n)
    return y / (np.std(y) + 1e-12)

def _random_eq_gain(rng, f, db, n_bumps=3):
    # Đường cong độ lợi trơn theo log-tần số: tổng vài "bướu" Gauss +/- db (đáp ứng cảm biến/ghép nối)
    lf = np.log(np.maximum(f, 1.0))
    g = np.zeros_like(f)
    for _ in range(n_bumps):
        c = rng.uniform(np.log(20.0), lf[-1]); s = rng.uniform(0.2, 0.8)
        g += rng.uniform(-db, db) * np.exp(-0.5 * ((lf - c) / s) ** 2)
    return 10 ** (g / 20.0)

def _transients_np(rng, n, sr, k):
    out = np.zeros(n)
    for _ in range(int(k)):
        L = int(sr * rng.uniform(0.1, 0.6)); s0 = int(rng.integers(0, max(1, n - L)))
        out[s0:s0 + L] += np.exp(-np.arange(L) / (rng.uniform(0.005, 0.05) * sr)) * rng.standard_normal(L)
    return out

def synthesise_domain(root, dom, style, n_rec_per_class, files_per_rec, dur, sr, rng, rid0=0):
    dname = DOM_NAMES[dom]; n = int(dur * sr); t = np.arange(n) / sr; fq = np.fft.rfftfreq(n, 1.0 / sr)
    rid = rid0
    for cls, fac in (("noleak", style["rec_factor"][0]), ("leak", style["rec_factor"][1])):
        (Path(root) / dname / cls).mkdir(parents=True, exist_ok=True)
        for _ in range(max(2, int(round(n_rec_per_class * fac)))):
            rid += 1
            warp, dist = rng.uniform(*style["warp"]), rng.uniform(*style["dist"])
            f0, bw = rng.uniform(*style["f0"]) * warp, rng.uniform(*style["bw"]) * warp
            lo, hi = max(10.0, f0 - bw), min(0.45 * sr, f0 + bw)
            gain_db, wit = rng.uniform(*style["leak_db"]), rng.uniform(*style["witness"])
            f_res, q_res = rng.uniform(40, 300) * warp, rng.uniform(4, 10)
            b_res, a_res = sps.iirpeak(f_res / (sr / 2), q_res)
            f_pump = rng.uniform(22, 45)
            eq = _random_eq_gain(rng, fq, style["eq_db"]) if style["eq_db"] > 0 else None
            for seg in range(1, files_per_rec + 1):
                x = style["bg_gain"] * _pink_np(rng, n, sr, style["bg_slope"])
                if style["res_gain"] > 0:            # cộng hưởng tĩnh của đoạn ống/cảm biến (cả hai lớp)
                    res = sps.lfilter(b_res, a_res, rng.standard_normal(n))
                    x = x + style["res_gain"] * res / (np.std(res) + 1e-12)
                for h in (1, 2, 3):
                    x += style["mains"] * rng.uniform(.5, 1.) * np.sin(2 * np.pi * cfg.mains_hz * h * t + rng.uniform(0, 6.28))
                x += style["pump"] * np.sin(2 * np.pi * f_pump * t + rng.uniform(0, 6.28))
                if cls == "leak":
                    sos = sps.butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band", output="sos")
                    lk = sps.sosfilt(sos, rng.standard_normal(n))
                    if dist > 0:
                        lk = np.fft.irfft(np.fft.rfft(lk) * np.exp(-cfg.daug_alpha * fq * dist), n)
                    lk /= (np.std(lk) + 1e-12)
                    if wit < 0.999:                  # tiếng rò gián đoạn (nhãn bản ghi là nhãn YẾU)
                        mask = np.zeros(n); k_int = int(rng.integers(2, 5))
                        for _ in range(k_int):
                            L = int(wit * n / k_int); s0 = int(rng.integers(0, max(1, n - L)))
                            mask[s0:s0 + L] = 1.0
                        win = np.hanning(int(.3 * sr)); mask = np.convolve(mask, win / win.sum(), "same")
                    else:
                        mask = 1.0
                    x = x + lk * mask * 10 ** (gain_db / 20.0)
                k_tr = int(rng.integers(style["transients"][0], style["transients"][1] + 1))
                if k_tr:
                    x = x + 3.0 * _transients_np(rng, n, sr, k_tr)
                if eq is not None:
                    x = np.fft.irfft(np.fft.rfft(x) * eq, n)
                x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
                sf.write(Path(root) / dname / cls / f"R{rid:03d}_seg{seg:02d}.wav", x.astype(np.float32), sr)
    return rid

_size = (dict(n_rec_per_class=6, files_per_rec=1, dur=12.0) if cfg.smoke_test
         else dict(n_rec_per_class=10, files_per_rec=2, dur=20.0))
if DATA_MODE == "synthetic":
    SYN_ROOT = OUT / "synthetic_da"
    shutil.rmtree(SYN_ROOT, ignore_errors=True)
    _rng = np.random.default_rng(cfg.seed)
    _r = synthesise_domain(SYN_ROOT, 0, SYN_STYLES[SCENARIO][0], sr=8000, rng=_rng, **_size)
    synthesise_domain(SYN_ROOT, 1, SYN_STYLES[SCENARIO][1], sr=8000, rng=_rng, rid0=_r, **_size)
    SRC_ROOT, TGT_ROOT = SYN_ROOT / cfg.source_name, SYN_ROOT / cfg.target_name
    ROOT_HOW = {0: "synthetic", 1: "synthetic"}
    print("!! SYNTHETIC DEMO MODE:", SYN_ROOT, "|", len(audio_files_in(SYN_ROOT)), "tệp")
elif DATA_MODE == "simulated_source":
    SIM_ROOT = OUT / "simulated_source"
    shutil.rmtree(SIM_ROOT, ignore_errors=True)
    _kw = dict(_size, n_rec_per_class=cfg.simulator_n_rec_per_class)
    synthesise_domain(SIM_ROOT, 0, SYN_STYLES[SCENARIO][0], sr=8000, rng=np.random.default_rng(cfg.seed), **_kw)
    SRC_ROOT = SIM_ROOT / cfg.source_name
    ROOT_HOW[0] = "bộ mô phỏng tích hợp"
    print(f"Sinh {cfg.source_name} mô phỏng: {len(audio_files_in(SRC_ROOT))} tệp -> {SRC_ROOT}")
''')

code(r'''
# =========================================================
# 2.3 Kiểm kê tệp + mã bản ghi (regex theo miền) + nhóm đích + miền giả lập (nếu cần)
# =========================================================
_REC_RE = {0: re.compile(cfg.record_regex_source, re.IGNORECASE), 1: re.compile(cfg.record_regex_target, re.IGNORECASE)}

def extract_record_id(path, dom):
    for s, how in ((path.stem, "regex:stem"), (path.parent.name, "regex:parent")):
        m = _REC_RE[dom].search(s)
        if m:
            return (m.group(1) if m.groups() else m.group(0)).upper(), how
    return path.stem, "fallback:stem"

def label_from_path(p, root):
    try:
        rel = p.relative_to(root)
    except ValueError:
        rel = Path(p.name)
    for s in [p.parent.name] + list(reversed(rel.parts)):
        lab = label_of_token(s)
        if lab is not None:
            return lab, _norm_token(s)
    return None, None

def scan_domain(root, dom, exclude=None):
    rows, skipped = [], []
    dname = DOM_NAMES[dom]
    for p in audio_files_in(root):
        if exclude is not None and _is_within(p, exclude):
            continue
        if any(domain_of_token(x) == 1 - dom for x in p.relative_to(root).parts):
            skipped.append((str(p), "mang tên miền còn lại")); continue
        lab, tok = label_from_path(p, root)
        if lab is None:
            if dom == 1:
                lab, tok = -1, "(không từ khoá) -> unlabeled"
            else:
                skipped.append((str(p), "không xác định được nhãn")); continue
        if dom == 0 and lab == -1:
            skipped.append((str(p), "nguồn không nhãn")); continue
        try:
            info = sf.info(str(p)); dur, sr0 = float(info.duration), int(info.samplerate)
        except Exception:
            try:
                dur, sr0 = float(librosa.get_duration(path=str(p))), int(librosa.get_samplerate(str(p)))
            except Exception as e:
                skipped.append((str(p), f"không đọc được ({type(e).__name__})")); continue
        rid, how = extract_record_id(p, dom)
        rows.append(dict(path=str(p), file_stem=p.stem, domain=dom, dom_name=dname, record_raw=rid,
                         record_id=f"{dname}:{rid}", rid_source=how, label=int(lab), label_token=tok,
                         duration_s=dur, sr_native=sr0))
    return rows, skipped

_rows, SKIPPED = scan_domain(SRC_ROOT, 0, exclude=TGT_ROOT if _is_within(TGT_ROOT, SRC_ROOT) else None)
if TGT_ROOT is not None:
    _r, _s = scan_domain(TGT_ROOT, 1, exclude=SRC_ROOT if _is_within(SRC_ROOT, TGT_ROOT) else None)
    _rows += _r; SKIPPED += _s
files = pd.DataFrame(_rows)
assert len(files) and set(files[files.domain == 0].label) == {0, 1}, \
    "Miền nguồn phải có tệp của CẢ HAI lớp leak/noleak"

# ---- Lấy mẫu phân tầng bản ghi nguồn (Sx mô phỏng thường rất lớn) ----
if cfg.source_max_records and files[files.domain == 0].record_id.nunique() > cfg.source_max_records:
    _rec0 = files[files.domain == 0].groupby("record_id").label.first()
    _rng = np.random.default_rng(cfg.seed + 7); _keep = []
    for lab in (0, 1):
        r = sorted(_rec0.index[_rec0 == lab])
        k = max(2, int(round(cfg.source_max_records * len(r) / len(_rec0))))
        _keep += list(_rng.choice(r, min(k, len(r)), replace=False))
    files = files[(files.domain == 1) | files.record_id.isin(_keep)].reset_index(drop=True)
    print(f"Lấy mẫu {len(_keep)} / {len(_rec0)} bản ghi {cfg.source_name} (cfg.source_max_records)")

# ---- Chế độ pseudo_target: biến đổi bản ghi nguồn TÁCH RỜI thành miền đích giả lập ----
SIM_PARAMS = {}
if DATA_MODE == "pseudo_target":
    _rec0 = files.groupby("record_id").label.first()
    _rng = np.random.default_rng(cfg.seed + 2024)
    _pick = []
    for lab in (0, 1):
        r = sorted(_rec0.index[_rec0 == lab])
        k = max(min(int(round(cfg.sim_target_frac * len(r))), len(r) - 2), 0)
        if k < 2:
            raise RuntimeError(f"Lớp {lab} chỉ có {len(r)} bản ghi nguồn - cần >= 4 để tách miền đích giả lập.")
        _pick += list(_rng.choice(r, size=k, replace=False))
    _msk = files.record_id.isin(_pick)
    files.loc[_msk, "domain"] = 1
    files.loc[_msk, "dom_name"] = cfg.target_name
    files.loc[_msk, "record_id"] = cfg.target_name + "sim:" + files.loc[_msk, "record_raw"]
    for rid in sorted(files.loc[_msk, "record_id"].unique()):
        rr = np.random.default_rng(int(hashlib.sha1(rid.encode()).hexdigest()[:8], 16) ^ cfg.seed)
        SIM_PARAMS[rid] = dict(warp=float(rr.uniform(*cfg.daug_warp_range)) if cfg.daug_warp == "on" else 1.0,
                               dist=float(rr.uniform(*cfg.daug_dist_m)), snr_db=float(rr.uniform(*cfg.sim_snr_db)),
                               eq=bool(cfg.sim_eq), transients=bool(cfg.sim_transients), seed=int(rr.integers(1 << 30)))
    print(f"pseudo_target: {len(SIM_PARAMS)} bản ghi {cfg.source_name} -> '{cfg.target_name}sim' "
          f"(warp {cfg.daug_warp}, x = {cfg.daug_dist_m} m, SNR = {cfg.sim_snr_db} dB, "
          f"EQ {cfg.sim_eq}, xung {cfg.sim_transients})")

if cfg.hide_target_labels:
    files.loc[files.domain == 1, "label"] = -1
    print(f"!! cfg.hide_target_labels=True -> mọi tệp {cfg.target_name} coi là KHÔNG nhãn.")

# ---- Nhóm của miền đích (điểm khảo sát / tuyến ống) ----
files["group"] = files.record_id
if cfg.target_group_csv:
    _g = pd.read_csv(cfg.target_group_csv)
    assert _g.shape[1] >= 2, "target_group_csv cần 2 cột: record, group"
    _gm = {str(a).strip().upper(): str(b).strip() for a, b in zip(_g.iloc[:, 0], _g.iloc[:, 1])}
    _t = files.domain == 1
    files.loc[_t, "group"] = [f"{cfg.target_name}:G:{_gm[r.upper()]}" if r.upper() in _gm
                              else rid for r, rid in zip(files.record_raw[_t], files.record_id[_t])]
    print(f"Nhóm đích: {files[_t].group.nunique()} nhóm cho {files[_t].record_id.nunique()} bản ghi "
          f"({sum(r.upper() in _gm for r in files.record_raw[_t].unique())} bản ghi có trong CSV)")

files["cls"] = files.label.map({1: "leak", 0: "noleak", -1: "unlabeled"})
files = files.sort_values(["domain", "cls", "record_id", "file_stem"]).reset_index(drop=True)
files.to_csv(OUT / "manifest_files.csv", index=False)

print(f"Số tệp: {len(files)} | bỏ qua {len(SKIPPED)}")
for p_, why in SKIPPED[:8]:
    print("   [bỏ qua]", Path(p_).name, "-", why)
print("Trích mã bản ghi:", files.groupby("dom_name").rid_source.value_counts().to_dict())
files.head(6)
''')

code(r'''
# =========================================================
# 2.4 Kiểm tra toàn vẹn + bảng mức BẢN GHI + tần số lấy mẫu gốc
# =========================================================
problems, notes = [], []
_ln = files.groupby("record_id").label.nunique()
if (_ln > 1).any():
    problems.append(f"{int((_ln > 1).sum())} bản ghi mang >1 nhãn: {list(_ln[_ln > 1].index[:8])}")
if files.path.duplicated().any():
    problems.append(f"{int(files.path.duplicated().sum())} đường dẫn xuất hiện 2 lần")

def _quick_hash(p):
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        h.update(fh.read(1 << 20))
    h.update(str(os.path.getsize(p)).encode())
    return h.hexdigest()

if DATA_MODE == "real":
    files["qhash"] = files.path.map(_quick_hash)
    _cross = set(files.qhash[files.domain == 0]) & set(files.qhash[files.domain == 1])
    if _cross:
        problems.append(f"{len(_cross)} tệp có NỘI DUNG trùng nhau giữa {cfg.source_name} và {cfg.target_name}")
for dom in (0, 1):
    _nfb = int(((files.rid_source == "fallback:stem") & (files.domain == dom)).sum())
    if _nfb:
        notes.append(f"{_nfb} tệp {DOM_NAMES[dom]} không khớp regex mã bản ghi -> mỗi tệp là một bản ghi "
                     f"(nếu nhiều tệp cắt từ cùng một lần ghi/mô phỏng, hãy sửa cfg.record_regex_"
                     f"{'source' if dom == 0 else 'target'})")

records = (files.groupby("record_id")
           .agg(domain=("domain", "first"), dom_name=("dom_name", "first"), label=("label", "first"),
                group=("group", "first"), n_files=("path", "size"), duration_s=("duration_s", "sum"),
                sr_native=("sr_native", "min"))
           .reset_index())
records["cls"] = records.label.map({1: "leak", 0: "noleak", -1: "unlabeled"})
records.to_csv(OUT / "manifest_records.csv", index=False)
REC_LABEL = dict(zip(records.record_id, records.label.astype(int)))
REC_DOM = dict(zip(records.record_id, records.domain.astype(int)))
REC_GROUP = dict(zip(records.record_id, records.group))

_src_counts = records[records.domain == 0].label.value_counts()
if min(_src_counts.get(0, 0), _src_counts.get(1, 0)) < 2:
    problems.append(f"Miền nguồn cần >= 2 bản ghi mỗi lớp, có {_src_counts.to_dict()}")
TGT_LAB_COUNTS = {lab: int(((records.domain == 1) & (records.label == lab)).sum()) for lab in (0, 1)}
N_TGT_UNLAB = int(((records.domain == 1) & (records.label < 0)).sum())
TARGET_HAS_LABELS = min(TGT_LAB_COUNTS.values()) >= 2
if (records.domain == 1).sum() == 0:
    problems.append("Không có bản ghi miền đích nào")
if not TARGET_HAS_LABELS:
    notes.append(f"Miền đích có nhãn: {TGT_LAB_COUNTS} (+{N_TGT_UNLAB} không nhãn) -> KHÔNG đủ để đánh giá "
                 f"(cần >= 2 bản ghi mỗi lớp). Notebook chỉ huấn luyện mô hình cuối + xếp hạng.")

# Tần số lấy mẫu gốc: dải phân tích/huấn luyện không được vượt Nyquist của miền có sr thấp nhất,
# nếu không thành phần tần cao CHỈ có ở một miền sẽ thành lối tắt để phân biệt miền.
SR_BY_DOM = files.groupby("domain").sr_native.agg(["min", "max"])
MIN_NATIVE_SR = int(files.sr_native.min())
if SR_BY_DOM["min"].nunique() > 1 or SR_BY_DOM["max"].nunique() > 1:
    notes.append(f"Tần số lấy mẫu gốc khác nhau giữa hai miền: "
                 f"{ {DOM_NAMES[d]: (int(r['min']), int(r['max'])) for d, r in SR_BY_DOM.iterrows()} } "
                 f"-> mọi dải bị giới hạn <= 0,45 x {MIN_NATIVE_SR} Hz")

for s in notes:
    print("  [lưu ý]", s)
if problems:
    for s in problems:
        print("  !! LỖI:", s)
    if cfg.strict_checks:
        raise RuntimeError("Dữ liệu có mâu thuẫn - xem danh sách LỖI ở trên (hoặc đặt cfg.strict_checks=False).")

summary = (records.groupby(["dom_name", "cls"])
           .agg(n_records=("record_id", "nunique"), n_groups=("group", "nunique"), n_files=("n_files", "sum"),
                total_min=("duration_s", lambda s: s.sum() / 60), mean_s=("duration_s", "mean"),
                sr_native=("sr_native", lambda s: ",".join(str(v) for v in sorted(set(int(v) for v in s)))))
           .round(2))
display(Markdown("### Table 1. Recording-level inventory per domain"))
display(summary)
summary.to_csv(TAB_DIR / "table1_inventory.csv")
print(f"Chế độ: {DATA_MODE} | bản ghi {cfg.source_name}: {int((records.domain == 0).sum())} | "
      f"bản ghi {cfg.target_name}: {int((records.domain == 1).sum())} "
      f"(có nhãn {TGT_LAB_COUNTS}, không nhãn {N_TGT_UNLAB}) | TARGET_HAS_LABELS = {TARGET_HAS_LABELS} | "
      f"sr gốc nhỏ nhất {MIN_NATIVE_SR} Hz")
''')

code(r'''
# =========================================================
# 2.5 Hình: tổng quan dữ liệu hai miền
# =========================================================
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.4))
_g = records.groupby(["domain", "label"]).agg(n=("record_id", "size"), m=("duration_s", lambda s: s.sum() / 60))
_lbl = [GROUP_STYLE[k]["label"] for k in _g.index]
_col = [GROUP_STYLE[k]["color"] for k in _g.index]
ax[0].bar(_lbl, _g.n.values, color=_col); ax[0].set_ylabel("số bản ghi"); ax[0].set_title("(a) Bản ghi theo miền × lớp")
ax[1].bar(_lbl, _g.m.values, color=_col); ax[1].set_ylabel("phút"); ax[1].set_title("(b) Thời lượng")
for a in ax:
    a.tick_params(axis="x", rotation=20)
savefig(fig, "inventory", f"Dữ liệu theo miền và lớp (chế độ `{DATA_MODE}`). Nhãn {cfg.target_name} chỉ dùng để đánh giá.")
''')

# =============================================================================
# 3. Vật lý & tiền xử lý
# =============================================================================
md({"sim2real": r'''
## 3. Vật lý và tiền xử lý / Physics and pre-processing

Với Sx → Rx, vật lý ống **không còn là nguồn lệch chính**. Korteweg chỉ cho biết có dịch tần hay không khi bộ mô phỏng
dùng **vật liệu / SDR khác** tuyến ống thực địa (`cfg.source_material`, `cfg.target_material`, `cfg.sdr_*`). Cùng vật liệu
thì $r = c_T/c_S = 1$ và tăng cường dịch tần chỉ được bật nếu dữ liệu cho thấy $r_{emp} \neq 1$ (mục 4.3).

Tiền xử lý dùng chung hai miền: tái lấy mẫu (chống chồng phổ) → khử DC → lọc dải (pha 0, dải chọn ở mục 4.2) →
lọc **răng lược 50 Hz × mọi hài** (một bộ lọc `iircomb`) → chuẩn hoá đỉnh theo tệp. Ở chế độ `pseudo_target`, bản ghi
miền đích còn được "hiện trường hoá": suy hao theo khoảng cách, **EQ kênh ngẫu nhiên**, **nhiễu nền** ở SNR thấp,
**xung va đập**.
''', "pvc2hdpe": r'''
## 3. Vật lý lệch miền và tiền xử lý / Physics of the shift and pre-processing

Korteweg: $c = c_0\,[1 + B D/(E e)]^{-1/2}$. Các tần số cộng hưởng $f_n = n c / 2L$ tỉ lệ thuận với $c$, nên phổ rò rỉ
bị **nhân với** $r = c_{HDPE}/c_{PVC}$ — trên trục tần số **log** đó là một phép **tịnh tiến** $\log r$; CNN 2D trên CWT
tần số log + gộp toàn cục vì thế có sẵn thiên hướng bất biến với lệch miền này.

Tiền xử lý: tái lấy mẫu 2 kHz → khử DC → lọc dải **15–350 Hz** (pha 0) → lọc răng lược 50 Hz × mọi hài → chuẩn hoá đỉnh
theo tệp. Ở chế độ `pseudo_target`, bản ghi đích được **dịch tần** (tái lấy mẫu với tốc độ giả định $f_s \cdot r$),
**suy hao** $e^{-\alpha f x}$ và thêm **nhiễu nền**.
'''})

code(r'''
# =========================================================
# 3.1 Vận tốc sóng theo Korteweg / Wave speed (Korteweg)
# =========================================================
def wave_speed(E, sdr, c0=None, K=None):
    c0 = cfg.c0 if c0 is None else c0
    K = cfg.K_water if K is None else K
    return c0 / np.sqrt(1.0 + K * sdr / E)

_mats = list(dict.fromkeys([cfg.source_material.upper(), cfg.target_material.upper(), "PVC", "HDPE"]))
_rows = [dict(material=m, E_GPa=MATERIAL_E[m] / 1e9, SDR=sdr, c_ms=round(float(wave_speed(MATERIAL_E[m], sdr)), 1))
         for m in _mats if m in MATERIAL_E for sdr in (11, 17, 21, 26)]
korteweg = pd.DataFrame(_rows).pivot_table(index=["material", "E_GPa"], columns="SDR", values="c_ms")
display(Markdown("### Table 2. Korteweg wave speed c (m/s) vs. Young modulus and SDR = D/e"))
display(korteweg)
korteweg.to_csv(TAB_DIR / "table2_korteweg.csv")

C_SRC = float(wave_speed(MATERIAL_E[cfg.source_material.upper()], cfg.sdr_source))
C_TGT = float(wave_speed(MATERIAL_E[cfg.target_material.upper()], cfg.sdr_target))
R_KORTEWEG = C_TGT / C_SRC
print(f"c_{cfg.source_name} ({cfg.source_material}, SDR {cfg.sdr_source:g}) = {C_SRC:6.1f} m/s")
print(f"c_{cfg.target_name} ({cfg.target_material}, SDR {cfg.sdr_target:g}) = {C_TGT:6.1f} m/s")
print(f"r = c_{cfg.target_name}/c_{cfg.source_name} = {R_KORTEWEG:.3f}  ->  "
      + ("KHÔNG kỳ vọng dịch tần do vật liệu." if abs(R_KORTEWEG - 1) < 0.05
         else f"phổ rò dự kiến nhân với {R_KORTEWEG:.2f} (chưa kể suy hao)."))
''')

code(r'''
# =========================================================
# 3.2 Nạp & tiền xử lý một tệp / Load and pre-process one file
# =========================================================
_FILT = {}
def band_sos(sr, band=None):
    lo, hi = band or cfg.band_hz
    hi = min(hi, 0.49 * sr)
    key = ("band", int(sr), round(lo, 3), round(hi, 3), int(cfg.filter_order))
    if key not in _FILT:
        _FILT[key] = sps.butter(cfg.filter_order, [lo / (sr / 2), hi / (sr / 2)], btype="band", output="sos")
    return _FILT[key]

def mains_filters(sr):
    # Lọc răng lược: triệt 50 Hz và MỌI hài bằng một bộ lọc (fs/50 phải nguyên); dự phòng = từng notch
    key = ("mains", int(sr), cfg.mains_hz, cfg.notch_q)
    if key not in _FILT:
        flt = None
        if float(sr / cfg.mains_hz).is_integer():
            try:
                flt = [sps.iircomb(cfg.mains_hz, cfg.notch_q, ftype="notch", fs=sr)]
            except Exception:
                flt = None
        if flt is None:
            flt = [sps.iirnotch(cfg.mains_hz * h, cfg.notch_q, fs=sr)
                   for h in range(1, 400) if cfg.mains_hz * h < sr / 2 - 5]
        _FILT[key] = flt
    return _FILT[key]

def apply_mains(x, sr):
    for b, a in mains_filters(sr):
        x = sps.filtfilt(b, a, x)
    return x

def noise_band(sr):
    lo, hi = cfg.band_search_hz if cfg.band_mode == "auto" else cfg.band_hz
    return (lo, min(hi, 0.45 * sr))

def viscoelastic_attenuation(x, sr, dist_m, alpha=None):
    alpha = cfg.daug_alpha if alpha is None else alpha
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-alpha * f * dist_m), len(x))

def _read_any(path):
    try:
        y, fs = sf.read(str(path), always_2d=True, dtype="float32")
        return y.mean(axis=1), int(fs)
    except Exception:
        y, fs = librosa.load(str(path), sr=None, mono=True)
        return y.astype(np.float32), int(fs)

def pseudo_target_transform(x, sr, sim):
    """'Hiện trường hoá' cố định theo bản ghi (chế độ pseudo_target): suy hao, EQ kênh, nhiễu nền, xung."""
    rng = np.random.default_rng(sim["seed"])
    if sim["dist"] > 0:
        x = viscoelastic_attenuation(x, sr, sim["dist"])
    if sim.get("eq"):
        x = np.fft.irfft(np.fft.rfft(x) * _random_eq_gain(rng, np.fft.rfftfreq(len(x), 1.0 / sr), cfg.daug_eq_db), len(x))
    sos = band_sos(sr, noise_band(sr))
    ps = float(np.mean(sps.sosfiltfilt(sos, x) ** 2))
    nz = sps.sosfiltfilt(sos, _pink_np(rng, len(x), sr, 1.0))
    x = x + nz * np.sqrt(ps / ((float(np.mean(nz ** 2)) + 1e-20) * 10 ** (sim["snr_db"] / 10.0)))
    if sim.get("transients"):
        tr = _transients_np(rng, len(x), sr, int(rng.integers(2, 7)) * max(1, int(len(x) / sr / 10)))
        snr = rng.uniform(*cfg.daug_transient_snr_db)
        x = x + tr * np.sqrt(ps / ((float(np.mean(tr ** 2)) + 1e-20) * 10 ** (snr / 10.0)))
    return x

LOAD_REPAIRED = []
def load_audio(path, sim=None, apply_band=True, sr=None, notch=True, normalize=True):
    """(sim: dịch tần r bằng tái lấy mẫu với tốc độ giả định fs*r) -> tái lấy mẫu về sr -> khử DC
       -> (sim: hiện trường hoá) -> lọc dải cfg.band_hz (pha 0) -> lọc răng lược 50 Hz -> chuẩn hoá đỉnh theo tệp."""
    sr = int(sr or cfg.target_sr)
    x, fs = _read_any(path)
    nbad = int((~np.isfinite(x)).sum())
    if nbad:
        LOAD_REPAIRED.append(dict(path=str(path), n_nonfinite=nbad))
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    warp = float(sim["warp"]) if sim else 1.0
    if abs(fs * warp - sr) > 1e-6:
        x = librosa.resample(np.ascontiguousarray(x, dtype=np.float32), orig_sr=float(fs) * warp, target_sr=float(sr))
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if sim:
        x = pseudo_target_transform(x, sr, sim)
    if apply_band:
        x = sps.sosfiltfilt(band_sos(sr), x)
    if notch and cfg.notch_mains:
        x = apply_mains(x, sr)
    if normalize:
        m = float(np.max(np.abs(x))) if len(x) else 0.0
        if m > 1e-12:
            x = x / m
    return x.astype(np.float32)

_t0 = time.time()
_r0 = files.iloc[0]
_probe = load_audio(_r0.path, sim=SIM_PARAMS.get(_r0.record_id))
print(f"Thử nạp 1 tệp: {len(_probe) / cfg.target_sr:.1f} s @ {cfg.target_sr} Hz trong {time.time() - _t0:.2f} s | "
      f"lọc 50 Hz: {'răng lược' if len(mains_filters(cfg.target_sr)) == 1 else str(len(mains_filters(cfg.target_sr))) + ' notch'}")
''')

# =============================================================================
# 4. Phân tích lệch miền & chọn dải
# =============================================================================
md(r'''
## 4. Phân tích lệch miền dựa trên dữ liệu và chọn dải tần chuyển giao / Data-driven shift analysis and band selection

Mọi phân tích ở mục này chạy ở `analysis_sr` (≤ sr gốc nhỏ nhất của hai miền) trên các bản ghi mẫu, và **không dùng
nhãn «T»** cho bất kỳ quyết định nào.

**4.1 Chỉ số chất lượng tín hiệu theo miền** (mức bản ghi, không nhãn): mức RMS (dBFS), tỉ lệ clipping, hệ số đỉnh,
độ nhọn (kurtosis — tính xung), độ phẳng phổ, độ nổi hài 50 Hz, hệ số biến thiên RMS ngắn hạn (tính **không dừng**),
tỉ lệ cửa sổ có xung, tần số trọng tâm, tỉ lệ năng lượng < 200 Hz. Mỗi chỉ số có **độ tách miền**
$2\,|AUC(d;\,\text{chỉ số})-0{,}5|$ (0 = hai miền như nhau, 1 = tách hoàn toàn) — chỉ ra *lệch ở đâu*.

**4.2 Hồ sơ dải chuyển giao.** Với mỗi dải hẹp $b$ (chia log), $e_b$ = năng lượng tương đối (dB) của dải trong bản ghi:

$$\text{điểm}_b=\underbrace{\max\!\big(0,\;2\,[AUC_{«S»}(y;\,e_b)-0{,}5]\big)}_{\text{rò làm TĂNG năng lượng dải (trên «S» có nhãn)}}\times\Big(1-\underbrace{\min_{c\in\{0,1\}}2\,\big|AUC\big(d;\,e_b \,\big|\, «S»_c\cup«T»\big)-0{,}5\big|}_{\text{tách miền, không nhãn «T», không lẫn với tỉ lệ lớp}}\Big)$$

- **Có hướng:** rò rỉ là năng lượng *cộng thêm*; năng lượng tương đối *giảm* ở các dải khác chỉ là hệ quả của phép chuẩn hoá
  theo tổng, nên không được tính là thông tin (nếu tính |AUC − 0,5| thì mọi dải ngoài dải rò đều "phân biệt" giả tạo).
- **Bền với tỉ lệ lớp:** «T» được so với *từng* lớp của «S» và lấy min — một dải chỉ bị coi là lệch khi phân phối «T» nằm
  ngoài cả hai lớp «S»; khác biệt tỉ lệ rò giữa hai miền không làm dải bị loại oan.

Ở `band_mode="auto"`, dải huấn luyện = bao của các dải có điểm ≥ `band_keep_frac` × điểm cao nhất (tối thiểu
`band_min_octaves` quãng tám), rồi `target_sr` được suy ra. Nếu «T» có nhãn, notebook vẽ thêm (nét đứt) sức phân biệt
trên «T» — **chỉ để đánh giá sau** mức trùng khớp khái niệm giữa hai miền, không dùng để chọn.
''')

code(r'''
# =========================================================
# 4.1 Mô tả tín hiệu + phổ mức bản ghi (một lượt đọc) ở analysis_sr
# =========================================================
A_SR = int(min(cfg.analysis_sr, MIN_NATIVE_SR))
SEARCH = (float(cfg.band_search_hz[0]), float(min(cfg.band_search_hz[1], 0.45 * A_SR)))
NPERSEG = int(2 ** round(math.log2(A_SR / 2)))              # phân giải ~2 Hz
print(f"Phân tích ở {A_SR} Hz | vùng tìm dải {SEARCH[0]:.0f}-{SEARCH[1]:.0f} Hz | Welch nperseg {NPERSEG}")

def _mains_excess_db(f, P, top=500.0):
    out = []
    for h in range(1, 40):
        fm = cfg.mains_hz * h
        if fm > min(top, SEARCH[1]):
            break
        pk = P[np.abs(f - fm) <= 1.0]; ref = P[(np.abs(f - fm) >= 3) & (np.abs(f - fm) <= 10)]
        if len(pk) and len(ref):
            out.append(10 * np.log10(pk.max() / (np.median(ref) + 1e-30) + 1e-30))
    return float(np.mean(out)) if out else np.nan

def analyse_file(path, sim):
    x0, _ = _read_any(path)
    clip = float(np.mean(np.abs(x0) >= 0.999)); lvl = 20 * np.log10(np.sqrt(np.mean(x0 ** 2)) + 1e-12)
    xr = load_audio(path, sim=sim, apply_band=False, notch=False, normalize=False, sr=A_SR)
    if len(xr) < NPERSEG:
        return None, None
    xn = apply_mains(xr.astype(np.float64), A_SR) if cfg.notch_mains else xr.astype(np.float64)
    xb = sps.sosfiltfilt(band_sos(A_SR, SEARCH), xn)
    rms = np.sqrt(np.mean(xb ** 2)) + 1e-12
    w = max(1, int(0.25 * A_SR)); e = np.sqrt(np.mean(xb[: len(xb) // w * w].reshape(-1, w) ** 2, 1))
    w2 = max(1, int(0.05 * A_SR)); e2 = np.mean(xb[: len(xb) // w2 * w2].reshape(-1, w2) ** 2, 1)
    f, P = sps.welch(xn, fs=A_SR, nperseg=NPERSEG, noverlap=NPERSEG // 2)
    _, Pr = sps.welch(xr, fs=A_SR, nperseg=NPERSEG, noverlap=NPERSEG // 2)
    inS = (f >= SEARCH[0]) & (f <= SEARCH[1])
    d = dict(level_dbfs=lvl, clip_frac=clip, crest_db=20 * np.log10(np.max(np.abs(xb)) / rms),
             kurtosis=float(kurtosis(xb)), flatness=float(np.exp(np.mean(np.log(P[inS] + 1e-30))) / (np.mean(P[inS]) + 1e-30)),
             mains_db=_mains_excess_db(f, Pr), rms_cv=float(np.std(e) / (np.mean(e) + 1e-12)),
             transient_rate=float(np.mean(e2 > 4 * np.median(e2))) if len(e2) else np.nan,
             centroid_hz=float((f[inS] * P[inS]).sum() / (P[inS].sum() + 1e-30)),
             low200_frac=float(P[inS & (f < 200)].sum() / (P[inS].sum() + 1e-30)))
    return d, (f, P)

_rng = np.random.default_rng(cfg.seed)
eda_recs = []
for _, g in records.groupby(["domain", "label"]):
    r = g.record_id.values
    eda_recs += list(_rng.choice(r, cfg.eda_max_records, replace=False) if len(r) > cfg.eda_max_records else r)

SPEC, DESC, F_W = {}, {}, None
_t0 = time.time()
for rid in eda_recs:
    ds, Ps = [], []
    for p in files.path[files.record_id == rid].head(4):
        d, fp = analyse_file(p, SIM_PARAMS.get(rid))
        if d is not None:
            ds.append(d); Ps.append(fp[1]); F_W = fp[0]
    if ds:
        DESC[rid] = {k: float(np.nanmean([d[k] for d in ds])) for k in ds[0]}
        SPEC[rid] = np.mean(Ps, axis=0)
print(f"Đã phân tích {len(SPEC)} bản ghi trong {time.time() - _t0:.1f} s")

desc_df = pd.DataFrame.from_dict(DESC, orient="index")
desc_df["domain"] = [REC_DOM[r] for r in desc_df.index]; desc_df["label"] = [REC_LABEL[r] for r in desc_df.index]

def fast_auc(y, p):
    y = np.asarray(y) > 0.5; p = np.asarray(p, dtype=float)
    ok = np.isfinite(p); y, p = y[ok], p[ok]
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

def sep(y, v):
    a = fast_auc(y, v)
    return 2 * abs(a - 0.5) if np.isfinite(a) else np.nan

DESC_COLS = ["level_dbfs", "clip_frac", "crest_db", "kurtosis", "flatness", "mains_db", "rms_cv",
             "transient_rate", "centroid_hz", "low200_frac"]
qual = pd.DataFrame([dict(descriptor=c, **{f"median_{DOM_NAMES[d]}": float(desc_df[c][desc_df.domain == d].median())
                                           for d in (0, 1)},
                          domain_sep=sep(desc_df.domain, desc_df[c]),
                          direction=("cao hơn ở " + DOM_NAMES[1]) if desc_df[c][desc_df.domain == 1].median()
                          > desc_df[c][desc_df.domain == 0].median() else ("cao hơn ở " + DOM_NAMES[0]))
                     for c in DESC_COLS]).sort_values("domain_sep", ascending=False).reset_index(drop=True)
display(Markdown("### Table 3. Signal-quality descriptors per domain (record-level medians) and domain separability"))
display(qual.round(3))
qual.to_csv(TAB_DIR / "table3_quality_shift.csv", index=False)
if (desc_df.clip_frac > 0.001).any():
    print(f"  [lưu ý] {int((desc_df.clip_frac > 0.001).sum())} bản ghi có >0,1% mẫu bị clipping "
          f"(theo miền: {desc_df[desc_df.clip_frac > 0.001].groupby('domain').size().rename(DOM_NAMES).to_dict()})")
''')

code(r'''
# =========================================================
# 4.2 Hồ sơ dải chuyển giao + chọn dải tự động (KHÔNG dùng nhãn miền đích)
# =========================================================
EDGES = np.geomspace(SEARCH[0], SEARCH[1], cfg.band_n + 1)
BAND_C = np.sqrt(EDGES[:-1] * EDGES[1:])
_inS = (F_W >= EDGES[0]) & (F_W < EDGES[-1])

def band_rel_db(P):
    tot = P[_inS].sum() + 1e-30
    return np.array([10 * np.log10(P[(F_W >= a) & (F_W < b)].sum() / tot + 1e-12) for a, b in zip(EDGES[:-1], EDGES[1:])])

def gain_disc(y, v):
    # Rò rỉ là năng lượng CỘNG THÊM: dải chỉ mang thông tin nếu bản ghi leak có năng lượng tương đối CAO hơn.
    # (Năng lượng tương đối GIẢM ở các dải khác chỉ là hệ quả của phép chuẩn hoá theo tổng -> không tính.)
    a = fast_auc(y, v)
    return max(0.0, 2 * (a - 0.5)) if np.isfinite(a) else np.nan

_rids = list(SPEC)
BR_DB = np.stack([band_rel_db(SPEC[r]) for r in _rids])
_dom = np.array([REC_DOM[r] for r in _rids]); _lab = np.array([REC_LABEL[r] for r in _rids])
_s = _dom == 0
src_disc = np.array([gain_disc(_lab[_s], BR_DB[_s, b]) for b in range(cfg.band_n)])
# Tách miền KHÔNG lẫn với khác biệt tỉ lệ lớp: so «T» (không nhãn) với TỪNG lớp của nguồn, lấy min.
# Dải chỉ bị coi là "lệch" khi phân phối đích nằm ngoài CẢ HAI lớp nguồn.
_tt = _dom == 1
dom_sep = np.array([np.nanmin([sep(np.r_[np.zeros((_s & (_lab == c)).sum()), np.ones(_tt.sum())],
                                   np.r_[BR_DB[_s & (_lab == c), b], BR_DB[_tt, b]]) for c in (0, 1)])
                    for b in range(cfg.band_n)])
score = np.nan_to_num(src_disc) * (1 - np.nan_to_num(dom_sep))
_t = (_dom == 1) & (_lab >= 0)
tgt_disc = (np.array([gain_disc(_lab[_t], BR_DB[_t, b]) for b in range(cfg.band_n)])
            if TARGET_HAS_LABELS and len(np.unique(_lab[_t])) == 2 else None)

BAND_INFO = dict(mode=cfg.band_mode, search=SEARCH, initial=tuple(cfg.band_hz))
if cfg.band_mode == "auto":
    if np.nanmax(src_disc) < 0.1:
        print(f"!! Không dải nào phân biệt rõ leak/noleak trên {cfg.source_name} (max {np.nanmax(src_disc):.2f}) "
              f"-> giữ dải khởi tạo {cfg.band_hz}")
    else:
        keep = score >= cfg.band_keep_frac * score.max()
        lo, hi = float(EDGES[:-1][keep].min()), float(EDGES[1:][keep].max())
        while math.log2(hi / lo) < cfg.band_min_octaves and (lo > SEARCH[0] or hi < SEARCH[1]):
            lo, hi = max(SEARCH[0], lo / 2 ** 0.25), min(SEARCH[1], hi * 2 ** 0.25)
        cfg.band_hz = (float(round(lo, 1)), float(round(hi, 1)))
        BAND_INFO.update(kept_bands=[(round(float(a), 1), round(float(b), 1)) for a, b, k
                                     in zip(EDGES[:-1], EDGES[1:], keep) if k])
finalize_signal_config(MIN_NATIVE_SR)
BAND_INFO.update(selected=tuple(cfg.band_hz), target_sr=cfg.target_sr)
BAND_RHO = (float(spearmanr(src_disc, tgt_disc, nan_policy="omit")[0]) if tgt_disc is not None else np.nan)
BAND_INFO["rho_src_tgt_disc"] = BAND_RHO

band_tab = pd.DataFrame(dict(lo_hz=EDGES[:-1].round(1), hi_hz=EDGES[1:].round(1), src_disc=src_disc.round(3),
                             dom_sep=dom_sep.round(3), score=score.round(3),
                             in_band=(EDGES[1:] > cfg.band_hz[0]) & (EDGES[:-1] < cfg.band_hz[1])))
if tgt_disc is not None:
    band_tab["tgt_disc_EVAL_ONLY"] = tgt_disc.round(3)
band_tab.to_csv(TAB_DIR / "table4_band_transferability.csv", index=False)
print(f"Dải {'CHỌN TỰ ĐỘNG' if cfg.band_mode == 'auto' else 'cố định'}: {cfg.band_hz[0]:.0f}-{cfg.band_hz[1]:.0f} Hz "
      f"| target_sr {cfg.target_sr} Hz")
if tgt_disc is not None:
    print(f"Trùng khớp sức phân biệt theo dải {cfg.source_name} vs {cfg.target_name} (Spearman, CHỈ để đánh giá): "
          f"rho = {BAND_RHO:.2f}" + (("  <- thấp: phù hợp với dịch tần do vật liệu (r Korteweg = %.2f), "
                                       "chưa chắc là concept shift" % R_KORTEWEG) if (BAND_RHO < 0.3 and abs(R_KORTEWEG - 1) > 0.05)
                                      else "  <- thấp: dấu hiệu CONCEPT SHIFT (cơ chế rò ở đích khác nguồn)" if BAND_RHO < 0.3 else ""))
''')

code(r'''
# =========================================================
# 4.3 Phổ theo miền × lớp, phổ nền, tỉ số dịch tần r_emp -> quyết định dịch tần của DAug
# =========================================================
_lo, _hi = cfg.band_hz
INB = (F_W >= _lo) & (F_W <= _hi)
SPEC_DB = {}
for rid, P in SPEC.items():
    d = 10 * np.log10(P + 1e-20)
    SPEC_DB[rid] = d - d[INB].mean()          # chỉ giữ HÌNH DẠNG phổ trong dải đã chọn

def _centroid(P):
    return float((F_W[INB] * P[INB]).sum() / (P[INB].sum() + 1e-30))
_cen = pd.Series({r: _centroid(P) for r, P in SPEC.items()}).groupby(lambda r: REC_DOM[r]).median()
R_EMP = float(_cen.get(1, np.nan) / _cen.get(0, np.nan))

if cfg.daug_warp == "auto":
    if abs(R_KORTEWEG - 1) > 0.05:
        DAUG_WARP_ON, DAUG_WARP_RANGE = True, (0.9 * R_KORTEWEG, 1.1 * R_KORTEWEG)
        _why = f"vật liệu khác nhau (r Korteweg = {R_KORTEWEG:.2f})"
    elif np.isfinite(R_EMP) and abs(R_EMP - 1) > 0.15:
        DAUG_WARP_ON, DAUG_WARP_RANGE = True, (0.9 * R_EMP, 1.1 * R_EMP)
        _why = f"r_emp = {R_EMP:.2f} lệch khỏi 1"
    else:
        DAUG_WARP_ON, DAUG_WARP_RANGE = False, (1.0, 1.0)
        _why = f"r_emp = {R_EMP:.2f} ≈ 1 và cùng vật liệu"
else:
    DAUG_WARP_ON, DAUG_WARP_RANGE = cfg.daug_warp == "on", tuple(cfg.daug_warp_range)
    _why = f"cfg.daug_warp = '{cfg.daug_warp}'"
print(f"Tần số trọng tâm trong dải (trung vị): {DOM_NAMES[0]} {_cen.get(0, np.nan):.1f} Hz | "
      f"{DOM_NAMES[1]} {_cen.get(1, np.nan):.1f} Hz -> r_emp = {R_EMP:.3f}")
print(f"DAug dịch tần: {'BẬT ' + str(tuple(round(v, 3) for v in DAUG_WARP_RANGE)) if DAUG_WARP_ON else 'TẮT'} ({_why})")

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1.1, 1]))
fz = (F_W >= SEARCH[0] * 0.7) & (F_W <= min(0.5 * A_SR, SEARCH[1] * 1.2))
for key, st in GROUP_STYLE.items():
    rids = [r for r in SPEC if REC_DOM[r] == key[0] and REC_LABEL[r] == key[1]]
    if not rids:
        continue
    M = np.stack([SPEC_DB[r] for r in rids]); q1, med, q3 = np.percentile(M, [25, 50, 75], axis=0)
    ax[0].plot(F_W[fz], med[fz], color=st["color"], lw=1.5, ls="-" if key[0] == 0 else "--", label=f"{st['label']} (n={len(rids)})")
    ax[0].fill_between(F_W[fz], q1[fz], q3[fz], color=st["color"], alpha=.10)
ax[0].axvspan(_lo, _hi, color=PAL["acc"], alpha=.07, lw=0, label="dải huấn luyện")
ax[0].set_xscale("log"); ax[0].set_xlabel("Hz (log)"); ax[0].set_ylabel("PSD chuẩn hoá trong dải (dB)")
ax[0].set_title("(a) Phổ trung vị mức bản ghi"); ax[0].legend(fontsize=7.5, loc="lower left")
for d in (0, 1):
    rids = [r for r in SPEC if REC_DOM[r] == d]
    if rids:
        M = np.stack([10 * np.log10(SPEC[r] / (SPEC[r][INB].sum() + 1e-30) + 1e-20) for r in rids])
        ax[1].plot(F_W[fz], np.percentile(M, 10, axis=0)[fz], color=PAL["tgt"] if d else PAL["src"], lw=1.5,
                   label=f"{DOM_NAMES[d]}: phân vị 10 %")
        ax[1].plot(F_W[fz], np.percentile(M, 90, axis=0)[fz], color=PAL["tgt"] if d else PAL["src"], lw=.8, ls=":",
                   label=f"{DOM_NAMES[d]}: phân vị 90 %")
ax[1].axvspan(_lo, _hi, color=PAL["acc"], alpha=.07, lw=0)
ax[1].set_xscale("log"); ax[1].set_xlabel("Hz (log)"); ax[1].set_ylabel("dB (so với năng lượng trong dải)")
ax[1].set_title("(b) Phổ nền / phổ đỉnh theo miền (không nhãn)"); ax[1].legend(fontsize=7.5)
q = qual.iloc[::-1]
ax[2].barh(q.descriptor, q.domain_sep, color=[PAL["tgt"] if v > 0.6 else PAL["grey"] for v in q.domain_sep])
ax[2].axvline(0.6, color="k", ls=":", lw=.7); ax[2].set_xlim(0, 1)
ax[2].set_xlabel("độ tách miền 2|AUC−0,5|"); ax[2].set_title("(c) Lệch miền ở chỉ số nào?")
savefig(fig, "domain_shift_analysis",
        f"Lệch miền {cfg.source_name} → {cfg.target_name}: (a) phổ theo miền × lớp (nét đứt: {cfg.target_name}; nhãn "
        f"{cfg.target_name} chỉ để mô tả); (b) phổ nền (phân vị 10 %) và phổ đỉnh (90 %); (c) độ tách miền của từng chỉ số chất lượng.")

fig, ax = plt.subplots(figsize=(12, 3.9))
wbar = np.diff(np.log10(EDGES)).mean() * 0.35
xl = np.log10(BAND_C)
ax.bar(xl - wbar / 2, src_disc, width=wbar, color=PAL["src"], label=f"rò làm tăng năng lượng dải ({cfg.source_name})")
ax.bar(xl + wbar / 2, dom_sep, width=wbar, color=PAL["grey"], alpha=.7,
       label=f"tách miền {cfg.source_name}/{cfg.target_name} (min theo lớp nguồn)")
ax.plot(xl, score, "o-", color=PAL["acc"], lw=1.6, label="điểm chuyển giao")
if tgt_disc is not None:
    ax.plot(xl, tgt_disc, "s--", color=PAL["tgt"], lw=1.1, ms=4,
            label=f"phân biệt trên {cfg.target_name} (CHỈ đánh giá, ρ = {BAND_RHO:.2f})")
ax.axvspan(np.log10(cfg.band_hz[0]), np.log10(cfg.band_hz[1]), color=PAL["acc"], alpha=.08, label="dải đã chọn")
ax.set_xticks(xl); ax.set_xticklabels([f"{c:.0f}" for c in BAND_C], rotation=45, fontsize=8)
ax.set_xlabel("tâm dải (Hz)"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8, ncol=2, loc="upper right")
savefig(fig, "band_transferability",
        f"Hồ sơ dải chuyển giao: điểm = phân biệt trên {cfg.source_name} × (1 − tách miền). Dải "
        f"{cfg.band_hz[0]:.0f}–{cfg.band_hz[1]:.0f} Hz ({'chọn tự động' if cfg.band_mode == 'auto' else 'cố định'}); "
        f"không dùng nhãn {cfg.target_name}.")
''')

# =============================================================================
# 5. Frame + VMD
# =============================================================================
md(r'''
## 5. Cắt frame và khử nhiễu VMD / Framing and VMD denoising

Frame `frame_sec` = 2 s, bước 1 s, ở `target_sr` vừa suy ra; mỗi frame mang mã bản ghi mẹ. Có thể giới hạn số frame mỗi
bản ghi (`max_frames_per_record`) khi «S» rất lớn.

**VMD** (Dragomiretskiy & Zosso, 2014) phân rã mỗi frame thành $K$ mode băng hẹp bằng ADMM trong miền tần số:

$$\hat u_k^{n+1}(\omega)=\frac{\hat f(\omega)-\sum_{i\neq k}\hat u_i(\omega)+\hat\lambda(\omega)/2}{1+2\alpha(\omega-\omega_k)^2},\qquad
\omega_k^{n+1}=\frac{\int_0^\infty\omega|\hat u_k(\omega)|^2d\omega}{\int_0^\infty|\hat u_k(\omega)|^2d\omega}$$

Tín hiệu khử nhiễu = tổng các mode có tâm **trong dải đã chọn** và **không trùng hài 50 Hz**. VMD chạy **theo lô trên GPU**
một lần rồi lưu cache; phân bố $\omega_k$ theo miền là thêm một thước đo lệch miền không cần nhãn.
''')

code(r'''
# =========================================================
# 5.1 Cắt frame + cache / Framing with on-disk cache
# =========================================================
def _fingerprint():
    h = hashlib.sha1()
    for r in files.sort_values("path").itertuples():
        try:
            st = os.stat(r.path); meta = f"{st.st_size}|{int(st.st_mtime)}"
        except Exception:
            meta = "NA"
        h.update(f"{r.path}|{meta}|{r.record_id}|{r.label}|{r.domain}|"
                 f"{json.dumps(SIM_PARAMS.get(r.record_id), sort_keys=True)}".encode("utf-8"))
    h.update(json.dumps(dict(sr=cfg.target_sr, band=list(cfg.band_hz), order=cfg.filter_order, notch=cfg.notch_mains,
                             mh=cfg.mains_hz, q=cfg.notch_q, fl=FRAME_LEN, hl=HOP_LEN, alpha=cfg.daug_alpha,
                             cap=cfg.max_frames_per_record, comb=1), sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:12]

_NF_REC = files.groupby("record_id").size()
def build_frames():
    chunks, metas = [], []
    for r in files.itertuples():
        x = load_audio(r.path, sim=SIM_PARAMS.get(r.record_id))
        if len(x) < FRAME_LEN:
            x = np.pad(x, (0, FRAME_LEN - len(x)))
        starts = np.arange(0, len(x) - FRAME_LEN + 1, HOP_LEN)
        if cfg.max_frames_per_record:
            cap = max(1, int(math.ceil(cfg.max_frames_per_record / _NF_REC[r.record_id])))
            if len(starts) > cap:
                starts = starts[np.linspace(0, len(starts) - 1, cap).round().astype(int)]
        for s in starts:
            seg = x[s:s + FRAME_LEN]
            chunks.append(seg)
            metas.append(dict(record_id=r.record_id, domain=int(r.domain), label=int(r.label), path=r.path,
                              start_s=s / cfg.target_sr, rms=float(np.sqrt(np.mean(seg ** 2)))))
    return np.stack(chunks).astype(np.float32), pd.DataFrame(metas)

FP = _fingerprint()
_xp, _mp = CACHE_DIR / f"frames_{FP}.npy", CACHE_DIR / f"frames_{FP}.csv"
if _xp.exists() and _mp.exists():
    X_PRE, frames = np.load(_xp), pd.read_csv(_mp)
    print("Nạp cache frame:", _xp.name)
else:
    _t0 = time.time()
    X_PRE, frames = build_frames()
    np.save(_xp, X_PRE); frames.to_csv(_mp, index=False)
    print(f"Cắt frame xong trong {time.time() - _t0:.1f} s -> {_xp.name}")
assert np.isfinite(X_PRE).all(), "Còn NaN/Inf trong frame"
print(f"Frames: {X_PRE.shape} ({X_PRE.nbytes / 1024 ** 2:.0f} MB)")
''')

code(r'''
# =========================================================
# 5.2 VMD theo lô trên GPU / Batched VMD in torch
# =========================================================
@torch.no_grad()
def vmd_torch(x, K=4, alpha=1500.0, tau=0.0, iters=150, tol=1e-6):
    """VMD (Dragomiretskiy & Zosso 2014), bám sát vmdpy nhưng chạy theo LÔ.
    x: [B, N] thực -> modes [B, K, N], omega [B, K] (chu kỳ/mẫu; Hz = omega * sr), số vòng lặp."""
    x = x.float()
    B, N = x.shape
    dev = x.device
    h = N // 2
    xm = torch.cat([x[:, :h].flip(1), x, x[:, N - h:].flip(1)], dim=1)
    T = xm.shape[1]
    freqs = torch.arange(1, T + 1, device=dev, dtype=torch.float32) / T - 0.5 - 1.0 / T
    f_hat = torch.fft.fftshift(torch.fft.fft(xm, dim=1), dim=1)
    f_plus = f_hat.clone(); f_plus[:, :T // 2] = 0
    u = torch.zeros(B, K, T, dtype=torch.complex64, device=dev)
    omega = ((0.5 / K) * torch.arange(K, device=dev, dtype=torch.float32)).repeat(B, 1)
    lam = torch.zeros(B, T, dtype=torch.complex64, device=dev)
    fpos = freqs[T // 2:]
    n_it = 0
    for it in range(int(iters)):
        u_prev = u.clone()
        total = u.sum(1)
        for k in range(K):
            others = total - u[:, k]
            uk = (f_plus - others - lam / 2) / (1.0 + alpha * (freqs[None, :] - omega[:, k:k + 1]) ** 2)
            u[:, k] = uk
            total = others + uk
            pw = uk[:, T // 2:].abs() ** 2
            omega[:, k] = (pw * fpos[None, :]).sum(1) / pw.sum(1).clamp_min(1e-20)
        if tau > 0:
            lam = lam + tau * (u.sum(1) - f_plus)
        n_it = it + 1
        if float((((u - u_prev).abs() ** 2).sum(dim=(1, 2)) / T).max()) < tol:
            break
    uh = torch.zeros_like(u)
    uh[:, :, T // 2:] = u[:, :, T // 2:]
    uh[:, :, torch.arange(T // 2, 0, -1, device=dev)] = u[:, :, T // 2:].conj()
    uh[:, :, 0] = uh[:, :, -1].conj()
    modes = torch.fft.ifft(torch.fft.ifftshift(uh, dim=2), dim=2).real[:, :, T // 4: 3 * T // 4]
    return modes, omega, n_it

def vmd_keep_mask(fhz):
    lo, hi = cfg.band_hz
    keep = (fhz >= lo) & (fhz <= hi)
    if cfg.notch_mains:
        h = 1
        while cfg.mains_hz * h <= hi + 25:
            keep &= (fhz - cfg.mains_hz * h).abs() > cfg.vmd_mains_tol_hz
            h += 1
    return keep

VMD_BATCH = max(8, int(cfg.vmd_batch * 4000 / max(FRAME_LEN, 1)))     # bộ nhớ ~ B x frame
def vmd_denoise_frames(X, device):
    n = len(X)
    out = np.empty_like(X); OM = np.zeros((n, cfg.vmd_K), np.float32)
    KEEP = np.zeros((n, cfg.vmd_K), bool); EF = np.zeros(n, np.float32); its = []
    for s in range(0, n, VMD_BATCH):
        xb = torch.from_numpy(np.ascontiguousarray(X[s:s + VMD_BATCH])).to(device)
        modes, om, it = vmd_torch(xb, cfg.vmd_K, cfg.vmd_alpha, cfg.vmd_tau, cfg.vmd_iters, cfg.vmd_tol)
        fhz = om * cfg.target_sr
        keep = vmd_keep_mask(fhz)
        rec = (modes * keep[:, :, None].float()).sum(1)
        rec = torch.where(keep.any(1)[:, None], rec, xb)
        e = (rec ** 2).sum(1) / (xb ** 2).sum(1).clamp_min(1e-12)
        sl = slice(s, s + len(xb))
        out[sl] = rec.cpu().numpy(); OM[sl] = fhz.cpu().numpy(); KEEP[sl] = keep.cpu().numpy()
        EF[sl] = e.cpu().numpy(); its.append(it)
    return out, OM, KEEP, EF, its

_sr = cfg.target_sr; _tt = np.arange(FRAME_LEN) / _sr
_f1, _f2 = 0.02 * _sr, 0.09 * _sr                    # tần số chuẩn hoá cố định -> kiểm thử không phụ thuộc dải
_sig = np.sin(2 * np.pi * _f1 * _tt) + .7 * np.sin(2 * np.pi * _f2 * _tt)
_m, _o, _ = vmd_torch(torch.from_numpy(_sig[None].astype(np.float32)), K=2, alpha=2000, iters=300)
_fo = sorted((_o[0] * _sr).tolist())
assert abs(_fo[0] - _f1) < max(3, .03 * _f1) and abs(_fo[1] - _f2) < max(5, .03 * _f2), f"VMD tìm sai tần số tâm: {_fo}"
print(f"Kiểm thử VMD: tâm mode {[round(v, 1) for v in _fo]} Hz (kỳ vọng {_f1:.1f}, {_f2:.1f})")

VMD_KEY = f"{FP}_vmd{int(cfg.use_vmd)}_K{cfg.vmd_K}_a{cfg.vmd_alpha:g}_t{cfg.vmd_tau:g}_i{cfg.vmd_iters}_m{cfg.vmd_mains_tol_hz:g}"
if cfg.use_vmd:
    _vp, _vm = CACHE_DIR / f"vmd_{VMD_KEY}.npy", CACHE_DIR / f"vmd_{VMD_KEY}.npz"
    if _vp.exists() and _vm.exists():
        X_ALL = np.load(_vp); _z = np.load(_vm)
        VMD_OM, VMD_KEEP, VMD_EF = _z["om"], _z["keep"], _z["ef"]
        print("Nạp cache VMD:", _vp.name)
    else:
        _t0 = time.time()
        X_ALL, VMD_OM, VMD_KEEP, VMD_EF, _its = vmd_denoise_frames(X_PRE, DEVICE)
        np.save(_vp, X_ALL); np.savez(_vm, om=VMD_OM, keep=VMD_KEEP, ef=VMD_EF)
        print(f"VMD {len(X_PRE)} frame trong {time.time() - _t0:.1f} s trên {DEVICE} "
              f"(lô {VMD_BATCH}, vòng lặp TB {np.mean(_its):.0f}/{cfg.vmd_iters})")
    for k in range(cfg.vmd_K):
        frames[f"vmd_f{k}"] = VMD_OM[:, k]
    frames["vmd_energy_kept"] = VMD_EF
else:
    X_ALL = X_PRE
    VMD_OM = VMD_KEEP = VMD_EF = None
    print("cfg.use_vmd=False -> dùng frame sau lọc dải.")
X_ALL = np.ascontiguousarray(X_ALL, dtype=np.float32)
assert np.isfinite(X_ALL).all()

Y_ALL = frames.label.values.astype(np.float32)          # -1 = không nhãn
DOM_ALL = frames.domain.values.astype(np.int64)
REC_ALL = frames.record_id.values.astype(str)
REC2IDX = {r: np.asarray(v, dtype=np.int64) for r, v in frames.groupby("record_id").indices.items()}
Y_SRC_VISIBLE = np.where(DOM_ALL == 0, Y_ALL, np.nan).astype(np.float32)   # nhãn mà UDA được phép thấy
assert np.isnan(Y_SRC_VISIBLE[DOM_ALL == 1]).all()

display(frames.groupby(["domain", "label"]).agg(n_frames=("record_id", "size"), n_records=("record_id", "nunique"),
                                               rms=("rms", "mean")).round(4))
if cfg.use_vmd:
    print("Tỉ lệ năng lượng giữ lại sau VMD (trung vị):",
          {DOM_NAMES[d]: round(float(np.median(VMD_EF[DOM_ALL == d])), 3) for d in (0, 1)})
''')

code(r'''
# =========================================================
# 5.3 Hình: VMD - mode, tần số tâm và phân bố theo miền
# =========================================================
if cfg.use_vmd:
    fig = plt.figure(figsize=(15.5, 6.2))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.2, 1.2, 1], hspace=.45, wspace=.28)
    for row, d in enumerate((0, 1)):
        lab = 1 if (d == 0 or TARGET_HAS_LABELS) else -1
        cand = np.where((DOM_ALL == d) & (Y_ALL == lab))[0]
        if not len(cand):
            cand = np.where(DOM_ALL == d)[0]
        i = int(cand[len(cand) // 2])
        modes, om, _ = vmd_torch(torch.from_numpy(X_PRE[i:i + 1]), cfg.vmd_K, cfg.vmd_alpha, cfg.vmd_tau,
                                 cfg.vmd_iters, cfg.vmd_tol)
        fhz = (om[0] * cfg.target_sr).numpy(); keep = vmd_keep_mask(om * cfg.target_sr)[0].numpy()
        n_show = int(0.6 * cfg.target_sr); tt = np.arange(n_show) / cfg.target_sr
        a0 = fig.add_subplot(gs[row, 0])
        a0.plot(tt, X_PRE[i, :n_show], color=PAL["grey"], lw=.6, label="sau lọc dải")
        a0.plot(tt, X_ALL[i, :n_show], color=PAL["tgt"] if d else PAL["src"], lw=.8, label="sau VMD")
        a0.set_title(f"({'ab'[row]}) {GROUP_STYLE[(d, lab)]['label']}: dạng sóng"); a0.set_xlabel("s")
        a0.legend(fontsize=8, loc="upper right")
        a1 = fig.add_subplot(gs[row, 1])
        for k in range(cfg.vmd_K):
            f_, P_ = sps.welch(modes[0, k].numpy(), fs=cfg.target_sr, nperseg=min(FRAME_LEN, int(2 ** round(math.log2(cfg.target_sr / 4)))))
            a1.semilogy(f_, P_ + 1e-14, lw=1.1, ls="-" if keep[k] else ":",
                        label=f"mode {k}: {fhz[k]:.0f} Hz {'(giữ)' if keep[k] else '(bỏ)'}")
        a1.set_xlim(0, min(cfg.target_sr / 2, 1.4 * cfg.band_hz[1])); a1.set_xlabel("Hz"); a1.set_title("PSD từng mode VMD")
        a1.legend(fontsize=7)
    a2 = fig.add_subplot(gs[:, 2])
    bins = np.linspace(0, min(cfg.target_sr / 2, 1.4 * cfg.band_hz[1]), 61)
    for d in (0, 1):
        a2.hist(VMD_OM[DOM_ALL == d].ravel(), bins=bins, density=True, histtype="step", lw=1.6,
                color=PAL["tgt"] if d else PAL["src"], label=DOM_NAMES[d])
    for v in cfg.band_hz:
        a2.axvline(v, color="k", ls=":", lw=.7)
    a2.set_xlabel("tần số tâm mode (Hz)"); a2.set_ylabel("mật độ")
    a2.set_title("(c) Phân bố tần số tâm VMD theo miền\n(không dùng nhãn)"); a2.legend()
    savefig(fig, "vmd", "VMD theo lô: (a, b) frame đại diện mỗi miền trước/sau VMD và PSD từng mode (nét chấm = mode bị loại "
            "vì ngoài dải hoặc trùng hài 50 Hz); (c) phân bố tần số tâm mode theo miền.")
''')

# =============================================================================
# 6. Kiến trúc
# =============================================================================
md(r'''
## 6. Kiến trúc 1D/2D-CNN + GRL (DANN, CDAN) + Deep CORAL / Architectures

```
            x (sóng 2 s @ target_sr, đã lọc dải + VMD)
                        │
        ┌───────────────┴────────────────┐
   [CNN2D]  CWT Morlet 64 tần số log (dải đã chọn)   [CNN1D]  chuẩn hoá theo mẫu
        → dB → SpecAugment → chuẩn hoá frame              → Conv1d ~32 ms (bộ lọc dải học được)
        → CMVN THEO MIỀN (z-score từng dải)                → 3 khối Conv1d + CBAM-1D
        → 4 khối Conv2d + CBAM                             │
        └───────────────┬────────────────┘
               gộp TB + cực đại → bottleneck → f ∈ R^d   (= G_f, dùng chung hai miền)
        ┌──────────┬────┴───────────────┬───────────────────────────┬──────────────────┐
   G_y: leak?   GRL(−λ_p) → G_d(f)    GRL(−λ_p) → G_dc(f ⊗ ŷ) (CDAN)   CORAL(f_S, f_T)
  (chỉ «S»)      DANN: căn chỉnh biên   căn chỉnh CÓ ĐIỀU KIỆN theo lớp    1/4d²·‖C_S − C_T‖²_F
```

- **GRL**: xuôi = đồng nhất; ngược nhân gradient với $-\lambda_p$ → $G_f$ học "đánh lừa" bộ phân biệt miền.
- **CDAN+E** (Long et al., 2018): bộ phân biệt nhận $f\otimes\hat y = [f(1-\hat p),\, f\hat p]$ ($\hat p$ tách gradient) nên căn
  chỉnh **từng mode lớp** thay vì cả phân phối biên; trọng số $1+e^{-H(\hat p)}$ ưu tiên mẫu dự đoán chắc chắn. Giảm rủi ro
  *negative transfer* khi tỉ lệ rò của hai miền khác nhau.
- **Deep CORAL**: căn chỉnh thống kê bậc 2, bất biến với dịch trung bình, tính ở fp32.
- **CMVN theo miền**: trừ phổ log trung bình và chia độ lệch chuẩn từng dải, thống kê riêng mỗi miền (nguồn: tập huấn luyện;
  đích: phần thích nghi không nhãn) → loại hàm truyền tĩnh của cảm biến/ghép nối/ống. Không chuẩn hoá theo từng frame vì
  rò rỉ là nhiễu **dừng**.
''')

code(r'''
# =========================================================
# 6.1 Front-end thời gian-tần số + SpecAugment + CMVN theo miền
# =========================================================
def frame_norm(s):
    s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=-100.0)
    mu = s.mean(dim=(2, 3), keepdim=True)
    sd = s.std(dim=(2, 3), keepdim=True).clamp_min(1e-3)
    return (s - mu) / sd

class CWTFrontend(nn.Module):
    """CWT Morlet phức, tần số chia đều theo LOG trong cfg.band_hz, tính bằng conv1d. Ra: dB [B, 1, F, T]."""
    def __init__(self, sr=None, band=None, n_freqs=None, hop=None, w0=None):
        super().__init__()
        sr = int(sr or cfg.target_sr); lo, hi = band or cfg.band_hz
        n = int(n_freqs or cfg.n_freqs); w0 = float(w0 or cfg.cwt_w0)
        self.hop = int(hop or cfg.tf_hop)
        self.freqs = np.geomspace(lo, hi, n)
        sig = w0 / (2 * np.pi * self.freqs)
        self.half = int(np.ceil(4 * sig.max() * sr))
        t = (np.arange(2 * self.half + 1) - self.half) / sr
        env = np.exp(-0.5 * (t[None, :] / sig[:, None]) ** 2)
        env /= env.sum(1, keepdims=True)
        w = np.concatenate([env * np.cos(2 * np.pi * self.freqs[:, None] * t[None, :]),
                            env * np.sin(2 * np.pi * self.freqs[:, None] * t[None, :])], 0)
        self.register_buffer("weight", torch.from_numpy(w[:, None, :].astype(np.float32)))
        self.n_bands = n
    def forward(self, x):
        with amp_autocast(x.device, False):
            y = F.conv1d(x.float().unsqueeze(1), self.weight, stride=self.hop, padding=self.half)
            p = y[:, :self.n_bands] ** 2 + y[:, self.n_bands:] ** 2
            db = 10.0 * torch.log10(p.clamp_min(1e-10))
            db = torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - 80.0)
            return db.unsqueeze(1)

class LogMelFrontend(nn.Module):
    def __init__(self, sr=None, band=None, n_mels=None, n_fft=None, hop=None):
        super().__init__()
        sr = int(sr or cfg.target_sr); lo, hi = band or cfg.band_hz
        n = int(n_mels or cfg.n_freqs); self.n_fft = int(n_fft or cfg.n_fft); self.hop = int(hop or cfg.tf_hop)
        fb = librosa.filters.mel(sr=sr, n_fft=self.n_fft, n_mels=n, fmin=lo, fmax=hi, htk=True, norm=None)
        self.register_buffer("fb", torch.from_numpy(fb.astype(np.float32)))
        self.register_buffer("win", torch.hann_window(self.n_fft))
        self.freqs = librosa.mel_frequencies(n_mels=n + 2, fmin=lo, fmax=hi, htk=True)[1:-1]
        self.n_bands = n
    def forward(self, x):
        with amp_autocast(x.device, False):
            S = torch.stft(x.float(), self.n_fft, hop_length=self.hop, window=self.win,
                           center=True, return_complex=True).abs() ** 2
            db = 10.0 * torch.log10(torch.matmul(self.fb, S).clamp_min(1e-10))
            db = torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - 80.0)
            return db.unsqueeze(1)

def make_tf_frontend():
    return CWTFrontend() if cfg.tf_repr == "cwt" else LogMelFrontend()

class SpecAugment(nn.Module):
    # MỖI MẪU một mặt nạ, chỉ khi training, áp TRƯỚC chuẩn hoá, giá trị che = mức nhỏ nhất của mẫu
    def __init__(self, f=None, t=None, n=2):
        super().__init__()
        self.f = cfg.specaug_freq_mask if f is None else int(f)
        self.t = cfg.specaug_time_mask if t is None else int(t)
        self.n = int(n)
        self.generator = None
    def _band_mask(self, B, L, W, device):
        g = self.generator
        w = torch.randint(0, W + 1, (B,), generator=g)
        s0 = (torch.rand(B, generator=g) * (L - w).clamp_min(1).float()).long()
        ar = torch.arange(L)
        return ((ar[None, :] >= s0[:, None]) & (ar[None, :] < (s0 + w)[:, None])).to(device)
    def forward(self, s):
        if not self.training or (self.f <= 0 and self.t <= 0):
            return s
        B, _, Fq, Tm = s.shape
        fill = s.amin(dim=(2, 3), keepdim=True).detach()
        mask = torch.zeros(B, 1, Fq, Tm, dtype=torch.bool, device=s.device)
        for _ in range(self.n):
            if self.f > 0 and Fq > self.f:
                mask = mask | self._band_mask(B, Fq, self.f, s.device)[:, None, :, None]
            if self.t > 0 and Tm > self.t:
                mask = mask | self._band_mask(B, Tm, self.t, s.device)[:, None, None, :]
        return torch.where(mask, fill, s)

def set_specaug_generators(model, generator):
    for m in model.modules():
        if isinstance(m, SpecAugment):
            m.generator = generator

class DomainCMVN(nn.Module):
    """z-score THEO DẢI, thống kê RIÊNG từng miền (0 = nguồn, 1 = đích)."""
    def __init__(self, n_bands, n_domains=2):
        super().__init__()
        self.register_buffer("mu", torch.zeros(n_domains, n_bands))
        self.register_buffer("sd", torch.ones(n_domains, n_bands))
        self.enabled = (cfg.input_norm == "domain_cmvn")
    def forward(self, s, dom):
        if not self.enabled or dom is None:
            return s
        mu = self.mu.index_select(0, dom)[:, None, :, None]
        sd = self.sd.index_select(0, dom)[:, None, :, None]
        return (s - mu) / sd
    @torch.no_grad()
    def set_stats(self, d, mu, sd):
        self.mu[d] = mu.to(self.mu.device); self.sd[d] = sd.clamp_min(1e-3).to(self.sd.device)
''')

code(r'''
# =========================================================
# 6.2 CBAM + khối tích chập + hai bộ trích đặc trưng G_f (1D / 2D)
# =========================================================
class ChannelAttention(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        hid = max(4, ch // r)
        self.mlp = nn.Sequential(nn.Linear(ch, hid), nn.ReLU(inplace=True), nn.Linear(hid, ch))
    def forward(self, x):
        dims = tuple(range(2, x.dim()))
        a = self.mlp(x.mean(dim=dims)) + self.mlp(x.amax(dim=dims))
        return x * torch.sigmoid(a).reshape(tuple(a.shape) + (1,) * len(dims))

class SpatialAttention(nn.Module):
    def __init__(self, dims=2, k=7):
        super().__init__()
        conv = nn.Conv2d if dims == 2 else nn.Conv1d
        self.conv = conv(2, 1, k, padding=k // 2, bias=False)
    def forward(self, x):
        m = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(m))

class CBAM(nn.Module):
    def __init__(self, ch, dims=2):
        super().__init__()
        self.ca, self.sa = ChannelAttention(ch), SpatialAttention(dims)
    def forward(self, x):
        return self.sa(self.ca(x))

class ConvBlock2D(nn.Module):
    def __init__(self, cin, cout, pool=(2, 2), p_drop=0.1, cbam=True):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.att = CBAM(cout, 2) if cbam else nn.Identity()
        self.pool, self.drop = nn.MaxPool2d(pool), nn.Dropout2d(p_drop)
    def forward(self, x):
        return self.drop(self.pool(self.att(self.body(x))))

class ConvBlock1D(nn.Module):
    def __init__(self, cin, cout, k=9, pool=4, p_drop=0.1, cbam=True):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
            nn.Conv1d(cout, cout, k, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True))
        self.att = CBAM(cout, 1) if cbam else nn.Identity()
        self.pool, self.drop = nn.MaxPool1d(pool), nn.Dropout(p_drop)
    def forward(self, x):
        return self.drop(self.pool(self.att(self.body(x))))

def gpool(h):
    dims = tuple(range(2, h.dim()))
    return torch.cat([h.mean(dim=dims), h.amax(dim=dims)], dim=1)

class CNN2D(nn.Module):
    """G_f 2D: CWT (hoặc log-Mel) trong dải đã chọn -> CMVN theo miền -> 4 khối Conv2d + CBAM."""
    kind = "2d"
    def __init__(self, emb_dim=None, widths=(32, 64, 128, 128), p_drop=0.1):
        super().__init__()
        emb_dim = int(emb_dim or cfg.emb_dim)
        self.front = make_tf_frontend()
        self.specaug = SpecAugment()
        self.cmvn = DomainCMVN(self.front.n_bands)
        self.stem = ConvBlock2D(1, widths[0], p_drop=p_drop, cbam=cfg.use_cbam)
        blocks, cin = [], widths[0]
        for w in widths[1:]:
            blocks.append(ConvBlock2D(cin, w, p_drop=p_drop, cbam=cfg.use_cbam)); cin = w
        self.mid = nn.Sequential(*blocks)
        self.bottleneck = nn.Sequential(nn.Linear(2 * cin, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(inplace=True))
        self.out_dim = emb_dim
    def tf_raw(self, x):
        return frame_norm(self.front(x))
    def tf(self, x, dom=None):
        return self.cmvn(frame_norm(self.specaug(self.front(x))), dom)
    def forward(self, x, dom=None):
        return self.bottleneck(gpool(self.mid(self.stem(self.tf(x, dom)))))
    def ft_groups(self):
        return dict(early=list(self.stem.parameters()), middle=list(self.mid.parameters()),
                    head=list(self.bottleneck.parameters()))

class CNN1D(nn.Module):
    """G_f 1D: sóng thô (chuẩn hoá theo mẫu) -> Conv1d ~32 ms (bước ~1 ms, theo target_sr) -> 3 khối Conv1d + CBAM-1D."""
    kind = "1d"
    def __init__(self, emb_dim=None, ch=(32, 64, 128, 128), p_drop=0.1):
        super().__init__()
        emb_dim = int(emb_dim or cfg.emb_dim)
        k0 = 2 * int(round(0.016 * cfg.target_sr)) + 1
        s0 = max(1, int(round(cfg.target_sr / 1000)))
        self.stem = nn.Sequential(nn.Conv1d(1, ch[0], k0, stride=s0, padding=k0 // 2, bias=False),
                                  nn.BatchNorm1d(ch[0]), nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.mid = nn.Sequential(ConvBlock1D(ch[0], ch[1], pool=4, p_drop=p_drop, cbam=cfg.use_cbam),
                                 ConvBlock1D(ch[1], ch[2], pool=4, p_drop=p_drop, cbam=cfg.use_cbam),
                                 ConvBlock1D(ch[2], ch[3], pool=2, p_drop=p_drop, cbam=cfg.use_cbam))
        self.bottleneck = nn.Sequential(nn.Linear(2 * ch[3], emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(inplace=True))
        self.out_dim = emb_dim
    def forward(self, x, dom=None):
        x = x.float()
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-4)
        return self.bottleneck(gpool(self.mid(self.stem(x.unsqueeze(1)))))
    def ft_groups(self):
        return dict(early=list(self.stem.parameters()), middle=list(self.mid.parameters()),
                    head=list(self.bottleneck.parameters()))

BACKBONES = {"CNN2D": CNN2D, "CNN1D": CNN1D}
''')

code(r'''
# =========================================================
# 6.3 GRL, G_y, G_d (DANN), G_dc (CDAN), mạng đầy đủ + CORAL / MMD
# =========================================================
class GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer: xuôi = đồng nhất, ngược = -lambda * grad."""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.lambd, None

class GRL(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambd = 0.0
    def forward(self, x):
        return GradReverse.apply(x, self.lambd)

class LabelPredictor(nn.Module):           # G_y
    def __init__(self, d, hidden=64, p=None):
        super().__init__()
        p = cfg.dropout if p is None else p
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(inplace=True), nn.Dropout(p), nn.Linear(hidden, 1))
    def forward(self, f):
        return self.net(f).squeeze(-1)

class DomainDiscriminator(nn.Module):      # G_d (DANN: vào d chiều) / G_dc (CDAN: vào 2d chiều)
    def __init__(self, d, hidden=None, p=None):
        super().__init__()
        h = int(hidden or cfg.disc_hidden); p = cfg.dropout if p is None else p
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(p),
                                 nn.Linear(h, h), nn.ReLU(inplace=True), nn.Dropout(p), nn.Linear(h, 1))
    def forward(self, f):
        return self.net(f).squeeze(-1)

def cdan_features(f, p):
    # Ánh xạ đa tuyến f ⊗ [1-p, p] cho bài toán nhị phân; p TÁCH gradient (như bản gốc CDAN)
    p = p.detach().to(f.dtype)[:, None]
    return torch.cat([f * (1 - p), f * p], dim=1)

class DANNNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone_name = backbone
        self.G_f = BACKBONES[backbone]()
        d = self.G_f.out_dim
        self.G_y, self.grl = LabelPredictor(d), GRL()
        self.G_d, self.G_dc = DomainDiscriminator(d), DomainDiscriminator(2 * d)
    def forward(self, x, dom=None):
        return self.G_y(self.G_f(x, dom))
    def forward_all(self, x, dom):
        f = self.G_f(x, dom)
        return self.G_y(f), self.G_d(self.grl(f)), f

def grl_schedule(p, gamma=None):
    gamma = cfg.grl_gamma if gamma is None else gamma
    return 2.0 / (1.0 + math.exp(-gamma * float(p))) - 1.0

def coral_loss(fs, ft):
    """Deep CORAL: 1/(4 d^2) ||C_S - C_T||_F^2 (fp32)."""
    with amp_autocast(fs.device, False):
        fs, ft = fs.float(), ft.float()
        d = fs.shape[1]
        def _cov(f):
            fc = f - f.mean(0, keepdim=True)
            return fc.t() @ fc / max(f.shape[0] - 1, 1)
        return ((_cov(fs) - _cov(ft)) ** 2).sum() / (4.0 * d * d)

def mmd_rbf(fs, ft, n_kernels=5, mul=2.0):
    """MMD^2 đa nhân Gauss (DAN)."""
    with amp_autocast(fs.device, False):
        x = torch.cat([fs, ft], 0).float(); n, ns = x.shape[0], fs.shape[0]
        d2 = torch.cdist(x, x).pow(2)
        bw = d2.detach().sum() / max(n * n - n, 1)
        K = sum(torch.exp(-d2 / (bw * mul ** (i - n_kernels // 2)).clamp_min(1e-8)) for i in range(n_kernels))
        return K[:ns, :ns].mean() + K[ns:, ns:].mean() - 2 * K[:ns, ns:].mean()

def entropy_weights(p, ns):
    # CDAN+E: w = 1 + exp(-H(p)), chuẩn hoá riêng từng miền
    p = p.detach().float().clamp(1e-6, 1 - 1e-6)
    w = 1.0 + torch.exp(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    ws, wt = w[:ns], w[ns:]
    return ws / ws.sum() * len(ws), wt / wt.sum() * len(wt)
''')

code(r'''
# =========================================================
# 6.4 Kiểm thử đơn vị + độ phức tạp mô hình
# =========================================================
_x = torch.randn(5, 3, requires_grad=True)
_y = GradReverse.apply(_x, 0.7)
assert torch.equal(_y.detach(), _x.detach())
_y.sum().backward()
assert torch.allclose(_x.grad, torch.full_like(_x, -0.7)), "GRL sai dấu/độ lớn"
_net = DomainDiscriminator(8).eval(); _f = torch.randn(16, 8, requires_grad=True); _dl = torch.randint(0, 2, (16,)).float()
_g1, = torch.autograd.grad(F.binary_cross_entropy_with_logits(_net(_f), _dl), _f)
_g2, = torch.autograd.grad(F.binary_cross_entropy_with_logits(_net(GradReverse.apply(_f, 1.0)), _dl), _f)
assert torch.allclose(_g1, -_g2, atol=1e-6)
_a = torch.randn(64, 16, dtype=torch.float64).float(); _b = _a * 2.0 + 0.3 * torch.randn(64, 16)
_ref = ((np.cov(_a.numpy(), rowvar=False) - np.cov(_b.numpy(), rowvar=False)) ** 2).sum() / (4 * 16 ** 2)
assert coral_loss(_a, _a).item() < 1e-10
assert abs(coral_loss(_a, _b).item() - _ref) < 1e-4 * max(1.0, _ref)
assert coral_loss(_a, _a + 5.0).item() < 1e-8, "CORAL phải bất biến với dịch trung bình"
_m0 = mmd_rbf(torch.randn(200, 8), torch.randn(200, 8)).item()
_m1 = mmd_rbf(torch.randn(200, 8), torch.randn(200, 8) + 1.5).item()
assert _m1 > 5 * max(_m0, 1e-4), (_m0, _m1)
assert abs(grl_schedule(0.0)) < 1e-9 and grl_schedule(1.0) > 0.99
# CDAN: f ⊗ [1-p, p] giữ thông tin lớp; p không nhận gradient
_fp = torch.randn(6, 4, requires_grad=True); _pp = torch.rand(6, requires_grad=True)
_h = cdan_features(_fp, _pp); assert _h.shape == (6, 8) and torch.allclose(_h[:, :4] + _h[:, 4:], _fp)
_h.sum().backward(); assert _pp.grad is None and _fp.grad is not None
_ws, _wt = entropy_weights(torch.tensor([.5, .99, .01, .5]), 2)
assert torch.allclose(_ws.sum(), torch.tensor(2.)) and torch.allclose(_wt.sum(), torch.tensor(2.))
assert _ws[1] > _ws[0] and _wt[0] > _wt[1], "CDAN+E phải ưu tiên mẫu dự đoán chắc chắn (entropy thấp)"
print(f"GRL / CDAN / CORAL / MMD / lịch lambda: OK  (CORAL ví dụ = {coral_loss(_a, _b).item():.4f}, MMD {_m0:.4f} vs {_m1:.4f})")

rows = []
with torch.no_grad():
    _xb = torch.from_numpy(X_ALL[:4].copy()); _dom = torch.tensor([0, 0, 1, 1])
    for bb in cfg.backbones:
        net = DANNNet(bb).eval()
        yl, dl, f = net.forward_all(_xb, _dom)
        assert yl.shape == (4,) and dl.shape == (4,) and f.shape == (4, cfg.emb_dim)
        assert net.G_dc(cdan_features(f, torch.sigmoid(yl))).shape == (4,)
        npar = lambda m: sum(p.numel() for p in m.parameters())
        info = dict(backbone=bb, G_f=npar(net.G_f), G_y=npar(net.G_y), G_d=npar(net.G_d), G_dc=npar(net.G_dc))
        if bb == "CNN2D":
            info["tf_shape"] = str(tuple(net.G_f.tf_raw(_xb).shape))
        rows.append(info)
param_table = pd.DataFrame(rows)
param_table["G_f+G_y_M"] = ((param_table.G_f + param_table.G_y) / 1e6).round(3)
display(Markdown("### Table 5. Model complexity (parameters)"))
display(param_table)
param_table.to_csv(TAB_DIR / "table5_model_complexity.csv", index=False)
''')

code(r'''
# =========================================================
# 6.5 Hình: ảnh CWT theo miền × lớp + lịch lambda_p
# =========================================================
fe = make_tf_frontend()
ex = {}
for d in (0, 1):
    for lab in ((1, 0) if (d == 0 or TARGET_HAS_LABELS) else (-1,)):
        cand = np.where((DOM_ALL == d) & (Y_ALL == lab))[0]
        if len(cand):
            ex[(d, lab)] = int(cand[np.argsort(frames.rms.values[cand])[len(cand) // 2]])
fig, axes = plt.subplots(1, len(ex) + 1, figsize=(3.6 * (len(ex) + 1), 3.4))
_cands = (20, 50, 100, 200, 350, 500, 1000, 2000, 4000)
yt_hz = [f for f in _cands if cfg.band_hz[0] <= f <= cfg.band_hz[1]] or [int(cfg.band_hz[0]), int(cfg.band_hz[1])]
yt_pos = np.interp(np.log(yt_hz), np.log(fe.freqs), np.arange(fe.n_bands))
with torch.no_grad():
    for a, ((d, lab), i) in zip(axes, ex.items()):
        s = fe(torch.from_numpy(X_ALL[i:i + 1].copy()))[0, 0].numpy()
        a.imshow(s, origin="lower", aspect="auto", cmap="magma", extent=[0, cfg.frame_sec, -.5, fe.n_bands - .5])
        a.set_yticks(yt_pos); a.set_yticklabels(yt_hz); a.grid(False)
        a.set_title(GROUP_STYLE[(d, lab)]["label"], color=GROUP_STYLE[(d, lab)]["color"])
        a.set_xlabel("s")
axes[0].set_ylabel(f"{cfg.tf_repr.upper()} - tần số (Hz)")
pp = np.linspace(0, 1, 200)
axes[-1].plot(pp, [cfg.lambda_dann * grl_schedule(p) for p in pp], color=PAL["aux"], label="λ_p (GRL)")
axes[-1].plot(pp, [cfg.coral_weight * grl_schedule(p) if cfg.ramp_alignment else cfg.coral_weight for p in pp],
              color=PAL["acc"], ls="--", label="β_p (CORAL)")
axes[-1].set_xlabel("tiến độ huấn luyện p"); axes[-1].set_title("Lịch trọng số căn chỉnh"); axes[-1].legend()
savefig(fig, "tf_inputs", f"Ảnh {cfg.tf_repr.upper()} (dB, trục tần số log {cfg.band_hz[0]:.0f}-{cfg.band_hz[1]:.0f} Hz) "
        f"của frame đại diện mỗi (miền, lớp) và lịch λ_p = 2/(1+e^(-γp)) - 1.")
''')

# =============================================================================
# 7. Chia dữ liệu
# =============================================================================
md(r'''
## 7. Giao thức chia dữ liệu chống rò rỉ / Leakage-free split protocol

- **Nguồn «S»:** bản ghi chia *train* / *val* (phân tầng). Val để theo dõi và báo hiệu năng trên chính «S».
- **Đích «T»:** bản ghi **có nhãn** chia `n_target_folds` fold phân tầng — **theo nhóm** (điểm khảo sát) nếu có
  `target_group_csv`. Fold *k*: *kiểm thử* = fold *k*; *thích nghi* = phần còn lại **cộng mọi bản ghi không nhãn**, đưa vào
  huấn luyện **không kèm nhãn**; *few-shot* ≈ 10 % bản ghi phần thích nghi (≥ 1/lớp), có nhãn, dùng cho Few-shot FT **và**
  Few-shot chỉ «T» (cùng bản ghi → so sánh ghép cặp). *Oracle* dùng nhãn toàn bộ phần thích nghi.
- Mỗi bản ghi «T» có nhãn được dự đoán **đúng một lần** bởi mô hình chưa từng thấy nó.
''')

code(r'''
# =========================================================
# 7.1 Dựng phân hoạch + assertion cứng
# =========================================================
def idx_of(recs):
    recs = sorted(recs)
    return np.sort(np.concatenate([REC2IDX[r] for r in recs])) if recs else np.zeros(0, np.int64)

def _target_folds(tl, seed):
    use_groups = tl.group.nunique() < len(tl)
    n_max = int(min(cfg.n_target_folds, tl.label.value_counts().min()))
    for n_f in range(n_max, 1, -1):
        if use_groups:
            split = StratifiedGroupKFold(n_splits=n_f, shuffle=True, random_state=seed).split(tl.record_id, tl.label, tl.group)
        else:
            split = StratifiedKFold(n_splits=n_f, shuffle=True, random_state=seed).split(tl.record_id, tl.label)
        folds = list(split)
        if all(tl.label.iloc[b].nunique() == 2 for _, b in folds):
            if n_f < n_max:
                print(f"  [lưu ý] chỉ dựng được {n_f} fold đích có đủ hai lớp (nhóm quá ít?)")
            return folds
    return []

def build_splits(seed):
    rng = np.random.default_rng(seed)
    src = records[records.domain == 0]; tgt = records[records.domain == 1]
    val = []
    for lab in (0, 1):
        r = sorted(src.record_id[src.label == lab])
        n_val = max(1, int(round(cfg.source_val_frac * len(r)))) if len(r) >= 3 else 0
        val += list(rng.choice(r, n_val, replace=False)) if n_val else []
    src_val = set(val); src_train = set(src.record_id) - src_val
    folds = []
    tl = tgt[tgt.label >= 0].reset_index(drop=True); unl = set(tgt.record_id[tgt.label < 0])
    if TARGET_HAS_LABELS:
        for k, (a, b) in enumerate(_target_folds(tl, seed)):
            test = set(tl.record_id.iloc[b]); adapt_lab = set(tl.record_id.iloc[a])
            fs = []
            for lab in (0, 1):
                r = sorted(x for x in adapt_lab if REC_LABEL[x] == lab)
                n = min(len(r), max(cfg.fewshot_min_records_per_class, int(round(cfg.fewshot_frac * len(r)))))
                fs += list(rng.choice(r, n, replace=False)) if n else []
            adapt = adapt_lab | unl
            folds.append(dict(fold=k, test_recs=test, adapt_recs=adapt, adapt_lab_recs=adapt_lab,
                              fewshot_recs=set(fs), t_test_idx=idx_of(test), t_adapt_idx=idx_of(adapt),
                              t_adapt_lab_idx=idx_of(adapt_lab), t_fewshot_idx=idx_of(fs)))
    final = dict(fold=None, test_recs=set(), adapt_recs=set(tgt.record_id), adapt_lab_recs=set(),
                 fewshot_recs=set(), t_test_idx=np.zeros(0, np.int64), t_adapt_idx=idx_of(set(tgt.record_id)),
                 t_adapt_lab_idx=np.zeros(0, np.int64), t_fewshot_idx=np.zeros(0, np.int64))
    return dict(seed=seed, src_train_recs=src_train, src_val_recs=src_val,
                src_train_idx=idx_of(src_train), src_val_idx=idx_of(src_val), folds=folds, final=final)

def check_splits(sp):
    a, b = sp["src_train_recs"], sp["src_val_recs"]
    assert not (a & b) and all(REC_DOM[r] == 0 for r in a | b)
    assert set(Y_ALL[sp["src_train_idx"]]) == {0.0, 1.0}, "nguồn train phải có cả hai lớp"
    tested = []
    for fd in sp["folds"]:
        assert not (fd["test_recs"] & fd["adapt_recs"]), "RÒ RỈ: bản ghi đích vừa thích nghi vừa kiểm thử"
        assert not ({REC_GROUP[r] for r in fd["test_recs"]} & {REC_GROUP[r] for r in fd["adapt_lab_recs"]}), \
            "RÒ RỈ: cùng một nhóm (điểm khảo sát) ở cả thích nghi và kiểm thử"
        assert fd["fewshot_recs"] <= fd["adapt_lab_recs"] <= fd["adapt_recs"]
        assert all(REC_DOM[r] == 1 for r in fd["test_recs"] | fd["adapt_recs"])
        assert all(REC_LABEL[r] >= 0 for r in fd["test_recs"])
        assert np.intersect1d(fd["t_test_idx"], fd["t_adapt_idx"]).size == 0
        assert set(Y_ALL[fd["t_test_idx"]]) == {0.0, 1.0}, "fold kiểm thử phải có cả hai lớp"
        tested += list(fd["test_recs"])
    if sp["folds"]:
        want = sorted(r for r in REC_LABEL if REC_DOM[r] == 1 and REC_LABEL[r] >= 0)
        assert sorted(tested) == want, "mỗi bản ghi đích có nhãn phải được kiểm thử ĐÚNG MỘT lần"

SPLITS = {s: build_splits(s) for s in cfg.seeds}
for s, sp in SPLITS.items():
    check_splits(sp)
EVAL_ON = bool(TARGET_HAS_LABELS and all(len(sp["folds"]) >= 2 for sp in SPLITS.values()))
sp0 = SPLITS[cfg.seeds[0]]
print(f"seed {cfg.seeds[0]}: {cfg.source_name} train {len(sp0['src_train_recs'])} bản ghi ({len(sp0['src_train_idx'])} frame) | "
      f"val {len(sp0['src_val_recs'])} ({len(sp0['src_val_idx'])} frame) | tỉ lệ leak nguồn (frame) "
      f"{Y_ALL[sp0['src_train_idx']].mean():.2f}")
for fd in sp0["folds"]:
    print(f"  fold {fd['fold']}: {cfg.target_name} kiểm thử {len(fd['test_recs'])} bản ghi ({len(fd['t_test_idx'])} frame) | "
          f"thích nghi {len(fd['adapt_recs'])} ({len(fd['t_adapt_idx'])} frame, không nhãn) | few-shot "
          f"{len(fd['fewshot_recs'])} bản ghi ({100 * len(fd['t_fewshot_idx']) / max(1, (DOM_ALL == 1).sum()):.1f}% frame đích)")
print("EVAL_ON =", EVAL_ON, "| assertion phân chia: OK")
''')

code(r'''
# =========================================================
# 7.2 Hình: bản đồ phân hoạch (seed đầu)
# =========================================================
recs_sorted = records.sort_values(["domain", "label", "record_id"]).record_id.tolist()
cols = ["nguồn"] + [f"fold {fd['fold']}" for fd in sp0["folds"]] + ["mô hình cuối"]
M = np.zeros((len(recs_sorted), len(cols)))
for i, r in enumerate(recs_sorted):
    if REC_DOM[r] == 0:
        M[i, :] = 1 if r in sp0["src_train_recs"] else 2
    else:
        for k, fd in enumerate(sp0["folds"]):
            M[i, k + 1] = (5 if r in fd["test_recs"] else 4 if r in fd["fewshot_recs"]
                           else 3 if r in fd["adapt_recs"] else 0)
        M[i, -1] = 3
cmap = ListedColormap(["#FFFFFF", "#2E6F95", "#9CC3DE", "#F4C08A", "#8C6BB1", "#C8553D"])
names = ["—", f"{cfg.source_name} train", f"{cfg.source_name} val", f"{cfg.target_name} thích nghi (không nhãn)",
         f"{cfg.target_name} few-shot (có nhãn)", f"{cfg.target_name} kiểm thử"]
fig, ax = plt.subplots(figsize=(1.3 * len(cols) + 3.5, min(9, 1.2 + .06 * len(recs_sorted))))
ax.imshow(M, aspect="auto", cmap=cmap, vmin=-.5, vmax=5.5, interpolation="nearest")
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols); ax.set_ylabel("bản ghi (theo miền, lớp)"); ax.grid(False)
ax.axhline(int((records.domain == 0).sum()) - .5, color="k", lw=1)
ax.legend(handles=[Patch(color=cmap(i), label=n) for i, n in enumerate(names) if i], loc="center left",
          bbox_to_anchor=(1.01, .5), fontsize=8)
savefig(fig, "split_map", f"Vai trò từng bản ghi trong mỗi fold. Nhãn {cfg.target_name} chỉ dùng ở ô 'kiểm thử' (đánh giá) "
        f"và 'few-shot' (riêng các phương pháp few-shot).")
''')

# =============================================================================
# 8. Bộ máy huấn luyện
# =============================================================================
md(r'''
## 8. Bộ máy huấn luyện / Training engine

Mỗi bước UDA: một lô **«S» có nhãn** (mặc định **cân bằng 50/50** leak/noleak — tránh truyền tỉ lệ lớp của «S» sang «T»)
và một lô **«T» không nhãn** cùng kích thước → tăng cường trên GPU (dịch vòng, nhiễu trong dải) → với phương pháp `*_aug`
thêm **DAug** cho lô «S»:

| Thành phần DAug | Mô phỏng điều gì ở «T» | Cần nhãn «T»? |
|---|---|---|
| nhiễu theo **phổ nền «T»** (phân vị `daug_floor_pct` của phổ các frame thích nghi) | nhiễu nền thực địa | **không** |
| xung va đập suy giảm hàm mũ | xe cộ, bước chân, van | không |
| EQ kênh ngẫu nhiên ± `daug_eq_db` | cảm biến / ghép nối / đất | không |
| suy hao $e^{-\alpha f x}$ | khoảng cách tới vết rò | không |
| dịch tần $r$ (chỉ khi bật ở mục 4.3) | vật liệu / vận tốc sóng khác | không |

Một lần lan truyền chung qua $G_f$ rồi $\mathcal L=\mathcal L_y+\mathcal L_{adv}+\beta_p\mathcal L_{CORAL}$, với
$\mathcal L_{adv}$ là DANN ($G_d(f)$) hoặc CDAN+E ($G_{dc}(f\otimes\hat y)$, trọng số entropy). Quy tắc cố định: **epoch cuối**,
ngưỡng 0,5; AUC «T» mỗi epoch chỉ để vẽ đường học. Cùng (seed, kiến trúc, fold) → cùng khởi tạo và cùng chuỗi lô «S» cho
mọi phương pháp (so sánh **ghép cặp**).
''')

code(r'''
# =========================================================
# 8.1 Tiện ích: dữ liệu trên GPU, tăng cường + DAug, đánh giá, CMVN, AdaBN
# =========================================================
_DEV_DATA, _DEV_LOCK, _INIT_LOCK = {}, threading.Lock(), threading.Lock()

def use_amp(device):
    return bool(cfg.amp and torch.device(device).type == "cuda")

def dev_data(device):
    key = str(device)
    with _DEV_LOCK:
        if key not in _DEV_DATA:
            X = torch.from_numpy(X_ALL)
            on_dev = torch.device(device).type == "cuda" and X_ALL.nbytes / 1024 ** 3 < cfg.gpu_resident_max_gb
            _DEV_DATA[key] = X.to(device) if on_dev else X
    return _DEV_DATA[key]

def fetch(device, idx):
    X = dev_data(device)
    ii = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=X.device)
    return X.index_select(0, ii).to(device, non_blocking=True)

def _uniform(n, lo, hi, g, device):
    return lo + (hi - lo) * torch.rand(n, generator=g, device=device)

def _band_mask(L, device):
    f = torch.fft.rfftfreq(L, 1.0 / cfg.target_sr).to(device)
    return f, ((f >= cfg.band_hz[0]) & (f <= cfg.band_hz[1])).float()

def band_noise(n, L, g, device, pink=False):
    f, shape = _band_mask(L, device)
    if pink:
        shape = shape / torch.sqrt(f.clamp_min(1.0))
    return torch.fft.irfft(torch.fft.rfft(torch.randn(n, L, generator=g, device=device), dim=1) * shape, n=L, dim=1)

def add_noise_snr(x, snr_db, sel, noise):
    ps = x.pow(2).mean(1, keepdim=True)
    pn = noise.pow(2).mean(1, keepdim=True).clamp_min(1e-12)
    k = torch.sqrt(ps / (pn * 10 ** (snr_db[:, None] / 10.0)))
    return torch.where(sel[:, None], x + k * noise, x)

@torch.no_grad()
def noise_floor_spectrum(idx, device, n_max=512, seed=0):
    """Phổ nền miền đích, KHÔNG dùng nhãn: phân vị daug_floor_pct của |rfft|^2 qua các frame thích nghi."""
    idx = np.asarray(idx)
    n_max = int(min(n_max, (1 << 24) // (FRAME_LEN // 2 + 1)))          # giới hạn kích thước của torch.quantile
    if len(idx) > n_max:
        idx = np.random.default_rng(seed).choice(idx, n_max, replace=False)
    P = torch.fft.rfft(fetch(device, idx).float(), dim=1).abs() ** 2
    fl = torch.quantile(P, cfg.daug_floor_pct / 100.0, dim=0)
    return fl * _band_mask(FRAME_LEN, device)[1]

def _warp(y, r, g):
    # y[n] = x(off + n*r): r < 1 dịch tần xuống (đọc một đoạn rồi kéo giãn), r > 1 dịch lên (đọc vòng - nhiễu dừng)
    n, L = y.shape; dev = y.device
    span = torch.where(r < 1, 1 - r, torch.ones_like(r))
    off = torch.rand(n, generator=g, device=dev) * (L - 1) * span
    pos = torch.remainder(off[:, None] + torch.arange(L, device=dev, dtype=torch.float32)[None, :] * r[:, None], L - 1)
    i0 = pos.floor().long().clamp(0, L - 2); w = pos - i0.float()
    return torch.gather(y, 1, i0) * (1 - w) + torch.gather(y, 1, i0 + 1) * w

def daug(x, g, floor=None):
    """Tăng cường miền cho lô NGUỒN (giữ nhãn) - xem bảng ở đầu mục 8."""
    B, L = x.shape; dev = x.device
    sel = torch.rand(B, generator=g, device=dev) < cfg.daug_p
    n = int(sel.sum())
    if n == 0:
        return x
    y = x[sel].float()
    if DAUG_WARP_ON:
        y = _warp(y, _uniform(n, *DAUG_WARP_RANGE, g, dev), g)
    f, _ = _band_mask(L, dev)
    H = torch.ones(n, f.shape[0], device=dev)
    if cfg.daug_dist_m[1] > 0:
        H = H * torch.exp(-cfg.daug_alpha * f[None, :] * _uniform(n, *cfg.daug_dist_m, g, dev)[:, None])
    if cfg.daug_eq_p > 0:
        lf = torch.log(f.clamp_min(1.0))[None, :]
        gdb = torch.zeros(n, f.shape[0], device=dev)
        for _ in range(3):
            c = _uniform(n, math.log(max(cfg.band_hz[0], 10.0)), math.log(cfg.band_hz[1]), g, dev)[:, None]
            s = _uniform(n, 0.2, 0.8, g, dev)[:, None]
            gdb = gdb + _uniform(n, -cfg.daug_eq_db, cfg.daug_eq_db, g, dev)[:, None] * torch.exp(-0.5 * ((lf - c) / s) ** 2)
        on = (torch.rand(n, generator=g, device=dev) < cfg.daug_eq_p).float()[:, None]
        H = H * (10 ** (gdb * on / 20.0))
    y = torch.fft.irfft(torch.fft.rfft(y, dim=1) * H, n=L, dim=1)
    if cfg.daug_transient_p > 0:
        t = torch.arange(L, device=dev, dtype=torch.float32)[None, :]
        env = torch.zeros(n, L, device=dev)
        for _ in range(3):
            pos = torch.rand(n, generator=g, device=dev)[:, None] * L
            tau = _uniform(n, 0.005, 0.05, g, dev)[:, None] * cfg.target_sr
            act = (torch.rand(n, generator=g, device=dev) < 0.6).float()[:, None]
            env = env + act * torch.exp(-(t - pos).clamp_min(0) / tau) * (t >= pos).float()
        burst = env * torch.randn(n, L, generator=g, device=dev)
        tsel = (torch.rand(n, generator=g, device=dev) < cfg.daug_transient_p) & (env.amax(1) > 0)
        y = add_noise_snr(y, _uniform(n, *cfg.daug_transient_snr_db, g, dev), tsel, burst)
    if cfg.daug_floor_p > 0:
        if floor is not None:
            nz = torch.fft.irfft(torch.fft.rfft(torch.randn(n, L, generator=g, device=dev), dim=1)
                                 * torch.sqrt(floor.clamp_min(0))[None, :], n=L, dim=1)
        else:
            nz = band_noise(n, L, g, dev, pink=True)
        fsel = torch.rand(n, generator=g, device=dev) < cfg.daug_floor_p
        y = add_noise_snr(y, _uniform(n, *cfg.daug_floor_snr_db, g, dev), fsel, nz)
    out = x.clone(); out[sel] = y.to(x.dtype)
    return out

def augment(x, g, daug_g=None, floor=None):
    B, L = x.shape; dev = x.device
    if cfg.aug_shift_p > 0:
        do = (torch.rand(B, generator=g, device=dev) < cfg.aug_shift_p).long()
        sh = (torch.rand(B, generator=g, device=dev) * L).long() * do
        x = torch.gather(x, 1, (torch.arange(L, device=dev)[None, :] + sh[:, None]) % L)
    if daug_g is not None:
        x = daug(x, daug_g, floor)
    if cfg.aug_noise_p > 0:
        sel = torch.rand(B, generator=g, device=dev) < cfg.aug_noise_p
        x = add_noise_snr(x, _uniform(B, *cfg.aug_noise_snr_db, g, dev), sel, band_noise(B, L, g, dev))
    return x

def bce_np(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def rec_scores(idx, p):
    idx = np.asarray(idx)
    return (pd.DataFrame(dict(rec=REC_ALL[idx], y=Y_ALL[idx], p=np.asarray(p, float)))
            .groupby("rec").agg(y=("y", "first"), p=("p", "mean")))

def record_auc(idx, p):
    if len(idx) == 0:
        return np.nan
    r = rec_scores(idx, p)
    return fast_auc(r.y.values, r.p.values)

@torch.no_grad()
def predict(model, idx, device, dom=None, return_emb=False):
    idx = np.asarray(idx, dtype=np.int64)
    if len(idx) == 0:
        z = np.zeros(0, np.float32)
        return (z, np.zeros((0, cfg.emb_dim), np.float32)) if return_emb else z
    model.eval(); P, E = [], []
    for s in range(0, len(idx), cfg.eval_batch):
        ib = idx[s:s + cfg.eval_batch]
        xb = fetch(device, ib)
        db = torch.as_tensor(DOM_ALL[ib] if dom is None else np.full(len(ib), dom), device=device)
        with amp_autocast(device, use_amp(device)):
            f = model.G_f(xb, db); lg = model.G_y(f)
        P.append(torch.sigmoid(lg.float()).nan_to_num(0.5).cpu().numpy())
        if return_emb:
            E.append(f.float().cpu().numpy())
    p = np.concatenate(P)
    return (p, np.concatenate(E)) if return_emb else p

@torch.no_grad()
def fit_cmvn(model, idx, dom, device):
    G = model.G_f
    if not hasattr(G, "cmvn") or len(idx) == 0:
        return
    s1 = s2 = None; n = 0
    for s in range(0, len(idx), cfg.eval_batch):
        S = G.tf_raw(fetch(device, idx[s:s + cfg.eval_batch])).float()
        a, b = S.sum(dim=(0, 1, 3)), (S ** 2).sum(dim=(0, 1, 3))
        s1 = a if s1 is None else s1 + a
        s2 = b if s2 is None else s2 + b
        n += S.shape[0] * S.shape[3]
    mu = s1 / n
    G.cmvn.set_stats(dom, mu, (s2 / n - mu ** 2).clamp_min(1e-6).sqrt())

@torch.no_grad()
def copy_cmvn(model, src_dom, dst_dom):
    G = model.G_f
    if hasattr(G, "cmvn"):
        G.cmvn.mu[dst_dom] = G.cmvn.mu[src_dom]; G.cmvn.sd[dst_dom] = G.cmvn.sd[src_dom]

@torch.no_grad()
def adabn_(model, idx, device, dom=1, seed=0):
    bns = [m for m in model.G_f.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    idx = np.random.default_rng(seed).permutation(np.asarray(idx))
    if not bns or len(idx) < 2:
        return 0
    mom = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats(); m.momentum = None
    model.eval()
    for m in bns:
        m.train()
    for s in range(0, len(idx), cfg.eval_batch):
        ib = idx[s:s + cfg.eval_batch]
        if len(ib) < 2:
            continue
        model.G_f(fetch(device, ib), torch.full((len(ib),), dom, dtype=torch.long, device=device))
    for m, mo in zip(bns, mom):
        m.momentum = mo
    model.eval()
    return len(bns)

def labels_for(idx, y_view):
    y = np.asarray(y_view, np.float32)[np.asarray(idx)]
    assert len(y) and np.isfinite(y).all() and np.isin(y, (0.0, 1.0)).all(), \
        "Tập huấn luyện có nhãn chứa nhãn ẨN/không hợp lệ - kiểm tra phân chia!"
    return y

def reveal_target_labels(idx):
    v = np.full(len(Y_ALL), np.nan, np.float32); v[np.asarray(idx)] = Y_ALL[np.asarray(idx)]
    return v

def pos_weight(y):
    pos = float(np.sum(y)); neg = float(len(y) - pos)
    return float(np.clip(neg / max(pos, 1.0), 0.1, 10.0))

def make_model(backbone, seed, device):
    with _INIT_LOCK:
        set_seed(seed)
        m = DANNNet(backbone)
    return m.to(device)

def make_sched(opt, total_steps):
    warm = max(1, int(cfg.warmup_frac * total_steps))
    def f(step):
        if step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, total_steps - warm))
        return 1e-3 + (1 - 1e-3) * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)

def _job_seed(job):
    k = job["fold"]
    return int(job["seed"]) * 1000 + (97 if k is None else int(k)) * 13 + {"CNN2D": 1, "CNN1D": 2}.get(job["backbone"], 3)

def _fold_of(job):
    sp = SPLITS[job["seed"]]
    return sp["final"] if job["fold"] is None else sp["folds"][job["fold"]]

MON_KEYS = ("src_val_auc", "src_val_loss", "tgt_auc_monitor", "tgt_rec_auc_monitor")
def monitor(model, sp, fd, device, info):
    out = {}
    if info.get("src") and len(sp["src_val_idx"]):
        i = sp["src_val_idx"]; p = predict(model, i, device)
        out["src_val_auc"], out["src_val_loss"] = fast_auc(Y_ALL[i], p), bce_np(Y_ALL[i], p)
    ti = fd["t_test_idx"]
    if len(ti) and (Y_ALL[ti] >= 0).all():
        p = predict(model, ti, device)        # CHỈ để vẽ đường học
        out["tgt_auc_monitor"], out["tgt_rec_auc_monitor"] = fast_auc(Y_ALL[ti], p), record_auc(ti, p)
    return out

print("Tiện ích huấn luyện: OK | DAug dịch tần:", DAUG_WARP_ON)
''')

code(r'''
# =========================================================
# 8.2 Huấn luyện một job: Source-only / CORAL / MMD / DANN / CDAN / +DAug / Few-shot chỉ đích / Oracle
# =========================================================
def train_da(job, device):
    device = torch.device(device)
    seed, bb, method = int(job["seed"]), job["backbone"], job["method"]
    info = METHOD_INFO[method]; sp = SPLITS[seed]; fd = _fold_of(job)
    js = _job_seed(job)
    model = make_model(bb, js, device)
    set_specaug_generators(model, torch.Generator().manual_seed(js + 4))
    g_src, g_tgt, g_aug = (make_generator(device, js + o) for o in (1, 2, 3))
    rng_src, rng_tgt = np.random.default_rng(js + 5), np.random.default_rng(js + 6)

    # ---- tập CÓ NHÃN và tập KHÔNG NHÃN ----
    if info["label_dom"] == 1:
        lab_idx = np.asarray(fd["t_fewshot_idx"] if info["label_set"] == "fewshot" else fd["t_adapt_lab_idx"])
        y_lab = labels_for(lab_idx, reveal_target_labels(lab_idx))
    else:
        lab_idx = np.asarray(sp["src_train_idx"]); y_lab = labels_for(lab_idx, Y_SRC_VISIBLE)
        assert (DOM_ALL[lab_idx] == 0).all()
    assert len(lab_idx) >= 2
    tgt_idx = np.asarray(fd["t_adapt_idx"]) if info["tgt"] else None
    if tgt_idx is not None:
        assert len(tgt_idx) > 1 and (DOM_ALL[tgt_idx] == 1).all()
        assert np.intersect1d(np.unique(REC_ALL[tgt_idx]), np.unique(REC_ALL[fd["t_test_idx"]])).size == 0

    # ---- CMVN theo miền ----
    if info["label_dom"] == 1:
        fit_cmvn(model, np.asarray(fd["t_adapt_idx"]), 1, device); copy_cmvn(model, 1, 0)
    else:
        fit_cmvn(model, lab_idx, 0, device)
        if tgt_idx is not None:
            fit_cmvn(model, tgt_idx, 1, device)
        else:
            copy_cmvn(model, 0, 1)                      # Source-only: nhìn đích bằng thống kê nguồn
    floor = noise_floor_spectrum(tgt_idx, device, seed=js) if (info["daug"] and tgt_idx is not None) else None

    params = list(model.G_f.parameters()) + list(model.G_y.parameters())
    if info["dann"]:
        params += list(model.G_d.parameters())
    if info["cdan"]:
        params += list(model.G_dc.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    bs = int(cfg.batch_size)
    balanced = bool(cfg.balanced_source_batches and info["label_dom"] == 0)
    spe = int(math.ceil(len(lab_idx) / bs))
    if info["label_dom"] == 1:
        spe = max(spe, int(cfg.min_steps_per_epoch))
    if cfg.max_steps_per_epoch:
        spe = min(spe, int(cfg.max_steps_per_epoch))
    spe = max(1, spe)
    total = spe * int(cfg.epochs)
    sched = make_sched(opt, total)
    amp_on = use_amp(device); scaler = make_grad_scaler(amp_on)
    pw = torch.tensor(1.0 if balanced else pos_weight(y_lab), device=device, dtype=torch.float32)
    cls_pos = [np.where(y_lab == c)[0] for c in (0, 1)]
    eps = float(cfg.label_smoothing)
    hist = defaultdict(list); step = 0; n_skip = 0; t0 = time.time(); lam = 0.0

    for ep in range(int(cfg.epochs)):
        model.train()
        perm = rng_src.permutation(len(lab_idx))
        logs = defaultdict(list)
        for s in range(spe):
            if balanced:
                h = bs // 2
                pos = np.concatenate([rng_src.choice(cls_pos[0], h, replace=len(cls_pos[0]) < h),
                                      rng_src.choice(cls_pos[1], bs - h, replace=len(cls_pos[1]) < bs - h)])
            else:
                pos = perm[np.arange(s * bs, (s + 1) * bs) % len(perm)]
                if len(np.unique(pos)) < 2:
                    pos = np.concatenate([pos, rng_src.integers(0, len(lab_idx), 2)])
            ns = len(pos)
            xs = augment(fetch(device, lab_idx[pos]), g_src, daug_g=(g_aug if info["daug"] else None), floor=floor)
            ys = torch.as_tensor(y_lab[pos], device=device)
            prog = step / max(1, total - 1)
            lam = grl_schedule(prog); ramp = lam if cfg.ramp_alignment else 1.0
            if tgt_idx is not None:
                it = rng_tgt.choice(tgt_idx, size=ns, replace=len(tgt_idx) < ns)
                x = torch.cat([xs, augment(fetch(device, it), g_tgt)], 0)
                dom = torch.cat([torch.zeros(ns, dtype=torch.long, device=device),
                                 torch.ones(ns, dtype=torch.long, device=device)])
            else:
                x = xs; dom = torch.full((ns,), int(info["label_dom"]), dtype=torch.long, device=device)
            with amp_autocast(device, amp_on):
                f = model.G_f(x, dom)
                fs = f[:ns]
                logit_all = model.G_y(f).float() if info["cdan"] else None
                logit_s = logit_all[:ns] if logit_all is not None else model.G_y(fs).float()
                l_y = F.binary_cross_entropy_with_logits(logit_s, ys * (1 - eps) + eps / 2, pos_weight=pw)
                loss = l_y
                if info["dann"] or info["cdan"]:
                    model.grl.lambd = float(cfg.lambda_dann) * lam
                    if info["cdan"]:
                        p_all = torch.sigmoid(logit_all)
                        dl = model.G_dc(model.grl(cdan_features(f, p_all))).float()
                        ws, wt = (entropy_weights(p_all, ns) if cfg.cdan_entropy
                                  else (torch.ones(ns, device=device), torch.ones(len(dl) - ns, device=device)))
                    else:
                        dl = model.G_d(model.grl(f)).float()
                        ws, wt = torch.ones(ns, device=device), torch.ones(len(dl) - ns, device=device)
                    l_d = (F.binary_cross_entropy_with_logits(dl[:ns], torch.zeros_like(dl[:ns]), weight=ws) +
                           F.binary_cross_entropy_with_logits(dl[ns:], torch.ones_like(dl[ns:]), weight=wt))
                    loss = loss + l_d
                    logs["l_d"].append(l_d.detach())
                    logs["acc_d"].append(0.5 * ((dl[:ns] < 0).float().mean() + (dl[ns:] > 0).float().mean()))
                if tgt_idx is not None:
                    ft = f[ns:]
                    if info["coral"]:
                        l_c = coral_loss(fs, ft); loss = loss + cfg.coral_weight * ramp * l_c
                    else:
                        with torch.no_grad():
                            l_c = coral_loss(fs.detach(), ft.detach())
                    logs["l_coral"].append(l_c.detach())
                    if info["mmd"]:
                        l_m = mmd_rbf(fs, ft); loss = loss + cfg.mmd_weight * ramp * l_m
                        logs["l_mmd"].append(l_m.detach())
            if not torch.isfinite(loss):
                n_skip += 1; opt.zero_grad(set_to_none=True); step += 1
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            logs["l_y"].append(l_y.detach()); logs["loss"].append(loss.detach())

        def _m(k):
            v = logs.get(k)
            return float(torch.stack([t.float() for t in v]).mean()) if v else np.nan
        hist["epoch"].append(ep); hist["lr"].append(opt.param_groups[0]["lr"]); hist["lam"].append(lam)
        for k in ("loss", "l_y", "l_d", "acc_d", "l_coral", "l_mmd"):
            hist[k].append(_m(k))
        mon = monitor(model, sp, fd, device, info)
        for k in MON_KEYS:
            hist[k].append(mon.get(k, np.nan))
        if cfg.verbose_epochs and (ep % max(1, cfg.epochs // 6) == 0 or ep == cfg.epochs - 1):
            print(f"    [{bb} {method} f{job['fold']}] ep{ep:02d} L_y {hist['l_y'][-1]:.3f} L_d {hist['l_d'][-1]:.3f} "
                  f"acc_d {hist['acc_d'][-1]:.2f} CORAL {hist['l_coral'][-1]:.4f} | AUC nguồn-val "
                  f"{hist['src_val_auc'][-1]:.3f} | đích (monitor) {hist['tgt_auc_monitor'][-1]:.3f}")

    p_test = predict(model, fd["t_test_idx"], device)
    sv = sp["src_val_idx"] if info["src"] else np.zeros(0, np.int64)
    p_sv = predict(model, sv, device)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out = dict(job=dict(job), hist=pd.DataFrame(hist), state=state, test_idx=np.asarray(fd["t_test_idx"]),
               p_test=p_test, srcval_idx=np.asarray(sv), p_srcval=p_sv, train_time_s=time.time() - t0,
               n_skip=n_skip, n_labeled=len(lab_idx), n_unlabeled=0 if tgt_idx is None else len(tgt_idx),
               steps=total)
    del model, opt; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out
''')

code(r'''
# =========================================================
# 8.3 Phương pháp dựng sau: AdaBN và Few-shot fine-tuning chọn lọc theo tầng
# =========================================================
def run_post(job, device):
    device = torch.device(device)
    seed, bb, method = int(job["seed"]), job["backbone"], job["method"]
    info = METHOD_INFO[method]; sp = SPLITS[seed]; fd = _fold_of(job)
    base = RESULTS[(seed, bb, info["base"], job["fold"])]
    js = _job_seed(job)
    model = make_model(bb, js, device); model.load_state_dict(base["state"])
    t0 = time.time(); hist = defaultdict(list)
    fit_cmvn(model, fd["t_adapt_idx"], 1, device)
    adabn_(model, fd["t_adapt_idx"], device, dom=1, seed=js)
    if info.get("finetune"):
        fs_idx = np.asarray(fd["t_fewshot_idx"])
        y_fs = labels_for(fs_idx, reveal_target_labels(fs_idx))
        gr = model.G_f.ft_groups()
        opt = torch.optim.AdamW([dict(params=gr["early"], lr=cfg.ft_lr_early),
                                 dict(params=gr["middle"], lr=cfg.ft_lr_mid),
                                 dict(params=gr["head"] + list(model.G_y.parameters()), lr=cfg.ft_lr_head)],
                                weight_decay=cfg.weight_decay)
        bns = [m for m in model.G_f.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        g = make_generator(device, js + 11); rng = np.random.default_rng(js + 12)
        set_specaug_generators(model, torch.Generator().manual_seed(js + 13))
        pw = torch.tensor(pos_weight(y_fs), device=device, dtype=torch.float32)
        amp_on = use_amp(device); scaler = make_grad_scaler(amp_on)
        bs = min(int(cfg.batch_size), max(2, len(fs_idx)))
        spe = max(int(cfg.min_steps_per_epoch), int(math.ceil(len(fs_idx) / bs)))
        for ep in range(int(cfg.ft_epochs)):
            model.train()
            for m in bns:
                m.eval()                          # giữ thống kê AdaBN
            ls = []
            for _ in range(spe):
                pos = rng.integers(0, len(fs_idx), bs)
                xb = augment(fetch(device, fs_idx[pos]), g)
                yb = torch.as_tensor(y_fs[pos], device=device)
                with amp_autocast(device, amp_on):
                    lg = model(xb, torch.ones(len(pos), dtype=torch.long, device=device)).float()
                    loss = F.binary_cross_entropy_with_logits(lg, yb * (1 - cfg.label_smoothing) + cfg.label_smoothing / 2,
                                                              pos_weight=pw)
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt); scaler.update(); ls.append(float(loss))
            hist["epoch"].append(ep); hist["l_y"].append(np.mean(ls) if ls else np.nan)
            mon = monitor(model, sp, fd, device, dict(src=False))
            for k in MON_KEYS:
                hist[k].append(mon.get(k, np.nan))
    p_test = predict(model, fd["t_test_idx"], device)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out = dict(job=dict(job), hist=pd.DataFrame(hist), state=state, test_idx=np.asarray(fd["t_test_idx"]),
               p_test=p_test, srcval_idx=np.zeros(0, np.int64), p_srcval=np.zeros(0, np.float32),
               train_time_s=time.time() - t0, n_skip=0,
               n_labeled=len(fd["t_fewshot_idx"]) if info.get("finetune") else 0,
               n_unlabeled=len(fd["t_adapt_idx"]), steps=0)
    del model; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out
''')

code(r'''
# =========================================================
# 8.4 Hàng đợi job trên nhiều GPU / Multi-GPU job queue
# =========================================================
def run_jobs(jobs, fn, title):
    devs = ([torch.device(f"cuda:{i}") for i in range(min(N_GPU, cfg.max_parallel))]
            if USE_CUDA else [torch.device("cpu")])
    if not cfg.parallel_jobs:
        devs = devs[:1]
    out, t0, n = {}, time.time(), len(jobs)
    key = lambda j: (j["seed"], j["backbone"], j["method"], j["fold"])
    def report(i, j, r, d):
        ti = r["test_idx"]
        a = fast_auc(Y_ALL[ti], r["p_test"]) if len(ti) else np.nan
        ra = record_auc(ti, r["p_test"]) if len(ti) else np.nan
        print(f"  [{i:>3}/{n}] {j['backbone']:<6} {j['method']:<16} fold {str(j['fold']):>4} | "
              f"AUC {cfg.target_name}: frame {a:.3f}, bản ghi {ra:.3f} | {r['train_time_s']:5.0f}s @ {d}")
    if len(devs) == 1:
        for i, j in enumerate(jobs, 1):
            r = fn(j, devs[0]); out[key(j)] = r; report(i, j, r, devs[0])
    else:
        dq = queue.Queue()
        for d in devs:
            dq.put(d)
        def work(j):
            d = dq.get()
            try:
                return fn(j, d), d
            finally:
                dq.put(d)
        with ThreadPoolExecutor(max_workers=len(devs)) as ex:
            futs = {ex.submit(work, j): j for j in jobs}
            for i, fu in enumerate(as_completed(futs), 1):
                j = futs[fu]; r, d = fu.result(); out[key(j)] = r; report(i, j, r, d)
    print(f"--> {title}: {n} job trong {(time.time() - t0) / 60:.1f} phút trên {len(devs)} thiết bị")
    return out

FINAL_MODELS = {}
def get_final(method, backbone=None):
    """Mô hình CUỐI: toàn bộ nguồn train + TOÀN BỘ đích (không nhãn) - để xuất và xếp hạng."""
    backbone = backbone or cfg.main_backbone
    if (backbone, method) not in FINAL_MODELS:
        FINAL_MODELS[(backbone, method)] = train_da(dict(seed=cfg.seeds[0], backbone=backbone, method=method,
                                                         fold=None), DEVICE)
    return FINAL_MODELS[(backbone, method)]
print("Hàng đợi job: OK |", "song song" if (cfg.parallel_jobs and N_GPU > 1) else "tuần tự")
''')

# =============================================================================
# 9. Chạy thí nghiệm
# =============================================================================
md(r'''
## 9. Chạy toàn bộ thí nghiệm / Run the experiment matrix

`len(seeds) × len(backbones) × |phương pháp huấn luyện| × n_fold` job, sau đó các job dựng sau (AdaBN, Few-shot FT).
Nếu «T» không có nhãn (`EVAL_ON = False`), mục này được bỏ qua và notebook chuyển sang mô hình cuối (mục 12).
''')

code(r'''
# =========================================================
# 9.1 Huấn luyện toàn bộ / Train everything
# =========================================================
RESULTS = {}
if EVAL_ON:
    _tm = [m for m in TRAIN_METHODS if not METHOD_INFO[m]["needs_target_labels"] or TARGET_HAS_LABELS]
    jobs = [dict(seed=s, backbone=b, method=m, fold=fd["fold"])
            for s in cfg.seeds for b in cfg.backbones for m in _tm for fd in SPLITS[s]["folds"]]
    print(f"{len(jobs)} job huấn luyện ({len(cfg.seeds)} seed × {len(cfg.backbones)} kiến trúc × {len(_tm)} phương pháp "
          f"× {len(sp0['folds'])} fold), {cfg.epochs} epoch mỗi job")
    RESULTS.update(run_jobs(jobs, train_da, "huấn luyện"))
    pjobs = [dict(seed=s, backbone=b, method=m, fold=fd["fold"])
             for s in cfg.seeds for b in cfg.backbones for m in POST_METHODS for fd in SPLITS[s]["folds"]
             if (s, b, METHOD_INFO[m]["base"], fd["fold"]) in RESULTS]
    if pjobs:
        RESULTS.update(run_jobs(pjobs, run_post, "AdaBN / few-shot"))
    _log = pd.DataFrame([dict(seed=k[0], backbone=k[1], method=k[2], fold=k[3], time_s=round(r["train_time_s"], 1),
                              steps=r["steps"], n_labeled=r["n_labeled"], n_unlabeled=r["n_unlabeled"], n_skip=r["n_skip"],
                              frame_auc=fast_auc(Y_ALL[r["test_idx"]], r["p_test"]),
                              record_auc=record_auc(r["test_idx"], r["p_test"]))
                         for k, r in RESULTS.items()])
    _log.to_csv(TAB_DIR / "job_log.csv", index=False)
    print(f"Tổng thời gian huấn luyện: {_log.time_s.sum() / 60:.1f} phút-thiết bị")
else:
    print(f"!! EVAL_ON = False: {cfg.target_name} không đủ nhãn để đánh giá -> bỏ qua ma trận thí nghiệm.")
''')

# =============================================================================
# 10. Kết quả
# =============================================================================
md(r'''
## 10. Kết quả trên miền đích / Results on the target domain

- **Chỉ số chính:** AUC **mức bản ghi** trên «T» kiểm thử (gộp `p_mean`); dự đoán gộp qua các fold, trung bình qua seed.
- **CI 95 %:** bootstrap **theo cụm bản ghi**. **So sánh với Source-only:** bootstrap **ghép cặp** của ΔAUC, Holm trong từng kiến trúc.
- **Khoảng cách đã thu hẹp:** $\frac{AUC_m - AUC_{src}}{AUC_{oracle} - AUC_{src}}$ (chỉ khi cận dưới CI của Oracle vượt Source-only).
- **Giá trị của «S» (Q3):** Few-shot FT (từ «S») so với Few-shot **chỉ «T»** trên **cùng** bản ghi few-shot — ΔAUC ghép cặp.
- **Lệch tỉ lệ lớp:** tỉ lệ rò của «T» ước lượng bằng **BBSE** (Lipton et al., 2018) từ TPR/FPR trên «S»-val và tỉ lệ dự
  đoán dương trên «T» — không dùng nhãn «T»; so với tỉ lệ thật (nếu có nhãn) để chẩn đoán.
''')

code(r'''
# =========================================================
# 10.1 Bảng kết quả + CI bootstrap theo cụm + kiểm định ghép cặp (Holm)
# =========================================================
def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (n - r) * p[i]); adj[i] = min(1.0, run)
    return adj

def oof_preds(bb, method):
    acc = defaultdict(list)
    for s in cfg.seeds:
        for fd in SPLITS[s]["folds"]:
            r = RESULTS.get((s, bb, method, fd["fold"]))
            if r is not None:
                for i, p in zip(r["test_idx"], r["p_test"]):
                    acc[int(i)].append(float(p))
    if not acc:
        return None, None
    idx = np.array(sorted(acc), dtype=np.int64)
    return idx, np.array([np.mean(acc[i]) for i in idx], np.float64)

def eval_preds(idx, p, thr=None):
    thr = cfg.decision_threshold if thr is None else thr
    y = Y_ALL[idx]; yh = (p >= thr).astype(int); r = rec_scores(idx, p)
    return dict(n_frames=len(idx), n_records=len(r), frame_auc=fast_auc(y, p), record_auc=fast_auc(r.y, r.p),
                frame_ap=average_precision_score(y, p), frame_bal_acc=balanced_accuracy_score(y, yh),
                frame_f1=f1_score(y, yh, zero_division=0),
                record_bal_acc=balanced_accuracy_score(r.y.astype(int), (r.p >= thr).astype(int)))

def cluster_bootstrap(idx, P_list, n_boot=None, seed=0):
    n_boot = cfg.bootstrap_n if n_boot is None else n_boot
    y = Y_ALL[idx]
    ur, inv = np.unique(REC_ALL[idx], return_inverse=True); R = len(ur)
    order = np.argsort(inv, kind="stable")
    pos = np.split(order, np.cumsum(np.bincount(inv, minlength=R))[:-1])
    yr = np.array([y[p[0]] for p in pos])
    pr = [np.array([P[p].mean() for p in pos]) for P in P_list]
    rng = np.random.default_rng(seed)
    BF = np.full((n_boot, len(P_list)), np.nan); BR = BF.copy()
    for b in range(n_boot):
        pick = rng.integers(0, R, R)
        if len(np.unique(yr[pick])) < 2:
            continue
        sel = np.concatenate([pos[j] for j in pick])
        for m, P in enumerate(P_list):
            BF[b, m] = fast_auc(y[sel], P[sel]); BR[b, m] = fast_auc(yr[pick], pr[m][pick])
    return BF, BR

def paired_p(d):
    d = d[np.isfinite(d)]
    return (max(1.0 / len(d), min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))) if len(d) else np.nan)

RES = pd.DataFrame(); BOOT_STORE = {}
if EVAL_ON:
    rows = []
    for bb in cfg.backbones:
        pr = {m: oof_preds(bb, m) for m in cfg.methods}
        pr = {m: v for m, v in pr.items() if v[0] is not None}
        if "source_only" not in pr:
            continue
        common = reduce(np.intersect1d, [v[0] for v in pr.values()])
        names = list(pr)
        P = [pr[m][1][np.searchsorted(pr[m][0], common)] for m in names]
        BF, BR = cluster_bootstrap(common, P, seed=cfg.seed)
        BOOT_STORE[bb] = dict(idx=common, names=names, P=P, BF=BF, BR=BR)
        i0 = names.index("source_only")
        ev = [eval_preds(common, p) for p in P]
        a_src = ev[i0]["record_auc"]
        a_or = ev[names.index("oracle")]["record_auc"] if "oracle" in names else np.nan
        # "khoảng cách đã thu hẹp" chỉ có nghĩa khi Oracle vượt Source-only RÕ RỆT (cận dưới CI của Oracle > Source-only)
        _gap_ok = bool("oracle" in names and a_or - a_src >= 0.05
                       and np.nanpercentile(BR[:, names.index("oracle")], 2.5) > a_src)
        praw = {m: paired_p(BR[:, j] - BR[:, i0]) for j, m in enumerate(names) if j != i0}
        _ms = [m for m in praw if np.isfinite(praw[m])]
        padj = dict(zip(_ms, holm([praw[m] for m in _ms]))) if _ms else {}
        for j, m in enumerate(names):
            e = ev[j]
            cf = np.nanpercentile(BF[:, j], [2.5, 97.5]); cr = np.nanpercentile(BR[:, j], [2.5, 97.5])
            dd = np.nanpercentile(BR[:, j] - BR[:, i0], [2.5, 97.5]) if j != i0 else (np.nan, np.nan)
            fold_auc = [record_auc(r["test_idx"], r["p_test"]) for k, r in RESULTS.items()
                        if k[1] == bb and k[2] == m and k[3] is not None]
            sv = [fast_auc(Y_ALL[r["srcval_idx"]], r["p_srcval"]) for k, r in RESULTS.items()
                  if k[1] == bb and k[2] == m and len(r["srcval_idx"])]
            gap = (100 * (e["record_auc"] - a_src) / (a_or - a_src)
                   if (_gap_ok and m not in ("oracle", "source_only")) else np.nan)
            rows.append(dict(backbone=bb, method=m, label=METHOD_LABEL[m], **e,
                             frame_auc_lo=cf[0], frame_auc_hi=cf[1], record_auc_lo=cr[0], record_auc_hi=cr[1],
                             delta_rec_auc=e["record_auc"] - a_src, delta_lo=dd[0], delta_hi=dd[1],
                             p_raw=praw.get(m, np.nan), p_holm=padj.get(m, np.nan), gap_closed_pct=gap,
                             fold_rec_auc_mean=np.nanmean(fold_auc) if fold_auc else np.nan,
                             fold_rec_auc_sd=np.nanstd(fold_auc) if fold_auc else np.nan,
                             src_val_auc=np.nanmean(sv) if sv else np.nan))
    RES = pd.DataFrame(rows)
    RES.to_csv(TAB_DIR / "table6_target_results_full.csv", index=False)

    def _ci(a, lo, hi):
        return f"{a:.3f} [{lo:.3f}, {hi:.3f}]" if np.isfinite(a) else "—"
    show = pd.DataFrame(dict(
        Backbone=RES.backbone, Method=RES.label,
        **{"Record AUC [95% CI]": [_ci(a, l, h) for a, l, h in zip(RES.record_auc, RES.record_auc_lo, RES.record_auc_hi)],
           "Frame AUC [95% CI]": [_ci(a, l, h) for a, l, h in zip(RES.frame_auc, RES.frame_auc_lo, RES.frame_auc_hi)],
           "ΔAUC vs Source-only [95% CI]": [("—" if not np.isfinite(l) else f"{d:+.3f} [{l:+.3f}, {h:+.3f}]")
                                            for d, l, h in zip(RES.delta_rec_auc, RES.delta_lo, RES.delta_hi)],
           "p (Holm)": [("—" if not np.isfinite(p) else f"{p:.3f}") for p in RES.p_holm],
           "Gap closed %": [("—" if not np.isfinite(g) else f"{g:.0f}") for g in RES.gap_closed_pct],
           "Bal.Acc@0.5 (frame)": RES.frame_bal_acc.round(3), "F1@0.5": RES.frame_f1.round(3),
           "Fold rec-AUC mean±sd": [f"{m:.3f} ± {s:.3f}" for m, s in zip(RES.fold_rec_auc_mean, RES.fold_rec_auc_sd)],
           f"{cfg.source_name}-val AUC": RES.src_val_auc.round(3)}))
    display(Markdown(f"### Table 6. {cfg.source_name} → {cfg.target_name} transfer: target-domain performance "
                     f"({int(RES.n_records.iloc[0])} bản ghi kiểm thử, {len(cfg.seeds)} seed)"))
    display(show)
    show.to_csv(TAB_DIR / "table6_target_results.csv", index=False)
    try:
        (TAB_DIR / "table6_target_results.tex").write_text(
            show.to_latex(index=False, escape=True, caption=f"{cfg.source_name} to {cfg.target_name} transfer",
                          label="tab:da_results"), encoding="utf-8")
    except Exception as e:
        print(f"  (bỏ qua LaTeX: {type(e).__name__})")
else:
    print("EVAL_ON = False -> không có bảng kết quả miền đích.")
''')

code(r'''
# =========================================================
# 10.2 Giá trị của dữ liệu nguồn (Q3) + ước lượng tỉ lệ rò của đích bằng BBSE
# =========================================================
VALUE_TAB, BBSE_TAB = pd.DataFrame(), pd.DataFrame()
if EVAL_ON and BOOT_STORE:
    rows = []
    pairs = [("fewshot_src", "fewshot_scratch", f"FT từ Source-only vs chỉ {cfg.target_name}"),
             ("fewshot_da", "fewshot_scratch", f"FT từ DANN+CORAL vs chỉ {cfg.target_name}"),
             ("fewshot_da", "fewshot_src", "FT từ DANN+CORAL vs FT từ Source-only"),
             ("oracle", "source_only", f"Oracle vs Source-only (khoảng cách {cfg.source_name}→{cfg.target_name})")]
    for bb, B in BOOT_STORE.items():
        prs = [(a, b, t) for a, b, t in pairs if a in B["names"] and b in B["names"]]
        ps = []
        for a, b, t in prs:
            ia, ib = B["names"].index(a), B["names"].index(b)
            d = B["BR"][:, ia] - B["BR"][:, ib]
            ra = RES[(RES.backbone == bb) & (RES.method == a)].record_auc.iloc[0]
            rb = RES[(RES.backbone == bb) & (RES.method == b)].record_auc.iloc[0]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            ps.append(paired_p(d))
            rows.append(dict(backbone=bb, comparison=t, auc_a=ra, auc_b=rb, delta=ra - rb, delta_lo=lo, delta_hi=hi, p_raw=ps[-1]))
        if ps:
            for r, pa in zip(rows[-len(ps):], holm(ps)):
                r["p_holm"] = pa
    VALUE_TAB = pd.DataFrame(rows)
    if len(VALUE_TAB):
        display(Markdown(f"### Table 7. Giá trị của dữ liệu {cfg.source_name} khi đã có vài nhãn {cfg.target_name} "
                         f"(ΔAUC bản ghi, bootstrap ghép cặp, Holm)"))
        display(VALUE_TAB.round(3))
        VALUE_TAB.to_csv(TAB_DIR / "table7_value_of_source.csv", index=False)

    thr = cfg.decision_threshold; rows = []
    for bb in cfg.backbones:
        for m in [m for m in TRAIN_METHODS if METHOD_INFO[m]["src"]]:
            idx, p = oof_preds(bb, m)
            rs = [r for k, r in RESULTS.items() if k[1] == bb and k[2] == m and len(r["srcval_idx"])]
            if idx is None or not rs:
                continue
            tpr = np.mean([np.mean(r["p_srcval"][Y_ALL[r["srcval_idx"]] == 1] >= thr) for r in rs])
            fpr = np.mean([np.mean(r["p_srcval"][Y_ALL[r["srcval_idx"]] == 0] >= thr) for r in rs])
            mu = float(np.mean(p >= thr))
            pi = float(np.clip((mu - fpr) / (tpr - fpr), 0, 1)) if tpr - fpr > 0.1 else np.nan
            rows.append(dict(backbone=bb, method=METHOD_LABEL[m], TPR_src=tpr, FPR_src=fpr, pred_pos_rate_tgt=mu,
                             prior_bbse=pi, prior_true=float(Y_ALL[idx].mean()),
                             prior_src=float(Y_ALL[sp0["src_train_idx"]].mean())))
    BBSE_TAB = pd.DataFrame(rows)
    if len(BBSE_TAB):
        display(Markdown(f"### Table 8. Tỉ lệ frame rò của {cfg.target_name}: ước lượng BBSE (không nhãn) vs thật"))
        display(BBSE_TAB.round(3))
        BBSE_TAB.to_csv(TAB_DIR / "table8_bbse_prior.csv", index=False)
        _d = float(abs(BBSE_TAB.prior_true.iloc[0] - BBSE_TAB.prior_src.iloc[0]))
        print(f"Lệch tỉ lệ lớp nguồn/đích (frame): {_d:.2f}" +
              (" -> đáng kể: ưu tiên CDAN / lấy mẫu cân bằng; cẩn trọng với DANN căn chỉnh biên." if _d > 0.15 else ""))
''')

code(r'''
# =========================================================
# 10.3 Hình: forest plot AUC mức bản ghi trên miền đích
# =========================================================
if EVAL_ON and len(RES):
    bbs = list(RES.backbone.unique())
    _order = [m for m in cfg.methods if m in set(RES.method)]
    fig, axes = plt.subplots(1, len(bbs), figsize=(4.6 * len(bbs) + 2.6, 0.5 * len(_order) + 1.8), squeeze=False,
                             sharey=True, constrained_layout=True)
    ypos = {m: len(_order) - 1 - j for j, m in enumerate(_order)}
    for ia, (a, bb) in enumerate(zip(axes[0], bbs)):
        g = RES[RES.backbone == bb].reset_index(drop=True)
        for r in g.itertuples():
            y, c = ypos[r.method], METHOD_COLORS.get(r.method, "k")
            a.errorbar(r.record_auc, y, xerr=[[max(0, r.record_auc - r.record_auc_lo)], [max(0, r.record_auc_hi - r.record_auc)]],
                       fmt="o", color=c, capsize=3, ms=6)
            a.plot(r.frame_auc, y, marker="x", color=c, ls="none", ms=6)
        a.axvline(float(g.record_auc[g.method == "source_only"].iloc[0]), color=METHOD_COLORS["source_only"], ls="--", lw=1,
                  label="Source-only")
        if (g.method == "oracle").any():
            a.axvline(float(g.record_auc[g.method == "oracle"].iloc[0]), color="k", ls="-.", lw=1, label="Oracle")
        a.axvline(0.5, color="k", ls=":", lw=.6)
        if ia == 0:
            a.set_yticks([ypos[m] for m in _order]); a.set_yticklabels([METHOD_LABEL[m] for m in _order])
        a.set_xlabel(f"AUC trên {cfg.target_name} kiểm thử")
        a.set_xlim(max(0.0, min(RES.record_auc_lo.min(), RES.frame_auc.min()) - .05), 1.01)
        a.set_title(f"{bb}: ● bản ghi (CI 95%)  × frame", fontsize=10); a.legend(fontsize=7, loc="best")
    savefig(fig, "target_auc_forest",
            f"AUC mức bản ghi (●, CI 95% bootstrap theo bản ghi) và mức frame (×) trên {cfg.target_name} kiểm thử. "
            f"Nét đứt: Source-only; gạch chấm: Oracle.")
''')

code(r'''
# =========================================================
# 10.4 Hình: ROC trên miền đích (kiến trúc chính)
# =========================================================
if EVAL_ON and cfg.main_backbone in BOOT_STORE:
    B = BOOT_STORE[cfg.main_backbone]
    y = Y_ALL[B["idx"]]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for m, p in zip(B["names"], B["P"]):
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, color=METHOD_COLORS.get(m), lw=1.4, label=f"{METHOD_LABEL[m]} ({fast_auc(y, p):.3f})")
        r = rec_scores(B["idx"], p)
        fpr, tpr, _ = roc_curve(r.y.astype(int), r.p)
        ax[1].step(fpr, tpr, where="post", color=METHOD_COLORS.get(m), lw=1.4,
                   label=f"{METHOD_LABEL[m]} ({fast_auc(r.y, r.p):.3f})")
    for a, t in zip(ax, ("(a) mức frame", "(b) mức bản ghi (p_mean)")):
        a.plot([0, 1], [0, 1], "k:", lw=.7); a.set_xlabel("FPR"); a.set_ylabel("TPR"); a.set_title(t)
        a.legend(fontsize=7, loc="lower right")
    savefig(fig, "target_roc", f"ROC trên {cfg.target_name} kiểm thử, kiến trúc {cfg.main_backbone}.")
''')

code(r'''
# =========================================================
# 10.5 Hình: động học huấn luyện (đối kháng & căn chỉnh)
# =========================================================
if EVAL_ON:
    s0, bb = cfg.seeds[0], cfg.main_backbone
    show = [m for m in ("source_only", "coral", "mmd", "dann", "dann_coral", "cdan_coral", "dann_coral_aug")
            if (s0, bb, m, 0) in RESULTS]
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 7.8), constrained_layout=True); ax = ax.ravel()
    for m in show:
        h = RESULTS[(s0, bb, m, 0)]["hist"]; c = METHOD_COLORS[m]; lb = METHOD_LABEL[m]
        ax[0].plot(h.epoch, h.l_y, color=c, label=lb)
        if h.l_d.notna().any():
            ax[1].plot(h.epoch, h.l_d, color=c, label=lb)
            ax[2].plot(h.epoch, h.acc_d, color=c, label=lb)
        if h.l_coral.notna().any():
            ax[3].semilogy(h.epoch, h.l_coral.clip(lower=1e-8), color=c, ls="-" if METHOD_INFO[m]["coral"] else ":", label=lb)
        ax[4].plot(h.epoch, h.src_val_auc, color=c, label=lb)
        ax[5].plot(h.epoch, h.tgt_rec_auc_monitor, color=c, label=lb)
    ax[1].axhline(2 * np.log(2), color="k", ls=":", lw=.8, label="2·ln2 (đoán mò)")
    ax[2].axhline(0.5, color="k", ls=":", lw=.8, label="0,5 (không phân biệt được miền)")
    ttl = [f"(a) L_y trên {cfg.source_name}", "(b) L_adv (BCE miền, sau GRL)", "(c) độ chính xác bộ phân biệt miền",
           "(d) CORAL(f_S, f_T) theo lô\n(nét chấm: chỉ đo, không tối ưu)", f"(e) AUC {cfg.source_name}-val",
           f"(f) AUC bản ghi {cfg.target_name}\nCHỈ THEO DÕI - không dùng để chọn"]
    for a, t in zip(ax, ttl):
        a.set_title(t, fontsize=9.5); a.set_xlabel("epoch")
    ax[0].legend(fontsize=7)
    for a in (ax[1], ax[2]):
        if a.lines:
            a.legend(fontsize=7)
    savefig(fig, "training_dynamics",
            f"Động học huấn luyện ({bb}, fold 0). L_adv tiến về 2·ln2 và độ chính xác bộ phân biệt về 0,5 khi G_f tạo được "
            f"đặc trưng không phân biệt được miền; CORAL giảm khi hiệp phương sai hai miền khớp nhau.")
''')

code(r'''
# =========================================================
# 10.6 Hình: nguồn vs đích + cải thiện ghép cặp + giá trị của nguồn
# =========================================================
if EVAL_ON and len(RES):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4), constrained_layout=True)
    for bb, mk in zip(RES.backbone.unique(), ("o", "s", "^")):
        g = RES[(RES.backbone == bb) & RES.src_val_auc.notna()]
        for r in g.itertuples():
            ax[0].scatter(r.src_val_auc, r.record_auc, color=METHOD_COLORS.get(r.method), marker=mk, s=55, zorder=3)
            ax[0].annotate(r.method, (r.src_val_auc, r.record_auc), fontsize=7, xytext=(3, 3), textcoords="offset points")
    lims = [min(.4, np.nanmin(RES[["src_val_auc", "record_auc"]].values) - .02), 1.01]
    ax[0].plot(lims, lims, "k:", lw=.7); ax[0].set_xlim(lims); ax[0].set_ylim(lims)
    ax[0].set_xlabel(f"AUC {cfg.source_name}-val (frame)"); ax[0].set_ylabel(f"AUC bản ghi {cfg.target_name}")
    ax[0].set_title("(a) Nguồn vs đích (" + ", ".join(f"{m}={b}" for b, m in zip(RES.backbone.unique(), "o□△")) + ")")
    g = RES[RES.backbone == cfg.main_backbone]
    ax[1].barh(g.label, g.delta_rec_auc, color=[METHOD_COLORS.get(m) for m in g.method],
               xerr=[(g.delta_rec_auc - g.delta_lo).clip(lower=0).fillna(0), (g.delta_hi - g.delta_rec_auc).clip(lower=0).fillna(0)],
               capsize=3)
    ax[1].axvline(0, color="k", lw=.8); ax[1].invert_yaxis()
    ax[1].set_xlabel("ΔAUC bản ghi so với Source-only (CI 95% ghép cặp)"); ax[1].set_title(f"(b) {cfg.main_backbone}")
    if len(VALUE_TAB):
        v = VALUE_TAB[VALUE_TAB.backbone == cfg.main_backbone]
        ax[2].barh(v.comparison, v.delta, color=[PAL["acc"] if lo > 0 else PAL["tgt"] if hi < 0 else PAL["grey"]
                                                 for lo, hi in zip(v.delta_lo, v.delta_hi)],
                   xerr=[(v.delta - v.delta_lo).clip(lower=0), (v.delta_hi - v.delta).clip(lower=0)], capsize=3)
        ax[2].axvline(0, color="k", lw=.8); ax[2].invert_yaxis(); ax[2].tick_params(axis="y", labelsize=8)
        ax[2].set_xlabel("ΔAUC bản ghi (CI 95% ghép cặp)"); ax[2].set_title(f"(c) Giá trị của {cfg.source_name} (Q3)")
    else:
        ax[2].axis("off")
    savefig(fig, "source_vs_target", f"(a) Hiệu năng giữ trên {cfg.source_name} so với chuyển sang {cfg.target_name}; "
            f"(b) cải thiện ghép cặp so với Source-only; (c) {cfg.source_name} có thêm giá trị khi đã có vài nhãn {cfg.target_name}?")
''')

# =============================================================================
# 11. Không gian ẩn
# =============================================================================
md(r'''
## 11. Không gian đặc trưng bất biến miền / Domain-invariant latent space

So sánh không gian ẩn $f = G_f(x)$ **trước** (Source-only) và **sau** căn chỉnh, trên «S»-val và «T» kiểm thử.

| Chỉ số (không gian GỐC d chiều, chuẩn hoá chung) | Ý nghĩa | Mong muốn |
|---|---|---|
| **Proxy $\mathcal A$-distance** $=2(1-2\epsilon)$ | lỗi hồi quy logistic phân biệt miền (CV theo bản ghi, fold chia riêng trong từng miền) | → 0 |
| **CORAL** / **MMD²** | lệch hiệp phương sai / lệch phân phối | ↓ |
| **cos(hướng rò)** $\cos(\mu^S_1-\mu^S_0,\;\mu^T_1-\mu^T_0)$ | hướng leak − noleak giữ qua miền? (nhãn «T» chỉ để đánh giá) | → 1 |
| **AUC chuyển giao tuyến tính** | logistic học trên «S», thử trên «T» | ↑ |

> Căn chỉnh **biên** tốt (A-distance → 0) mà **cos thấp** = *negative transfer*: hai miền chồng lên nhau nhưng leak «T» nằm
> cạnh noleak «S» — đặc biệt dễ xảy ra khi tỉ lệ lớp khác nhau; CDAN được thêm vào để giảm rủi ro này.
''')

code(r'''
# =========================================================
# 11.1 Nhúng + chỉ số căn chỉnh miền
# =========================================================
def coral_np(a, b):
    return float(coral_loss(torch.from_numpy(a).float(), torch.from_numpy(b).float()))

def mmd_np(a, b, n_max=1500, seed=0):
    rng = np.random.default_rng(seed)
    a = a[rng.permutation(len(a))[:n_max]]; b = b[rng.permutation(len(b))[:n_max]]
    return float(mmd_rbf(torch.from_numpy(a).float(), torch.from_numpy(b).float()))

def domain_metrics(Es, Et, gs, gt, ys=None, yt=None):
    Z = np.vstack([Es, Et]); mu, sd = Z.mean(0), Z.std(0) + 1e-6
    Zs, Zt = (Es - mu) / sd, (Et - mu) / sd
    X = np.vstack([Zs, Zt]); d = np.r_[np.zeros(len(Zs)), np.ones(len(Zt))]
    us, ut = np.unique(gs), np.unique(gt)
    n_split = int(min(5, len(us), len(ut)))
    fold = np.r_[pd.Series(np.arange(len(us)) % max(n_split, 1), index=us).loc[gs].values,
                 pd.Series(np.arange(len(ut)) % max(n_split, 1), index=ut).loc[gt].values]
    errs = []
    if n_split >= 2:
        for k in range(n_split):
            te = fold == k; tr = ~te
            clf = LogisticRegression(max_iter=2000).fit(X[tr], d[tr])
            errs.append(float((clf.predict(X[te]) != d[te]).mean()))
    eps = float(np.mean(errs)) if errs else np.nan
    out = dict(proxy_A=2 * (1 - 2 * min(eps, 0.5)) if np.isfinite(eps) else np.nan, dom_clf_err=eps,
               coral=coral_np(Zs, Zt), mmd=mmd_np(Zs, Zt))
    if ys is not None and yt is not None and len(np.unique(ys)) == 2 and len(np.unique(yt)) == 2:
        vs = Zs[ys == 1].mean(0) - Zs[ys == 0].mean(0); vt = Zt[yt == 1].mean(0) - Zt[yt == 0].mean(0)
        out["cos_leak_dir"] = float(vs @ vt / (np.linalg.norm(vs) * np.linalg.norm(vt) + 1e-12))
        clf = LogisticRegression(max_iter=2000).fit(Zs, ys)
        out["lin_transfer_auc"] = fast_auc(yt, clf.decision_function(Zt))
    return out

def _subsample(idx, n, seed):
    idx = np.asarray(idx)
    return idx if len(idx) <= n else np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))

LATENT, LATENT_TABLE = {}, pd.DataFrame()
_cands = ("source_only", "coral", "mmd", "dann", "dann_coral", "cdan_coral", "dann_coral_aug")
if EVAL_ON:
    s0, bb = cfg.seeds[0], cfg.main_backbone
    fd0 = SPLITS[s0]["folds"][0]
    lat_src = {m: RESULTS[(s0, bb, m, 0)]["state"] for m in _cands if (s0, bb, m, 0) in RESULTS}
    i_s = _subsample(SPLITS[s0]["src_val_idx"], cfg.tsne_max_points // 2, 1)
    i_t = _subsample(fd0["t_test_idx"], cfg.tsne_max_points // 2, 2)
    lat_scope = f"{bb}, fold 0: {cfg.source_name}-val vs {cfg.target_name} kiểm thử"
else:
    bb = cfg.main_backbone
    lat_src = {m: get_final(m)["state"] for m in ("source_only", cfg.final_method)}
    i_s = _subsample(sp0["src_val_idx"], cfg.tsne_max_points // 2, 1)
    i_t = _subsample(np.where(DOM_ALL == 1)[0], cfg.tsne_max_points // 2, 2)
    lat_scope = f"{bb}, mô hình cuối: {cfg.source_name}-val vs toàn bộ {cfg.target_name} (không nhãn)"

rows = []
for m, st in lat_src.items():
    net = DANNNet(bb).to(DEVICE); net.load_state_dict(st)
    _, Es = predict(net, i_s, DEVICE, dom=0, return_emb=True)
    _, Et = predict(net, i_t, DEVICE, dom=1, return_emb=True)
    del net
    yt = Y_ALL[i_t] if (Y_ALL[i_t] >= 0).all() else None
    met = domain_metrics(Es, Et, REC_ALL[i_s], REC_ALL[i_t], Y_ALL[i_s], yt)
    LATENT[m] = dict(Es=Es, Et=Et, **met)
    rows.append(dict(method=METHOD_LABEL[m], **met))
LATENT_TABLE = pd.DataFrame(rows).round(4)
display(Markdown(f"### Table 9. Latent-space alignment ({lat_scope}; {len(i_s)} + {len(i_t)} frame)"))
display(LATENT_TABLE)
LATENT_TABLE.to_csv(TAB_DIR / "table9_latent_alignment.csv", index=False)
''')

code(r'''
# =========================================================
# 11.2 Hình: t-SNE trước / sau căn chỉnh (4 nhóm)
# =========================================================
def run_tsne(Z, seed=0):
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    Zp = PCA(n_components=min(50, Z.shape[1], len(Z) - 1), random_state=seed).fit_transform(Z)
    return TSNE(n_components=2, perplexity=float(min(30, max(5, (len(Z) - 1) / 4))), init="pca",
                learning_rate="auto", random_state=seed).fit_transform(Zp)

ms = list(LATENT)
fig, axes = plt.subplots(1, len(ms), figsize=(4.3 * len(ms), 4.4), squeeze=False)
for a, m in zip(axes[0], ms):
    L = LATENT[m]
    T2 = run_tsne(np.vstack([L["Es"], L["Et"]]), seed=cfg.seed)
    keys = [(0, int(v)) for v in Y_ALL[i_s]] + [(1, int(v)) for v in Y_ALL[i_t]]
    for key, st in GROUP_STYLE.items():
        sel = np.array([k == key for k in keys])
        if sel.any():
            a.scatter(T2[sel, 0], T2[sel, 1], s=7, c=st["color"], marker=st["marker"], alpha=.65, lw=0, label=st["label"])
    a.set_title(f"{METHOD_LABEL[m]}\nA-dist {L['proxy_A']:.2f} | CORAL {L['coral']:.4f}"
                + (f" | cos {L['cos_leak_dir']:.2f}" if "cos_leak_dir" in L else ""), fontsize=9)
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
axes[0][0].legend(fontsize=7, markerscale=2, loc="best")
savefig(fig, "latent_tsne",
        f"t-SNE của f = G_f(x) ({lat_scope}). ○ {cfg.source_name}, △ {cfg.target_name}. Căn chỉnh tốt: hai miền chồng lên "
        f"nhau (A-distance → 0) trong khi Leak/Safe vẫn tách rời (cos hướng rò → 1).")
''')

# =============================================================================
# 12. Mô hình cuối & xuất
# =============================================================================
md(r'''
## 12. Mô hình triển khai / Deployment model

Phương pháp xuất là `cfg.final_method` (**DANN + CORAL**, chọn **trước** khi xem kết quả). Mô hình cuối học trên toàn bộ «S»-train
có nhãn và **toàn bộ** «T» không nhãn, rồi xuất TorchScript (CMVN miền «T» đã nhúng), state_dict, `manifest.json` (mọi tham số
tiền xử lý — **kể cả dải và tần số lấy mẫu đã chọn** — cần để tái lập đúng đầu vào) và bảng xếp hạng bản ghi «T» theo xác
suất rò (danh sách ưu tiên khảo sát hiện trường).
''')

code(r'''
# =========================================================
# 12.1 Huấn luyện mô hình cuối + xuất TorchScript
# =========================================================
class DeployNet(nn.Module):
    def __init__(self, net, dom=1):
        super().__init__()
        self.net, self.dom = net, int(dom)
    def forward(self, x):
        d = torch.zeros_like(x[:, 0]).long() + self.dom
        return torch.sigmoid(self.net(x, d))

FINAL = None
if cfg.train_final_model:
    FINAL = get_final(cfg.final_method)
    print(f"Mô hình cuối {cfg.main_backbone}/{cfg.final_method}: {FINAL['n_labeled']} frame nguồn có nhãn + "
          f"{FINAL['n_unlabeled']} frame đích không nhãn, huấn luyện {FINAL['train_time_s']:.0f} s")
    print(f"Kiểm tra hợp lý: AUC {cfg.source_name}-val của mô hình cuối = "
          f"{fast_auc(Y_ALL[FINAL['srcval_idx']], FINAL['p_srcval']):.3f}")
    net = DANNNet(cfg.main_backbone); net.load_state_dict(FINAL["state"]); net.eval()
    dep = DeployNet(net, 1).eval()
    _ex = torch.from_numpy(X_ALL[np.where(DOM_ALL == 1)[0][:4]].copy())
    with torch.no_grad():
        _ref = dep(_ex)
    stem = f"leak_da_{SCENARIO}_{cfg.main_backbone}_{cfg.final_method}"
    ts = torch.jit.trace(dep, _ex, check_trace=False)
    ts.save(str(EXPORT_DIR / f"{stem}.ts"))
    with torch.no_grad():
        _chk = torch.jit.load(str(EXPORT_DIR / f"{stem}.ts"))(_ex)
    assert torch.allclose(_ref, _chk, atol=1e-5), "TorchScript cho kết quả khác mô hình gốc"
    torch.save(dict(state_dict=FINAL["state"], backbone=cfg.main_backbone, method=cfg.final_method,
                    cfg=asdict(cfg)), EXPORT_DIR / f"{stem}_state.pt")
    _perf = {}
    if EVAL_ON and len(RES):
        _r = RES[(RES.backbone == cfg.main_backbone) & (RES.method == cfg.final_method)]
        if len(_r):
            _r = _r.iloc[0]
            _perf = dict(target_record_auc=float(_r.record_auc), ci95=[float(_r.record_auc_lo), float(_r.record_auc_hi)],
                         delta_vs_source_only=float(_r.delta_rec_auc), p_holm=float(_r.p_holm),
                         note="ước lượng từ CV theo fold đích; mô hình xuất được huấn luyện lại trên toàn bộ dữ liệu")
    _warn = {"pseudo_target": [f"{cfg.target_name} là {cfg.source_name} biến đổi (pseudo_target) - CHƯA kiểm chứng trên dữ liệu thật."],
             "simulated_source": [f"{cfg.source_name} sinh bằng bộ mô phỏng tích hợp đơn giản."],
             "synthetic": ["Dữ liệu mô phỏng - chỉ để kiểm thử pipeline."]}.get(DATA_MODE, [])
    manifest = dict(
        scenario=SCENARIO, model=f"{stem}.ts", input=dict(shape=["B", FRAME_LEN], sr=cfg.target_sr,
                                                          frame_sec=cfg.frame_sec, hop_sec=cfg.hop_sec),
        preprocessing=dict(resample_to=cfg.target_sr, dc_removal=True, bandpass_hz=list(cfg.band_hz),
                           band_selection={k: (list(v) if isinstance(v, tuple) else v) for k, v in BAND_INFO.items()},
                           butter_order=cfg.filter_order, zero_phase=True,
                           mains_comb_hz=cfg.mains_hz if cfg.notch_mains else None, notch_q=cfg.notch_q,
                           peak_normalise="per file (OFFLINE)",
                           vmd=(dict(K=cfg.vmd_K, alpha=cfg.vmd_alpha, tau=cfg.vmd_tau, iters=cfg.vmd_iters,
                                     keep="modes with centre in band_hz and >%.1f Hz from mains harmonics" % cfg.vmd_mains_tol_hz)
                                if cfg.use_vmd else None)),
        output="P(leak) per frame; record decision = mean over frames (p_mean) >= threshold",
        threshold=cfg.decision_threshold, domain_embedded=cfg.target_name, tf_repr=cfg.tf_repr,
        data_mode=DATA_MODE, source=str(SRC_ROOT), target=str(TGT_ROOT), performance=_perf, warnings=_warn)
    (EXPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("Đã xuất:", sorted(p.name for p in EXPORT_DIR.iterdir()))
''')

code(r'''
# =========================================================
# 12.2 Xếp hạng bản ghi đích theo xác suất rò (danh sách ưu tiên khảo sát hiện trường)
# =========================================================
if FINAL is not None:
    net = DANNNet(cfg.main_backbone).to(DEVICE); net.load_state_dict(FINAL["state"])
    it = np.where(DOM_ALL == 1)[0]
    p = predict(net, it, DEVICE, dom=1)
    rank = (pd.DataFrame(dict(record_id=REC_ALL[it], p=p))
            .groupby("record_id").agg(p_mean=("p", "mean"), p_p90=("p", lambda v: np.percentile(v, 90)),
                                      frac_frames_pos=("p", lambda v: float(np.mean(v >= cfg.decision_threshold))),
                                      n_frames=("p", "size"))
            .reset_index().sort_values("p_mean", ascending=False))
    rank["decision"] = np.where(rank.p_mean >= cfg.decision_threshold, "NGHI RÒ", "bình thường")
    rank["group"] = rank.record_id.map(REC_GROUP)
    rank["label_if_known"] = rank.record_id.map(lambda r: {1: "leak", 0: "noleak"}.get(REC_LABEL[r], "?"))
    rank.to_csv(EXPORT_DIR / "target_record_ranking.csv", index=False)
    display(Markdown(f"### Table 10. {cfg.target_name} records ranked by leak probability (mô hình cuối, top 15)"))
    display(rank.head(15).round(4))
    if TARGET_HAS_LABELS:
        _known = rank[rank.label_if_known != "?"]
        print("Lưu ý: mô hình cuối ĐÃ thấy các bản ghi này (không nhãn) khi huấn luyện -> AUC dưới đây là "
              f"transductive, chỉ để đối chiếu: {fast_auc((_known.label_if_known == 'leak').values, _known.p_mean.values):.3f}")
    del net
''')

code(r'''
# =========================================================
# 12.3 Suy luận trọn tuyến trên một tệp đích thô (mô phỏng triển khai)
# =========================================================
TEST_FILE = ""          # để trống = tự chọn tệp của bản ghi đích có điểm cao nhất
if FINAL is not None:
    if not TEST_FILE:
        TEST_FILE = files.path[files.record_id == rank.record_id.iloc[0]].iloc[0]
    _rid = files.record_id[files.path == TEST_FILE].iloc[0] if (files.path == TEST_FILE).any() else None
    x = load_audio(TEST_FILE, sim=SIM_PARAMS.get(_rid))
    if len(x) < FRAME_LEN:
        x = np.pad(x, (0, FRAME_LEN - len(x)))
    starts = np.arange(0, len(x) - FRAME_LEN + 1, HOP_LEN)
    Xf = np.stack([x[s:s + FRAME_LEN] for s in starts]).astype(np.float32)
    if cfg.use_vmd:
        Xf = vmd_denoise_frames(Xf, DEVICE)[0]
    ts_model = torch.jit.load(str(EXPORT_DIR / f"leak_da_{SCENARIO}_{cfg.main_backbone}_{cfg.final_method}.ts"))
    with torch.no_grad():
        pf = ts_model(torch.from_numpy(Xf)).numpy()
    fig = plt.figure(figsize=(13.5, 6.4))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1.4, 1], hspace=.45)
    a0 = fig.add_subplot(gs[0]); tt = np.arange(len(x)) / cfg.target_sr
    a0.plot(tt, x, lw=.4, color=PAL["grey"]); a0.set_title(f"{Path(TEST_FILE).name} - sau tiền xử lý"); a0.set_xlim(0, tt[-1])
    a1 = fig.add_subplot(gs[1], sharex=a0)
    fe_ = make_tf_frontend()
    with torch.no_grad():
        S = fe_(torch.from_numpy(x[None].copy()))[0, 0].numpy()
    a1.imshow(S, origin="lower", aspect="auto", cmap="magma", extent=[0, tt[-1], -.5, fe_.n_bands - .5])
    yt = [f for f in (20, 50, 100, 200, 350, 500, 1000, 2000, 4000) if cfg.band_hz[0] <= f <= cfg.band_hz[1]]
    a1.set_yticks(np.interp(np.log(yt), np.log(fe_.freqs), np.arange(fe_.n_bands))); a1.set_yticklabels(yt)
    a1.set_ylabel("Hz"); a1.grid(False)
    a2 = fig.add_subplot(gs[2], sharex=a0)
    a2.plot(starts / cfg.target_sr + cfg.frame_sec / 2, pf, "o-", color=PAL["tgt"], ms=3)
    a2.axhline(cfg.decision_threshold, color="k", ls=":"); a2.set_ylim(-.02, 1.02)
    a2.set_ylabel("P(leak)"); a2.set_xlabel("thời gian (s)")
    a2.set_title(f"p_mean = {pf.mean():.3f} -> {'NGHI RÒ' if pf.mean() >= cfg.decision_threshold else 'bình thường'}"
                 + (f" | nhãn thật: {'leak' if REC_LABEL[_rid] == 1 else 'noleak'}"
                    if _rid is not None and REC_LABEL[_rid] >= 0 else ""))
    savefig(fig, "field_inference", f"Suy luận trọn tuyến (tiền xử lý → VMD → TorchScript) trên một tệp {cfg.target_name}.")
''')

# =============================================================================
# 13. Đoạn kết quả tự sinh + kết luận
# =============================================================================
md(r'''
## 13. Đoạn kết quả tự sinh và kết luận / Auto-generated results and conclusions

Đoạn dưới đây được **sinh từ số liệu** của lần chạy này; mọi khẳng định kèm CI và notebook tự liệt kê những điều **không
được** khẳng định.
''')

code(r'''
# =========================================================
# 13.1 Đoạn Results tự sinh - CHỈ phát biểu điều số liệu cho phép
# =========================================================
L = []
nS, nT = int((records.domain == 0).sum()), int((records.domain == 1).sum())
L.append(f"**Dữ liệu.** Kịch bản `{SCENARIO}`, chế độ `{DATA_MODE}`: {nS} bản ghi {cfg.source_name} "
         f"({records.duration_s[records.domain == 0].sum() / 60:.1f} phút) và {nT} bản ghi {cfg.target_name} "
         f"({TGT_LAB_COUNTS[1]} leak / {TGT_LAB_COUNTS[0]} no-leak có nhãn, {N_TGT_UNLAB} không nhãn"
         f"{f', {records[records.domain == 1].group.nunique()} nhóm' if cfg.target_group_csv else ''}); frame "
         f"{cfg.frame_sec:g} s @ {cfg.target_sr} Hz, dải {cfg.band_hz[0]:.0f}-{cfg.band_hz[1]:.0f} Hz "
         f"({'chọn tự động' if cfg.band_mode == 'auto' else 'cố định'}), {'VMD + ' if cfg.use_vmd else ''}{cfg.tf_repr.upper()}.")
_top = qual.head(3)
L.append(f"**Lệch miền.** Chỉ số tách hai miền mạnh nhất (không nhãn): "
         + "; ".join(f"{r.descriptor} (tách {r.domain_sep:.2f}, {r.direction})" for r in _top.itertuples())
         + f". Tỉ số tần số trọng tâm r_emp = {R_EMP:.2f}"
         + (f", Korteweg r = {R_KORTEWEG:.2f}" if abs(R_KORTEWEG - 1) > 0.05 else "") + "."
         + (f" Trùng khớp sức phân biệt theo dải {cfg.source_name}/{cfg.target_name}: ρ = {BAND_RHO:.2f}"
            + ((f" — thấp nhưng phù hợp với dịch tần đã biết (r ≈ {R_EMP:.2f}), chưa chắc là concept shift." if DAUG_WARP_ON
                else " — thấp, gợi ý **concept shift** (cơ chế rò ở đích khác nguồn)." ) if BAND_RHO < 0.3 else ".")
            if np.isfinite(BAND_RHO) else ""))
if EVAL_ON and len(RES):
    for bb in RES.backbone.unique():
        g = RES[RES.backbone == bb].set_index("method")
        so = g.loc["source_only"]
        s = (f"**{bb}.** Source-only: AUC bản ghi {so.record_auc:.3f} [{so.record_auc_lo:.3f}, {so.record_auc_hi:.3f}] "
             f"trên {cfg.target_name} ({cfg.source_name}-val {so.src_val_auc:.3f})")
        if "oracle" in g.index:
            s += f"; Oracle {g.loc['oracle'].record_auc:.3f} [{g.loc['oracle'].record_auc_lo:.3f}, {g.loc['oracle'].record_auc_hi:.3f}]"
        s += "."
        if cfg.final_method in g.index:
            fm = g.loc[cfg.final_method]
            verdict = ("cải thiện **có ý nghĩa thống kê**" if (fm.delta_lo > 0 and fm.p_holm < 0.05) else
                       "**giảm** có ý nghĩa thống kê (negative transfer)" if (fm.delta_hi < 0 and fm.p_holm < 0.05) else
                       "**chưa đủ bằng chứng** để kết luận khác Source-only (CI của Δ chứa 0)")
            s += (f" {METHOD_LABEL[cfg.final_method]} (đăng ký trước): {fm.record_auc:.3f} "
                  f"[{fm.record_auc_lo:.3f}, {fm.record_auc_hi:.3f}], ΔAUC = {fm.delta_rec_auc:+.3f} "
                  f"[{fm.delta_lo:+.3f}, {fm.delta_hi:+.3f}], p_Holm = {fm.p_holm:.3f} → {verdict}")
            if np.isfinite(fm.gap_closed_pct):
                s += f"; thu hẹp {fm.gap_closed_pct:.0f}% khoảng cách Source-only→Oracle"
            s += "."
        uda = g.loc[[m for m in UDA_METHODS if m in g.index]]
        if len(uda):
            b = uda.record_auc.idxmax()
            s += (f" Trong các phương pháp không nhãn, cao nhất là {METHOD_LABEL[b]} ({uda.record_auc.max():.3f}) — "
                  f"*chọn sau khi xem kết quả nên chỉ mang tính mô tả*.")
        L.append(s)
    if len(VALUE_TAB):
        v = VALUE_TAB[(VALUE_TAB.backbone == cfg.main_backbone) & VALUE_TAB.comparison.str.startswith("FT từ Source-only")]
        if len(v):
            v = v.iloc[0]
            L.append(f"**Giá trị của {cfg.source_name} (Q3, {cfg.main_backbone}).** Với cùng các bản ghi few-shot, tinh chỉnh từ "
                     f"mô hình học trên {cfg.source_name} đạt {v.auc_a:.3f} so với {v.auc_b:.3f} khi chỉ dùng {cfg.target_name}: "
                     f"Δ = {v.delta:+.3f} [{v.delta_lo:+.3f}, {v.delta_hi:+.3f}], p_Holm = {v.p_holm:.3f}"
                     + (" → dữ liệu nguồn **có giá trị** (có ý nghĩa thống kê)." if v.delta_lo > 0 and v.p_holm < 0.05 else
                        " → dữ liệu nguồn **gây hại** (có ý nghĩa thống kê)." if v.delta_hi < 0 and v.p_holm < 0.05 else
                        " → **chưa đủ bằng chứng** về giá trị của dữ liệu nguồn (CI chứa 0 hoặc p_Holm ≥ 0,05)."))
    if len(BBSE_TAB):
        b = BBSE_TAB.iloc[0]
        L.append(f"**Tỉ lệ lớp.** Tỉ lệ frame rò: {cfg.source_name}-train {b.prior_src:.2f}, {cfg.target_name} thật "
                 f"{b.prior_true:.2f}, ước lượng BBSE (không nhãn, {b.method}) {b.prior_bbse:.2f}.")
if len(LATENT_TABLE):
    t = LATENT_TABLE.set_index("method")
    a = t.loc[METHOD_LABEL["source_only"]] if METHOD_LABEL["source_only"] in t.index else None
    b = t.loc[METHOD_LABEL[cfg.final_method]] if METHOD_LABEL[cfg.final_method] in t.index else None
    if a is not None and b is not None:
        s = (f"**Không gian ẩn ({cfg.main_backbone}).** Proxy A-distance {a.proxy_A:.2f} → {b.proxy_A:.2f}, "
             f"CORAL {a.coral:.4f} → {b.coral:.4f}, MMD² {a.mmd:.4f} → {b.mmd:.4f} (Source-only → {METHOD_LABEL[cfg.final_method]})")
        if "cos_leak_dir" in t.columns and np.isfinite(a.get("cos_leak_dir", np.nan)):
            s += f"; cos hướng rò {a.cos_leak_dir:.2f} → {b.cos_leak_dir:.2f}"
        L.append(s + ".")

NOT = []
if DATA_MODE == "pseudo_target":
    NOT.append(f"{cfg.target_name} là **{cfg.source_name} biến đổi**, KHÔNG phải dữ liệu thật: không được trình bày như kết quả trên {cfg.target_name}.")
if DATA_MODE == "simulated_source":
    NOT.append(f"{cfg.source_name} do **bộ mô phỏng tích hợp đơn giản** sinh ra: kết quả nói về bộ mô phỏng này, không về bộ "
               f"mô phỏng/testbed thật của bạn.")
if DATA_MODE == "synthetic":
    NOT.append("Dữ liệu **mô phỏng**: mọi con số chỉ kiểm chứng pipeline.")
if not EVAL_ON:
    NOT.append(f"{cfg.target_name} không có nhãn: không có chỉ số hiệu năng nào trên miền đích; bảng xếp hạng chỉ là danh sách ưu tiên khảo sát.")
if EVAL_ON and len(RES) and int(RES.n_records.iloc[0]) < 20:
    NOT.append(f"Chỉ {int(RES.n_records.iloc[0])} bản ghi {cfg.target_name} kiểm thử: CI rộng, không khái quát hoá cho mạng lưới.")
if len(cfg.seeds) < 3:
    NOT.append(f"Chỉ {len(cfg.seeds)} seed: chưa đo độ bất định do khởi tạo/phân hoạch (đặt cfg.seeds = (42, 43, 44)).")
if not cfg.target_group_csv:
    NOT.append(f"Không có thông tin điểm khảo sát của {cfg.target_name}: chưa loại trừ 'dấu vân tay vị trí' (điền cfg.target_group_csv).")
NOT.append("Không có quy tắc chọn mô hình không nhãn nào được kiểm chứng: dùng epoch cuối + ngưỡng 0,5 đăng ký trước.")
NOT.append("Xử lý OFFLINE (chuẩn hoá đỉnh theo cả tệp, lọc pha 0): chưa phải bộ phát hiện thời gian thực.")

txt = "\n\n".join(L) + "\n\n**Không được khẳng định:**\n" + "\n".join(f"- {s}" for s in NOT)
display(Markdown(txt))
(OUT / "results_paragraph.md").write_text(txt, encoding="utf-8")
print("Hình:", len(FIG_INDEX), "| bảng:", sorted(p.name for p in TAB_DIR.iterdir()))
''')

md({"sim2real": r'''
### Kết luận, hạn chế và hướng tiếp theo / Conclusions, limitations and next steps

| Câu hỏi | Nơi trả lời |
|---|---|
| Q1 khoảng cách mô phỏng → thực địa | Bảng 6 (Source-only vs Oracle), Bảng 7 dòng cuối |
| Q2 thích nghi không nhãn | Bảng 6, Hình forest / ΔAUC ghép cặp |
| Q3 giá trị của Sx | Bảng 7, Hình 10.6 (c) |
| Q4 dải chuyển giao | Bảng 4, Hình hồ sơ dải (ρ Sx/Rx) |
| Q5 lệch ở đâu | Bảng 3, Hình phân tích lệch miền, Bảng 9, t-SNE |

**Hướng tiếp theo:** (1) điền `target_group_csv` và chạy lại để có đánh giá **theo điểm khảo sát**; (2) tăng `cfg.seeds` lên 3–5
trước khi chốt số; (3) nếu ρ Sx/Rx thấp (concept shift), hiệu chỉnh **bộ mô phỏng** (dải rò, SNR, gián đoạn) theo Bảng 3–4
thay vì chỉ tăng trọng số căn chỉnh; (4) nếu Q3 cho thấy Sx có giá trị, khảo sát **đường cong số nhãn Rx** (5 %, 10 %, 25 %)
bằng `cfg.fewshot_frac`; (5) học tương phản có giám sát đa miền (SupCon) và DA một lớp (chỉ Rx no-leak) là các nhánh mở rộng
trên cùng `G_f`.
''', "pvc2hdpe": r'''
### Kết luận, hạn chế và hướng tiếp theo / Conclusions, limitations and next steps

| Tài liệu | Hiện thực |
|---|---|
| §1 Korteweg, dịch tần, suy hao | 3.1, 4 (phổ, hồ sơ dải, $r_{emp}$), 5.3 (tâm mode VMD) |
| §3.1 CMVN, lọc dải 15–350 Hz | 3.2, `DomainCMVN` |
| §3.2 AdaBN, fine-tuning chọn lọc | `adabn_`, `run_post` |
| §3.3A DANN + GRL | `GradReverse`, `train_da` (+ CDAN+E) |
| §3.3B Deep CORAL, Deep MMD | `coral_loss`, `mmd_rbf` |
| §4 Giả lập vật lý PVC→HDPE | `daug` (dịch tần + suy hao + nhiễu nền HDPE), chế độ `pseudo_target` |
| §5 Source-only / UDA / Few-shot / Oracle | 7–10 (+ Few-shot chỉ HDPE để đo giá trị của dữ liệu PVC) |

Hạn chế: cần **dữ liệu HDPE thật**; chọn mô hình không nhãn vẫn là bài toán mở; nên chạy leave-one-site-out khi có thông tin
điểm khảo sát và tăng `cfg.seeds` lên 3–5 trước khi chốt số liệu.
'''})
