# -*- coding: utf-8 -*-
"""Notebook ADA v1 — phần 1: giới thiệu, thiết lập, dò dữ liệu đa miền, EDA dịch chuyển miền.

Mỗi phần tử của CELLS là (loại cell, mã nguồn). tools/build_ada.py ghép các phần lại.
"""

CELLS = []


def md(s):
    CELLS.append(("markdown", s))


def code(s):
    CELLS.append(("code", s))


# =============================================================================
# 0. GIỚI THIỆU
# =============================================================================
md(r'''
# Thích ứng miền đối nghịch cho nhận dạng rò rỉ ống PVC áp lực thấp bằng âm thanh — tập `tapdulieuMix_16k` (v2)

**Adversarial Domain Adaptation for Acoustic Leak Detection in Low-Pressure PVC Pipes across Acquisition Domains**

---

### Bối cảnh / Motivation

Pipeline v7 (`leak-cnn-pvc-kaggle-v7.ipynb`) huấn luyện và đánh giá **trong cùng một miền** thu âm.
Khi đem mô hình sang một miền khác (phòng thí nghiệm → hiện trường, cảm biến khác, điểm khảo sát khác,
chuỗi thu 8 kHz → 16 kHz...), phân bố đặc trưng **P(x)** thay đổi trong khi quan hệ vật lý
"có rò rỉ ↔ tín hiệu" **P(y|x)** được giả thiết giữ nguyên → **dịch chuyển hiệp biến (covariate shift)**.
Notebook này đo mức dịch chuyển đó, rồi áp dụng **thích ứng miền đối nghịch** (DANN, CDAN+E) để học
đặc trưng **phân biệt được rò rỉ nhưng không phân biệt được miền**, dùng **dữ liệu miền đích không nhãn**.

### Câu hỏi nghiên cứu / Research questions

| RQ | Câu hỏi | Bằng chứng trong notebook |
|---|---|---|
| **RQ1** | Các miền trong `tapdulieuMix_16k` khác nhau đến mức nào, và dấu hiệu rò rỉ có **cùng chiều** giữa các miền không? | Mục 3: băng thông, PSD, AUC theo dải trong từng miền, ma trận "huấn luyện miền A → kiểm thử miền B", PAD đầu vào |
| **RQ2** | Thích ứng đối nghịch có cải thiện nhận dạng **mức bản ghi ở miền đích** so với *source-only* và các phương pháp không đối nghịch (AdaBN, Deep CORAL) không? | Mục 9: bảng chính + ΔAUC ghép cặp, bootstrap theo bản ghi, Holm |
| **RQ3** | Thích ứng làm gì với không gian đặc trưng (căn chỉnh miền, giữ cấu trúc lớp)? | Mục 10–11: động học huấn luyện, PAD theo epoch, t-SNE, PAD/MMD theo lớp |
| **RQ4** | Kết quả nhạy thế nào với λ, augmentation, và **cách chọn mô hình khi không có nhãn đích**? | Mục 12–13: quy tắc chọn (last / source-val / DEV / oracle), quét λ, ablation |
| **RQ5** | Cần bao nhiêu bản ghi đích **có nhãn** để vượt UDA? (tuỳ chọn) | Mục 14: đường cong bán giám sát |

### Phương pháp so sánh / Compared methods

| Mã | Phương pháp | Dùng dữ liệu đích | Vai trò |
|---|---|---|---|
| `source_only` | Huấn luyện chỉ trên miền nguồn | không | **cận dưới** |
| `joint_bn` | Lô trộn nguồn + đích qua BatchNorm, **không** hàm căn chỉnh | không nhãn | **đối chứng cơ chế** (v2) |
| `adabn` | Source-only + ước lượng lại BatchNorm trên đích (Li et al., 2018) | không nhãn | baseline không huấn luyện |
| `tent` | Thích ứng lúc kiểm thử: cực tiểu entropy trên tham số BN (Wang et al., 2021) | không nhãn | baseline không cần nguồn (v2) |
| `coral` | Deep CORAL — khớp hiệp phương sai (Sun & Saenko, 2016) | không nhãn | baseline không đối nghịch |
| `mmd` *(tuỳ chọn)* | DAN — MK-MMD (Long et al., 2015) | không nhãn | baseline không đối nghịch |
| `mcc` | Minimum Class Confusion (Jin et al., 2020) | không nhãn | baseline hiện đại (v2) |
| **`dann`** | **DANN — Gradient Reversal Layer** (Ganin et al., 2016) | không nhãn | phương pháp đối nghịch chuẩn |
| **`cdan_e`** | **CDAN+E — đối nghịch có điều kiện + trọng số entropy** (Long et al., 2018) | không nhãn | biến thể đối nghịch có điều kiện |
| **`rcst`** | **DANN + tự huấn luyện nhất quán mức bản ghi** (nhãn giả theo bản ghi + phạt không nhất quán trong bản ghi) | không nhãn | **phương pháp đề xuất** (v2) |
| `oracle` | Nguồn + đích **có nhãn** (giám sát đầy đủ) | có nhãn | cận trên tham chiếu |
| `target_only` | Chỉ đích **có nhãn** (cùng fold) | có nhãn | cận trên tham chiếu thứ hai (v2) |

### Điểm mới của v2 (sửa theo phản biện lần chạy v1 — `docs/DANH_GIA_KET_QUA_ADA_v1.md`)

| Mã | Vấn đề ở v1 | Thay đổi ở v2 |
|---|---|---|
| **V1** | 165 tệp `Sxx_*` bị coi là 165 bản ghi độc lập → AUC nguồn-val = 1.000 ở mọi job nguồn S, oracle S = 1.000 | Mã bản ghi `[RS]xx` **ghép miền và lớp** (`S01_leak01` + `S01_leak02` = một bản ghi; `S01_leak` ≠ `S01_noleak`) |
| **V2** | Dải < 90 Hz và > 3 kHz phân biệt hoàn hảo trong S nhưng không chuyển sang R | Mục 3.9: bảng **khả năng chuyển giao theo dải** (trong miền vs chéo miền) |
| **V3** | Chuỗi R cắt thông cao < 100 Hz, chuỗi S không; quy tắc hoà hợp chỉ xét phía cao | Hoà hợp **hai phía** từ PSD trung bình đã căn mức (không dùng nhãn) + **ngẫu nhiên hoá chuỗi thu** khi huấn luyện |
| **V4** | Bộ phân biệt ở mức may rủi từ epoch 1, PAD không giảm, quét λ phẳng → lợi ích có thể do BatchNorm | Đối chứng **Joint-BN (λ=0)**, bảng **phân rã cơ chế** (mục 9.2b); bộ phân biệt theo bản gốc (1024, lr ×10) và đo độ chính xác **ở chế độ eval** |
| **V5** | Ablation 1 seed; p ghi "0.000"; Holm gộp cả cận trên | Ablation 3 seed; p ghi "< 1/B"; Holm chỉ trên các phương pháp UDA |
| **M** | Chỉ các baseline 2015–2018 | Thêm **MCC**, **TENT**, **target-only** và phương pháp đề xuất **RCST** |

### Giao thức đánh giá (tóm tắt)

1. **Đơn vị thống kê = bản ghi** (`record_uid = miền|Rxxx`; mã `Rxxx` trùng nhau giữa các miền được tách riêng).
2. **Quy nạp (inductive) theo K-fold trên bản ghi đích**: dữ liệu đích không nhãn dùng khi huấn luyện **không bao giờ**
   chứa bản ghi đang được kiểm thử → không có rò rỉ kiểu *transductive*.
3. **Không dùng nhãn đích để chọn mô hình**: quy tắc chính đăng ký trước là `last` (lịch huấn luyện cố định);
   `source-val`, `DEV` (You et al., 2019) là phân tích độ nhạy; `oracle` chỉ để **đo mức lạc quan**.
4. **CV lặp** (mặc định 3 seed: phân hoạch mới + khởi tạo mới) và **trung bình OOF qua các seed** làm ước lượng chính.
5. **CI bootstrap theo bản ghi**, **ΔAUC ghép cặp** so với *source-only*, **hiệu chỉnh Holm**.

> **Cách dùng nhanh / Quick start**
> 1. Tải thư mục `tapdulieuMix_16k` lên Kaggle (`Datasets → New dataset`), rồi `Add Data` vào notebook.
> 2. Notebook **tự dò** miền: thư mục `<miền>/<leak|noleak>/`, thư mục ghép `leak_16k/`, ký hiệu trong tên tệp
>    (`ptn_R001_leak01.wav`) hoặc chuỗi thu 8/16 kHz đo từ phổ. Nếu tập Mix chỉ là **một** miền, gắn thêm dataset thứ hai
>    và đặt `cfg.domain_roots = {"mix": "...tapdulieuMix_16k", "field": "...field_rec_8k_V7.2"}`. Không dò được → mục 2.3
>    in chẩn đoán + tạo `domain_map_template.csv` để gán miền bằng tay (`cfg.domain_source="csv"`).
> 3. **Kiểm tra mục 2.3–2.4** (miền nào được nhận ra, mỗi miền bao nhiêu bản ghi mỗi lớp) trước khi để chạy hết.
> 4. `Settings → Accelerator = GPU T4 x2`, rồi `Run All`. Cấu hình mặc định ~1.5–3 giờ tuỳ kích thước dữ liệu.
>    Không tìm thấy dữ liệu → notebook tự sinh **dữ liệu mô phỏng hai miền** (SYNTHETIC DEMO MODE).
> 5. Chạy thử nhanh: biến môi trường `LEAK_SMOKE=1` (vài vòng lặp, 2 fold, 1 seed).
''')

md(r'''
## 0. Cơ sở lý thuyết / Theoretical background

**Cận sai số miền đích (Ben-David et al., 2010).** Với mọi giả thuyết $h \in \mathcal{H}$:

$$\varepsilon_T(h) \;\le\; \varepsilon_S(h) \;+\; \tfrac{1}{2}\, d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) \;+\; \lambda^\*$$

Sai số đích bị chặn bởi sai số nguồn, **độ lệch giữa hai miền** đo bằng một bộ phân lớp miền, và sai số của
giả thuyết lý tưởng chung $\lambda^\*$. Độ lệch được ước lượng thực nghiệm bằng **Proxy A-distance**
$\mathrm{PAD} = 2(1 - 2\epsilon)$, với $\epsilon$ là sai số của bộ phân lớp nguồn-vs-đích (mục 3.7, 10, 11).

**DANN (Ganin et al., 2016).** Bộ trích đặc trưng $G_f$, bộ dự đoán nhãn $G_y$, bộ phân biệt miền $G_d$:

$$\min_{G_f, G_y}\; \max_{G_d}\;\; \mathcal{L}_y(G_y(G_f(x^s)), y^s) \;-\; \lambda_p\, \mathcal{L}_d\big(G_d(G_f(x)), d\big)$$

thực hiện bằng **lớp đảo gradient (GRL)**: truyền xuôi là ánh xạ đồng nhất, truyền ngược nhân gradient với $-\lambda_p$.
Hệ số tăng dần $\lambda_p = \lambda_{\max}\left(\frac{2}{1+e^{-\gamma p}} - 1\right)$, $\gamma = 10$, $p \in [0,1]$ là tiến độ huấn luyện.

**CDAN+E (Long et al., 2018).** $G_d$ nhận **ánh xạ đa tuyến** $f \otimes g$ (đặc trưng ⊗ xác suất lớp) để căn chỉnh
**phân bố có điều kiện theo lớp**, và mỗi mẫu được đánh trọng số $w = 1 + e^{-H(g)}$ để ưu tiên mẫu dự đoán chắc chắn.

**Giới hạn đã biết — phải kiểm tra trên dữ liệu thật:**

- **Dịch chuyển nhãn** (tỉ lệ leak/noleak khác nhau giữa miền): căn chỉnh biên P(f) có thể *làm hại*
  (Zhao et al., 2019). → mục 2.4 in tỉ lệ lớp theo miền; nguồn được lấy mẫu **cân bằng lớp**; CDAN+E giảm nhẹ vấn đề.
- **Dịch chuyển có điều kiện** (dấu hiệu rò rỉ đổi chiều giữa các miền): khi đó không phương pháp UDA nào đảm bảo
  cải thiện. → mục 3.5 kiểm tra chiều của AUC theo dải trong từng miền.
- **Chọn mô hình không có nhãn đích**: chọn epoch/siêu tham số bằng nhãn đích làm kết quả UDA **lạc quan giả tạo**
  (You et al., 2019). → mục 12 đo trực tiếp mức lạc quan này.

**Tài liệu tham khảo chính.** Ben-David et al., *Mach. Learn.* 79 (2010) · Ganin et al., *JMLR* 17 (2016) ·
Long et al., *NeurIPS* (2018) · Sun & Saenko, *ECCV-W* (2016) · Long et al., *ICML* (2015) ·
Li et al., *Pattern Recognit.* 80 (2018) · Zhao et al., *ICML* (2019) · You et al., *ICML* (2019).
''')

# =============================================================================
# 1. THIẾT LẬP
# =============================================================================
md(r'''
## 1. Thiết lập môi trường / Environment setup
''')

code(r'''
# =========================================================
# 1.1 Imports & phụ thuộc / Imports and dependencies
# =========================================================
import os, sys, gc, json, math, re, time, random, shutil, hashlib, warnings, subprocess
import threading, pickle, copy, queue, traceback, zipfile, io
import unicodedata
from pathlib import Path
from dataclasses import dataclass, asdict, field
from collections import defaultdict

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

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
import scipy.stats as spstats

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch
from IPython.display import display, Markdown, Image as IPyImage

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_fscore_support,
                             confusion_matrix, matthews_corrcoef, balanced_accuracy_score,
                             accuracy_score, silhouette_score)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def amp_autocast(device, enabled):
    try:
        return torch.amp.autocast(device_type=torch.device(device).type, enabled=bool(enabled))
    except Exception:
        return torch.cuda.amp.autocast(enabled=bool(enabled))

def make_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=bool(enabled))
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=bool(enabled))

T_START = time.time()
print("Python      :", sys.version.split()[0])
print("NumPy       :", np.__version__, "| pandas:", pd.__version__)
print("PyTorch     :", torch.__version__, "| torchaudio:", torchaudio.__version__)
print("librosa     :", librosa.__version__)
print("CUDA        :", torch.cuda.is_available())
N_GPU = torch.cuda.device_count()
for i in range(N_GPU):
    p = torch.cuda.get_device_properties(i)
    print(f"  [GPU {i}] {p.name} | {p.total_memory/1024**3:.1f} GB")
''')

code(r'''
# =========================================================
# 1.2 Cấu hình trung tâm / Central configuration
#     Mọi tham số thí nghiệm nằm ở đây. Ghi chú [P] = tham số ĐĂNG KÝ TRƯỚC (không chỉnh theo
#     kết quả trên miền đích); [S] = chỉ dùng cho phân tích độ nhạy.
# =========================================================
@dataclass
class CFG:
    # ---- Dữ liệu / Data ------------------------------------------------------
    data_root: str = ""                       # để trống -> tự dò trong /kaggle/input theo dataset_hint
    dataset_hint: str = "tapdulieuMix_16k"     # tên thư mục gốc của tập Mix
    leak_keys: tuple = ("leak", "ro_ri", "rori")
    noleak_keys: tuple = ("noleak", "no_leak", "non_leak", "nonleak", "normal",
                          "background", "khong_ro_ri", "khongroli", "no-leak")
    audio_ext: tuple = (".wav", ".flac", ".ogg", ".mp3", ".WAV")
    # [V2] Mã bản ghi: Rxxx (hiện trường) và Sxx (mô hình thí nghiệm). Chuỗi S ĐÁNH SỐ LẠI TRONG MỖI LỚP
    #      (S01_leak và S01_noleak là hai lần thu khác nhau, ở hai điểm đo khác nhau) nên mã bản ghi được
    #      GHÉP THÊM LỚP: S01_leak01 + S01_leak02 = một bản ghi "S01|leak". v1 coi mỗi tệp S là một bản ghi
    #      -> rò rỉ giữa các tệp cùng lần thu (AUC nguồn-val = 1.000 ở mọi job nguồn S).
    record_regex: str = r"([RS]\d{1,4})"
    record_id_by_class: bool = True
    strict_checks: bool = True                 # dừng khi có mâu thuẫn nhãn

    # ---- MIỀN / Domains ------------------------------------------------------
    # domain_roots: MỖI DATASET MỘT MIỀN, ví dụ tập Mix + tập hiện trường v7:
    #   {"mix": "/kaggle/input/.../tapdulieuMix_16k", "field": "/kaggle/input/.../field_rec_8k_V7.2"}
    #   (chỉ cần đúng TÊN thư mục cuối nếu không chắc đường dẫn đầy đủ). Rỗng -> một dataset, miền tự dò.
    domain_roots: dict = field(default_factory=dict)
    # domain_source (khi không dùng domain_roots):
    #   auto         : thử lần lượt regex (nếu đặt) -> folder -> label_folder -> filename -> bandwidth (mục 2.3)
    #   folder       : thư mục không phải nhãn            tapdulieuMix_16k/<miền>/leak/...
    #   label_folder : phần còn lại của thư mục nhãn      leak_16k/, noleak_tt/        -> "16k", "tt"
    #   filename     : phần còn lại của tên tệp           lab_R001_leak01.wav          -> "lab"
    #   bandwidth    : chuỗi thu 8 kHz vs 16 kHz đo từ phổ (tệp 8 kHz nâng lên 16 kHz)
    #   regex        : domain_regex trên đường dẫn tương đối | csv: bảng gán miền (cột domain + rel/file/file_stem/record_id)
    domain_source: str = "auto"
    domain_depth: int = 1                      # số cấp thư mục ghép thành tên miền (cách "folder")
    # [V2] tapdulieuMix_16k: miền = tiền tố mã bản ghi (R = hiện trường, S = mô hình thí nghiệm)
    domain_regex: str = r"(?:^|/)([RS])\d+_"   # trên đường dẫn tương đối (nhóm 1 = miền); "" -> chỉ tự dò
    domain_regex_group: int = 1
    domain_csv: str = ""
    domain_alias: dict = field(default_factory=lambda: {"R": "R_field", "r": "R_field",
                                                        "S": "S_lab", "s": "S_lab"})   # đổi tên miền
    exclude_domains: tuple = ()
    min_records_per_class: int = 2             # miền cần >= ngần này bản ghi MỖI lớp để làm đích
    max_auto_domains: int = 6                  # cách tự dò cho > ngần này miền bị coi là không hợp lệ
    bw_split_db: float = -30.0                 # ngưỡng chuỗi thu (cách "bandwidth"); xem biểu đồ chẩn đoán
    derivative_markers: tuple = ("_aug", "_augment", "_events_removed", "_clean", "_denois", "_copy", "_dup")

    # ---- Bài toán chuyển miền / Transfer tasks -------------------------------
    # () = tự động: 2 miền -> A→B và B→A; >2 miền -> leave-one-domain-out (các miền còn lại → miền giữ lại)
    # Khai báo tay: (("lab", "field"), ("lab+field8k", "field16k"))
    tasks: tuple = ()
    max_tasks: int = 6

    # ---- Tín hiệu / Signal ---------------------------------------------------
    target_sr: int = 16000
    highpass_hz: float = 20.0
    notch_mains: bool = True
    mains_hz: float = 50.0
    mains_harmonics: int = 6
    notch_q: float = 30.0
    frame_sec: float = 2.0
    hop_sec: float = 1.0
    # [P][V2] Hoà hợp băng thông HAI PHÍA: so PSD trung bình (đã căn mức ở 300-1500 Hz) giữa các miền;
    #     dải thấp/cao nơi hai miền lệch nhau > harmonise_tol_db chỉ phản ánh CHUỖI THU (v1: chuỗi R cắt
    #     thông cao < 100 Hz, chuỗi S thì không) -> mọi front-end phổ giới hạn về dải CHUNG.
    #     "auto" | "off" | (f_min, f_max) cố định. Quy tắc không dùng nhãn.
    bandwidth_harmonize: str = "auto"
    harmonise_tol_db: float = 6.0
    f_min_cap: float = 300.0                   # không bao giờ cắt quá 300 Hz (dải rò rỉ PVC bắt đầu ~300 Hz)
    bw_energy_frac: float = 0.995
    check_duplicates: bool = True              # dò bản ghi trùng nội dung giữa/trong các miền

    # ---- Biểu diễn / Front-end ----------------------------------------------
    n_fft: int = 2048                          # 128 ms @ 16 kHz
    hop_length: int = 512                      # 32 ms
    n_mels: int = 64
    f_min: float = 20.0
    f_max: float = 8000.0
    specaug_freq_mask: int = 8
    specaug_time_mask: int = 8

    # ---- Kiến trúc / Architecture -------------------------------------------
    backbone: str = "cnn2d"                    # cnn2d (log-Mel, như LeakCNN2D v7) | cnn1d (dạng sóng)
    widths: tuple = (32, 64, 128, 256)
    bottleneck_dim: int = 256
    # [V2] Bộ phân biệt miền theo bản cài đặt gốc của DANN/CDAN (rộng 1024, lr x10). v1 dùng 256 / x1 và bộ
    #      phân biệt ở mức may rủi ngay từ epoch 1 trong khi đặc trưng vẫn tách miền (PAD 1.92) -> quá yếu.
    disc_hidden: int = 1024
    disc_dropout: float = 0.5

    # ---- Phương pháp / Methods ----------------------------------------------
    # [V2] joint_bn = đối chứng CƠ CHẾ (lô trộn nguồn+đích qua BatchNorm, KHÔNG có hàm căn chỉnh);
    #      tent = thích ứng lúc kiểm thử (Wang et al., 2021); mcc = Minimum Class Confusion (Jin et al., 2020);
    #      rcst = PHƯƠNG PHÁP ĐỀ XUẤT (DANN + tự huấn luyện nhất quán mức bản ghi); target_only = cận trên thứ hai.
    methods: tuple = ("source_only", "joint_bn", "adabn", "tent", "coral", "mcc", "dann", "cdan_e", "rcst",
                      "oracle", "target_only")
    lambda_max: float = 1.0                    # [P] hệ số GRL cuối lịch (DANN/CDAN)
    lambda_gamma: float = 10.0
    coral_weight: float = 1.0
    mmd_weight: float = 1.0
    mcc_weight: float = 1.0
    mcc_temp: float = 2.5
    tent_steps: int = 200
    tent_lr: float = 1e-3
    # ---- [V2] RCST: tự huấn luyện nhất quán mức bản ghi (Record-Consistent Self-Training) ----------
    #   * mọi frame của một bản ghi đích chung MỘT nhãn -> nhãn giả lấy từ TRUNG BÌNH xác suất của bản ghi,
    #     chọn cân bằng theo thứ hạng (top/bottom) với tỉ lệ tăng dần rcst_sel (đầu, cuối);
    #   * hàm phạt nhất quán: phương sai xác suất giữa các frame của cùng bản ghi trong lô;
    #   * lô đích gồm rcst_bag = (số bản ghi, số frame mỗi bản ghi); chỉ dùng bản ghi tgt_train (quy nạp).
    rcst_start_frac: float = 0.3
    rcst_sel: tuple = (0.2, 0.6)
    rcst_pl_weight: float = 1.0
    rcst_cons_weight: float = 1.0
    rcst_bag: tuple = (8, 8)

    # ---- Giao thức / Protocol -----------------------------------------------
    target_protocol: str = "inductive"         # [P] inductive (K-fold bản ghi đích) | transductive
    tgt_folds: int = 5
    src_val_frac: float = 0.2                  # bản ghi nguồn giữ lại (chọn mô hình, ngưỡng, DEV)
    seeds: tuple = (42, 43, 44)                # CV lặp: mỗi seed = phân hoạch mới + khởi tạo mới

    # ---- Huấn luyện / Training ----------------------------------------------
    epochs: int = 20                           # [V2] v1: 30; AUC đích ổn định từ epoch ~15-20 (Hình 16 v1)
    iters_per_epoch: int = 80                  # "epoch" = số vòng lặp cố định (chuẩn của DANN)
    batch_size: int = 64                       # MỖI miền trong một vòng lặp
    lr: float = 3e-4
    disc_lr_mult: float = 10.0                 # [V2] như bản gốc DANN/CDAN
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    label_smoothing: float = 0.05
    grad_clip: float = 5.0
    amp: bool = True
    class_balanced_source: bool = True         # lô nguồn cân bằng lớp (giảm ảnh hưởng dịch chuyển nhãn)
    augment: bool = True
    aug_noise_p: float = 0.5
    aug_noise_snr_db: tuple = (10.0, 30.0)
    aug_fir_p: float = 0.5                     # đáp ứng kênh ngẫu nhiên (bất biến cảm biến)
    aug_fir_taps: int = 48
    aug_shift_p: float = 0.5
    # [V2] NGẪU NHIÊN HOÁ CHUỖI THU (có cơ sở vật lý): thông cao ngẫu nhiên (mô phỏng cảm biến/máy ghi cắt
    #      tần thấp khác nhau), độ dốc phổ ngẫu nhiên, thông thấp ngẫu nhiên — áp trong miền tần số, theo mẫu.
    aug_chain_p: float = 0.5
    aug_hp_hz: tuple = (20.0, 200.0)
    aug_tilt_db_oct: tuple = (-3.0, 3.0)
    aug_lp_hz: tuple = (3500.0, 8000.0)
    eval_every: int = 2                        # đánh giá mỗi ngần này epoch
    select_rule: str = "last"                  # [P] last | src_val | dev   (oracle chỉ để đo lạc quan)
    select_from_frac: float = 0.3              # không chọn checkpoint trước 30% lịch (λ còn nhỏ)
    dev_max: int = 1500                        # số frame tối đa mỗi phía cho DEV/PAD

    # ---- Thống kê / Statistics ----------------------------------------------
    record_agg: str = "p_mean"                 # [P] xác suất bản ghi = trung bình xác suất frame
    threshold: float = 0.5                     # [P] ngưỡng quyết định mức bản ghi
    bootstrap_n: int = 2000

    # ---- Thí nghiệm bổ sung / Extra experiments -----------------------------
    run_lambda_sweep: bool = True
    lambda_grid: tuple = (0.1, 0.3, 3.0)       # cộng với lambda_max của lần chạy chính
    run_ablation: bool = True
    # [V2] weak_disc = bộ phân biệt của v1 (256, lr x1); no_chain_aug = RCST không ngẫu nhiên hoá chuỗi thu;
    #      rcst_no_pl / rcst_no_cons = bỏ từng thành phần của RCST; so_chain_aug = source-only không chain-aug.
    ablation_variants: tuple = ("weak_disc", "no_chain_aug", "rcst_no_pl", "rcst_no_cons", "so_no_chain_aug")
    ablation_seeds: tuple = (42, 43, 44)       # [V2] v1 chỉ 1 seed -> kết luận ablation mong manh
    run_shortcut_check: bool = True            # [V2] mục 3.9: dải nào phân biệt TRONG miền / chuyển giao được
    sweep_seeds: tuple = (42,)
    sweep_task: int = 0
    run_ssda: bool = False                     # đường cong bán giám sát (tốn thêm thời gian)
    ssda_shots: tuple = (1, 2, 4)              # số bản ghi đích có nhãn MỖI lớp
    run_band_occlusion: bool = True
    tsne_max_points: int = 3000

    # ---- Phần cứng & đầu ra / Hardware and outputs --------------------------
    parallel_jobs: bool = True                 # mỗi GPU chạy một job (không dùng DataParallel)
    max_parallel: int = 2
    gpu_resident_max_gb: float = 6.0
    out_dir: str = "/kaggle/working"
    fig_dpi: int = 300
    save_models: bool = True
    resume: bool = True                        # dùng lại kết quả job đã chạy (cùng cấu hình)
    # [V2.1] chạy tiếp qua NHIỀU phiên Kaggle: tìm ada_results.pkl (cùng dấu vân tay cấu hình) trong các thư mục
    #        này — thêm output của phiên bản trước làm Input là đủ; chỉ các job còn thiếu được chạy.
    resume_dirs: tuple = ("/kaggle/input",)
    heartbeat_min: float = 5.0                 # [V2.1] luồng chính in tiến độ mỗi N phút (kể cả khi job dài)
    stall_min: float = 20.0                    # [V2.1] một GPU không tiến triển N phút -> in stack trace mọi luồng
    save_every_min: float = 3.0                # [V2.1] lưu ada_results.pkl tối đa mỗi N phút (v1/v2: mỗi 10 job)

    smoke_test: bool = False


cfg = CFG()
if os.environ.get("LEAK_SMOKE") == "1":
    cfg.smoke_test = True
if os.environ.get("LEAK_DATA") is not None:
    cfg.data_root = os.environ["LEAK_DATA"]
if os.environ.get("LEAK_OUT"):
    cfg.out_dir = os.environ["LEAK_OUT"]
elif not Path("/kaggle").exists():
    cfg.out_dir = str(Path("./ada_outputs").resolve())

if cfg.smoke_test:
    print("!! SMOKE TEST: vài vòng lặp, 2 fold, 1 seed — chỉ để kiểm tra toàn tuyến.")
    cfg.epochs, cfg.iters_per_epoch, cfg.batch_size, cfg.eval_every = 4, 4, 16, 1
    cfg.tgt_folds, cfg.seeds, cfg.bootstrap_n = 2, (42,), 100
    cfg.methods = ("source_only", "joint_bn", "adabn", "tent", "coral", "mmd", "mcc", "dann", "cdan_e", "rcst",
                   "oracle", "target_only")
    cfg.lambda_grid, cfg.ablation_variants = (0.3,), ("weak_disc", "rcst_no_pl")
    cfg.ablation_seeds, cfg.tent_steps, cfg.rcst_bag = (42,), 5, (4, 4)
    cfg.run_ssda, cfg.ssda_shots = True, (1,)
    cfg.tsne_max_points, cfg.dev_max = 300, 200
    cfg.heartbeat_min = 0.2

if not torch.cuda.is_available():
    cfg.amp = False
    cfg.parallel_jobs = False

OUT = Path(cfg.out_dir)
FIG_DIR = OUT / "figures";      FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR = OUT / "tables";       TAB_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUT / "checkpoints"; CKPT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT / "cache";      CACHE_DIR.mkdir(parents=True, exist_ok=True)

FRAME_LEN = int(round(cfg.frame_sec * cfg.target_sr))
HOP_LEN = int(round(cfg.hop_sec * cfg.target_sr))
TOTAL_ITERS = int(cfg.epochs * cfg.iters_per_epoch)
SELECT_FROM_EPOCH = int(math.ceil(cfg.select_from_frac * cfg.epochs))
assert cfg.select_rule in ("last", "src_val", "dev"), "select_rule chính phải KHÔNG dùng nhãn đích"
assert cfg.epochs > SELECT_FROM_EPOCH

_cfg_print = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()}
print(json.dumps(_cfg_print, indent=1, ensure_ascii=False))
print(f"\nFrame {cfg.frame_sec}s = {FRAME_LEN} mẫu | hop {cfg.hop_sec}s | "
      f"{cfg.epochs} epoch x {cfg.iters_per_epoch} vòng = {TOTAL_ITERS} vòng lặp/job")
print(f"Chọn checkpoint chính: '{cfg.select_rule}' (chỉ từ epoch {SELECT_FROM_EPOCH}) | "
      f"giao thức đích: {cfg.target_protocol} | seeds {list(cfg.seeds)}")
print("Thư mục ra:", OUT)
''')

code(r'''
# =========================================================
# 1.3 Tái lập, phong cách hình vẽ, xuất bảng / Reproducibility, plot style, table export
# =========================================================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True
    return int(seed)

set_seed(cfg.seeds[0])

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": cfg.fig_dpi, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white",
    "figure.constrained_layout.use": True,
})
PAL = {"leak": "#C8553D", "noleak": "#2E6F95", "grey": "#6E7B8B", "warn": "#E0A458",
       "src": "#2E6F95", "tgt": "#C8553D", "ok": "#3F7D20"}
DOMAIN_PALETTE = ["#2E6F95", "#C8553D", "#3F7D20", "#8C6BB1", "#E0A458", "#4C9F9F", "#A0522D"]
METHOD_LABEL = {"source_only": "Source-only", "joint_bn": "Joint-BN (λ=0)", "adabn": "AdaBN", "tent": "TENT",
                "coral": "Deep CORAL", "mmd": "DAN (MK-MMD)", "mcc": "MCC", "dann": "DANN", "cdan": "CDAN",
                "cdan_e": "CDAN+E", "rcst": "RCST (ours)", "oracle": "Oracle (S+T labelled)",
                "target_only": "Target-only (labelled)", "oracle_shots": "Source + few target labels",
                "dann_shots": "DANN + few target labels"}
METHOD_COLOR = {"source_only": "#6E7B8B", "joint_bn": "#9AA5B1", "adabn": "#4C9F9F", "tent": "#2A9D8F",
                "coral": "#8C6BB1", "mmd": "#A0522D", "mcc": "#B5838D", "dann": "#2E6F95", "cdan": "#E0A458",
                "cdan_e": "#C8553D", "rcst": "#D62828", "oracle": "#3F7D20", "target_only": "#6A994E",
                "oracle_shots": "#3F7D20", "dann_shots": "#2E6F95"}
# dùng đích không nhãn khi huấn luyện (lô trộn nguồn + đích qua cùng một lượt truyền)
ADAPT_METHODS = ("joint_bn", "coral", "mmd", "mcc", "dann", "cdan", "cdan_e", "rcst")
ADV_METHODS = ("dann", "cdan", "cdan_e", "rcst")                # có bộ phân biệt miền + GRL
POSTHOC_METHODS = ("adabn", "tent")                             # dựng từ mô hình source-only, không huấn luyện lại

def mlabel(m):
    return METHOD_LABEL.get(m, m)

FIG_INDEX, TAB_INDEX = {}, {}

def savefig(fig, name, caption=""):
    # Lưu PNG (xem nhanh) + PDF (vector cho manuscript) và HIỂN THỊ ngay trong notebook.
    n = len(FIG_INDEX) + 1
    stem = f"Fig{n:02d}_{name}"
    fig.savefig(FIG_DIR / f"{stem}.png")
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    FIG_INDEX[stem] = caption
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110)      # bản xem trước, không phụ thuộc backend
    display(IPyImage(data=buf.getvalue()))
    plt.close(fig)
    if caption:
        display(Markdown(f"**Figure {n}.** {caption}"))
    return stem

def _fmt(v, nd=4):
    if isinstance(v, (float, np.floating)):
        return "–" if not np.isfinite(v) else f"{v:.{nd}f}"
    return str(v)

def _tex_escape(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("→", r"$\rightarrow$"),
                 ("±", r"$\pm$"), ("Δ", r"$\Delta$"), ("λ", r"$\lambda$")]:
        s = s.replace(a, b)
    return s

def df_to_latex(df, caption, label, nd=4):
    cols = list(df.columns)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{" + _tex_escape(caption) + "}", r"\label{" + label + "}",
             r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule",
             " & ".join(_tex_escape(str(c)) for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(_tex_escape(_fmt(r[c], nd)) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)

def df_to_markdown(df, nd=4):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(_fmt(r[c], nd) for c in cols) + " |")
    return "\n".join(out)

def save_table(df, name, caption="", nd=4):
    # CSV (số liệu gốc) + LaTeX booktabs + Markdown, đánh số tự động như hình.
    n = len(TAB_INDEX) + 1
    stem = f"Table{n:02d}_{name}"
    df = df.reset_index(drop=True)
    df.to_csv(TAB_DIR / f"{stem}.csv", index=False)
    (TAB_DIR / f"{stem}.tex").write_text(df_to_latex(df, caption, f"tab:{name}", nd), encoding="utf-8")
    (TAB_DIR / f"{stem}.md").write_text(df_to_markdown(df, nd), encoding="utf-8")
    TAB_INDEX[stem] = caption
    display(Markdown(f"**Table {n}.** {caption}"))
    with pd.option_context("display.max_columns", 50, "display.width", 250):
        display(df.round(nd) if hasattr(df, "round") else df)
    return stem

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
''')

# =============================================================================
# 2. DỮ LIỆU ĐA MIỀN
# =============================================================================
md(r'''
## 2. Đọc dữ liệu và nhận diện miền / Data discovery and domain inventory

Ba định danh được trích cho **mỗi tệp**:

| Định danh | Nguồn | Vai trò |
|---|---|---|
| `label` | thư mục `leak/` / `noleak/` (ưu tiên), rồi tên tệp | nhãn mức bản ghi |
| `domain` | `cfg.domain_roots` (mỗi dataset một miền) **hoặc** tự dò: thư mục miền → phần còn lại của thư mục nhãn (`leak_16k/`) → ký hiệu trong tên tệp → chuỗi thu 8/16 kHz (hoặc regex / CSV) | miền thu âm |
| `record_uid = domain\|Rxxx` | mã `Rxxx` trong tên tệp **ghép với miền** | **đơn vị nhóm và đơn vị thống kê** |

> Ghép miền vào mã bản ghi là bắt buộc: tập *Mix* có thể chứa `R001` ở cả hai miền nhưng là hai lần thu khác nhau.
> Ngược lại, nếu **cùng một lần thu** xuất hiện ở hai miền (ví dụ bản 8 kHz và bản 16 kHz), mục 3.6 sẽ phát hiện
> nhờ so khớp đường bao năng lượng — đây là rò rỉ dữ liệu nguồn↔đích phải loại bỏ trước khi báo cáo.
''')

code(r'''
# =========================================================
# 2.1 Gán nhãn theo từ + dò thư mục dữ liệu / Word-level labelling and dataset discovery
#     Hỗ trợ NHIỀU dataset = nhiều miền qua cfg.domain_roots (ví dụ tập Mix + tập hiện trường v7).
# =========================================================
def _norm_token(s):
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.lower().replace("-", "_").replace(" ", "_")

def _key_regex(keys):
    ks = sorted({_norm_token(k) for k in keys}, key=len, reverse=True)
    return re.compile(r"(?:^|[_.])(?:" + "|".join(map(re.escape, ks)) + r")(?=$|[_.\d])")

_NOLEAK_RE = _key_regex(cfg.noleak_keys)
_LEAK_RE = _key_regex(cfg.leak_keys)

def _label_of_token(s):
    s = _norm_token(s)
    if _NOLEAK_RE.search(s):
        return 0
    if _LEAK_RE.search(s):
        return 1
    return None

def label_from_rel(rel):
    # Thư mục chứa tệp -> các thư mục cha (gần trước) -> tên tệp
    parts = list(Path(rel).parts)
    for tok in list(reversed(parts[:-1])) + [parts[-1]]:
        lab = _label_of_token(tok)
        if lab is not None:
            return lab, tok
    return None, None

for _t, _w in [("leak", 1), ("noleak", 0), ("No-Leak", 0), ("R001_leak01", 1), ("R12_noleak_03", 0),
               ("rò_rỉ", 1), ("khong_ro_ri", 0), ("leakage", None), ("tapdulieuMix_16k", None),
               ("field", None), ("lab", None), ("leak_16k", 1), ("noleak_tt", 0)]:
    assert _label_of_token(_t) == _w, f"bộ gán nhãn sai với '{_t}'"

def _audio_in(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix in cfg.audio_ext)

def _has_both(root, files):
    labs = {label_from_rel(p.relative_to(root))[0] for p in files}
    return {0, 1}.issubset(labs)

_BASES = [b for b in (Path("/kaggle/input"), Path("./data")) if b.exists()]
_key = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())   # Kaggle: "tapdulieuMix_16k" -> slug "tapdulieumix-16k"

def find_dir(path_or_name):
    # Đường dẫn tồn tại -> dùng luôn; nếu không, tìm thư mục cùng TÊN (bỏ qua hoa/thường, ký tự đặc biệt)
    p = Path(str(path_or_name))
    if str(path_or_name) and p.exists():
        return p
    for base in _BASES:
        for d in sorted(base.rglob("*")):
            if d.is_dir() and _key(d.name) == _key(p.name):
                fs = _audio_in(d)
                if fs and _has_both(d, fs):
                    return d
    return None

def discover_root(explicit="", hint=""):
    if explicit:
        ex = Path(explicit)
        if ex.exists():
            fs = _audio_in(ex)
            if fs and _has_both(ex, fs):
                return ex, fs
            print(f"!! cfg.data_root = {ex} tồn tại nhưng không chứa đủ hai lớp leak/noleak.")
        else:
            print(f"!! cfg.data_root = {ex} KHÔNG tồn tại -> tự dò.")
    if hint:                                                # (1) thư mục trùng tên gợi ý
        d = find_dir(hint)
        if d is not None:
            return d, _audio_in(d)
    for base in _BASES:                                     # (2) tổ tiên chung thấp nhất của mọi tệp có nhãn
        fs = [p for p in _audio_in(base) if label_from_rel(p.relative_to(base))[0] is not None]
        if fs and _has_both(base, fs):
            root = Path(os.path.commonpath([str(p.parent) for p in fs]))
            while _label_of_token(root.name) is not None:
                root = root.parent
            return root, _audio_in(root)
    return None, []

def candidate_dataset_roots(max_depth=5):
    # Mọi thư mục trong /kaggle/input có thư mục con leak/ VÀ noleak/ -> gợi ý cho cfg.domain_roots
    out = []
    for base in _BASES:
        stack = [(base, 0)]
        while stack:
            d, k = stack.pop()
            try:
                kids = [x for x in d.iterdir() if x.is_dir()]
            except Exception:
                continue
            if {0, 1}.issubset({_label_of_token(x.name) for x in kids}):
                out.append(d)
            if k < max_depth:
                stack += [(x, k + 1) for x in kids]
    return sorted(out)

# ROOTS: danh sách (tên miền cố định hoặc None, thư mục gốc, danh sách tệp)
ROOTS = []
if cfg.domain_roots:
    for _name, _pth in cfg.domain_roots.items():
        _d = find_dir(_pth)
        if _d is None:
            print(f"!! Không tìm thấy thư mục của miền '{_name}': {_pth}")
            print("   Các thư mục trong /kaggle/input có leak/ + noleak/ (chép đường dẫn đúng vào cfg.domain_roots):")
            for _c in candidate_dataset_roots():
                print("     ", _c)
            raise FileNotFoundError(f"cfg.domain_roots['{_name}'] = {_pth} không tồn tại")
        _fs = _audio_in(_d)
        assert _fs and _has_both(_d, _fs), f"Miền '{_name}' ({_d}) không có đủ hai lớp leak/noleak"
        ROOTS.append((str(_name), _d, _fs))
        print(f"Miền '{_name}': {_d} | {len(_fs)} tệp")
    DATA_ROOT, AUDIO_FILES, SYNTHETIC = ROOTS[0][1], [p for r in ROOTS for p in r[2]], False
else:
    DATA_ROOT, AUDIO_FILES = discover_root(cfg.data_root, cfg.dataset_hint)
    SYNTHETIC = DATA_ROOT is None
    if not SYNTHETIC:
        ROOTS = [(None, DATA_ROOT, AUDIO_FILES)]
if SYNTHETIC:
    print("!! KHÔNG tìm thấy dataset -> SYNTHETIC DEMO MODE (dữ liệu mô phỏng hai miền).")
else:
    print("Dataset root:", DATA_ROOT, "|", len(AUDIO_FILES), "tệp âm thanh")

def print_tree(root, max_depth=3, max_children=12):
    # Cây thư mục kèm số tệp âm thanh -> giúp cấu hình miền khi tự dò sai.
    root = Path(root)
    def _walk(d, depth):
        kids = sorted(k for k in d.iterdir() if k.is_dir())
        for k in kids[:max_children]:
            n = len(_audio_in(k))
            print("  " * depth + f"├─ {k.name}/  ({n} tệp)")
            if depth + 1 < max_depth:
                _walk(k, depth + 1)
        if len(kids) > max_children:
            print("  " * depth + f"└─ ... (+{len(kids) - max_children} thư mục)")
    print(f"{root.name}/  ({len(_audio_in(root))} tệp)")
    _walk(root, 0)
''')

code(r'''
# =========================================================
# 2.2 Dữ liệu mô phỏng HAI MIỀN / Two-domain synthetic data (chỉ khi không có dữ liệu thật)
#     "lab"  : nền nhiễu hồng, rò rỉ băng hẹp 350-900 Hz SNR cao, băng thông đầy đủ 8 kHz.
#     "field": nền nhiễu nâu + xe cộ + bơm, rò rỉ yếu hơn & dịch xuống 250-700 Hz, cảm biến có
#              độ dốc phổ khác, chuỗi thu 8 kHz nâng lên 16 kHz (không có năng lượng > 4 kHz),
#              tỉ lệ lớp khác "lab" (dịch chuyển nhãn), mã Rxxx TRÙNG với "lab".
# =========================================================
def _synthesise_mixed(root, spec, files_per_rec=2, dur=24.0, sr=16000, seed=0, layout="folder"):
    rng = np.random.default_rng(seed)
    root = Path(root)
    if root.exists():
        try:
            shutil.rmtree(root)
        except Exception:
            root = Path(str(root) + "_" + time.strftime("%H%M%S"))
    n = int(dur * sr); t = np.arange(n) / sr
    f = np.fft.rfftfreq(n, 1 / sr); f[0] = f[1]

    def colored(alpha):
        X = np.fft.rfft(rng.standard_normal(n))
        return np.fft.irfft(X / f ** (alpha / 2), n)

    prof = {"lab": dict(alpha=1.0, snr=(-2, 8), f0=(350, 900), bw=(80, 200), hum=0.02,
                        tr=(0, 2), pump=0.0, chain8k=False, tilt=0.0),
            "field": dict(alpha=1.6, snr=(-8, 2), f0=(250, 700), bw=(60, 160), hum=0.08,
                          tr=(3, 7), pump=0.5, chain8k=True, tilt=0.6)}

    def _dst(dom, cls, rid, seg):
        name = f"R{rid:03d}_{cls}{seg:02d}.wav"
        if layout == "labelfolder":
            d = root / f"{cls}_{dom}"
        elif layout == "filename":
            d, name = root / cls, f"{dom}_{name}"
        elif layout == "bandwidth":
            d = root / cls
        elif layout == "mixrs":                                         # như tapdulieuMix_16k: leak/, noleak/ phẳng
            d = root / cls
            name = (f"S{rid:02d}_{cls}{seg:02d}.wav" if dom == "lab" else f"R{rid:03d}_{cls}{seg:02d}.wav")
        else:                                                           # folder, roots
            d = root / dom / cls
        d.mkdir(parents=True, exist_ok=True)
        return d / name

    for di, (dom, n_leak, n_no) in enumerate(spec):
        P = prof[dom]
        rid = 100 * di if layout == "bandwidth" else 0                  # thư mục phẳng: tránh trùng tên tệp
        for cls, n_r in (("leak", n_leak), ("noleak", n_no)):
            if layout == "mixrs" and dom == "lab":
                rid = 0                                                 # chuỗi S đánh số lại trong mỗi lớp
            for _ in range(n_r):
                rid += 1
                f0 = rng.uniform(*P["f0"]); bw = rng.uniform(*P["bw"])
                for seg in range(1, files_per_rec + 1):
                    x = colored(P["alpha"]); x /= np.std(x) + 1e-12
                    for h in range(1, 4):
                        x += P["hum"] * np.sin(2 * np.pi * 50 * h * t + rng.uniform(0, 6.28))
                    if cls == "leak":
                        b, a = sps.butter(4, [max(20., f0 - bw) / (sr / 2), (f0 + bw) / (sr / 2)], "band")
                        lk = sps.lfilter(b, a, rng.standard_normal(n)); lk /= np.std(lk) + 1e-12
                        am = 1 + 0.3 * np.sin(2 * np.pi * rng.uniform(.05, .3) * t)
                        x = x + lk * am * 10 ** (rng.uniform(*P["snr"]) / 20)
                    elif rng.random() < P["pump"]:                       # tiếng bơm: nhầm lẫn dễ có
                        fp = rng.uniform(100, 300)
                        x = x + 0.3 * np.sin(2 * np.pi * fp * t) * (1 + 0.2 * rng.standard_normal(n))
                    for _ in range(int(rng.integers(*P["tr"])) if P["tr"][1] > 0 else 0):
                        s0 = int(rng.integers(0, n - sr))
                        x[s0:s0 + sr] += 3.0 * np.exp(-np.arange(sr) / (sr * .05)) * rng.standard_normal(sr)
                    if P["tilt"]:
                        x = sps.lfilter([1 - P["tilt"]], [1, -P["tilt"]], x)
                    if P["chain8k"]:                                     # thu 8 kHz rồi nâng lên 16 kHz
                        x = sps.resample_poly(sps.resample_poly(x, 1, 2), 2, 1)[:n]
                    x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
                    sf.write(_dst(dom, cls, rid, seg), x.astype(np.float32), sr)
    return root

if SYNTHETIC:
    # LEAK_SYN_LAYOUT chọn bố cục thư mục mô phỏng để kiểm thử bộ nhận miền:
    #   folder (mặc định) <miền>/<lớp>/ | labelfolder <lớp>_<miền>/ | filename <lớp>/<miền>_Rxxx...
    #   bandwidth <lớp>/ phẳng, miền chỉ khác chuỗi thu | roots: mỗi miền một thư mục gốc | single: 1 miền
    #   mixrs: như tapdulieuMix_16k (<lớp>/Rxxx_* hiện trường + <lớp>/Sxx_* thí nghiệm, Sxx đánh số lại theo lớp)
    _layout = os.environ.get("LEAK_SYN_LAYOUT", "folder")
    _n = (5, 4) if cfg.smoke_test else (14, 10)
    _spec = ((("lab",) + _n),) if _layout == "single" else (("lab",) + _n, ("field",) + _n[::-1])
    _fpr, _dur = (1, 12.0) if cfg.smoke_test else (2, 24.0)
    DATA_ROOT = _synthesise_mixed(OUT / "synthetic_mix_16k", _spec, _fpr, _dur, cfg.target_sr,
                                  layout="bandwidth" if _layout == "single" else _layout)
    AUDIO_FILES = _audio_in(DATA_ROOT)
    ROOTS = ([(d, DATA_ROOT / d, _audio_in(DATA_ROOT / d)) for d in ("lab", "field")] if _layout == "roots"
             else [(None, DATA_ROOT, AUDIO_FILES)])
    print(f"Synthetic dataset ({_layout}):", DATA_ROOT, "|", len(AUDIO_FILES), "tệp")

for _name, _root, _ in ROOTS:
    if _name:
        print(f"\n[miền '{_name}']")
    print_tree(_root)
''')

code(r'''
# =========================================================
# 2.3 Bảng kiểm kê mức TỆP + NHẬN DIỆN MIỀN / File-level manifest and domain assignment
#     cfg.domain_source = "auto" thử lần lượt các cách dưới đây và lấy cách ĐẦU TIÊN cho >= 2 miền
#     hợp lệ (mỗi miền >= 2 bản ghi, >= 2 miền có đủ hai lớp, >= 1 miền đủ điều kiện làm đích):
#       regex        : cfg.domain_regex trên đường dẫn tương đối (nếu được đặt)
#       folder       : thư mục KHÔNG phải nhãn          tapdulieuMix_16k/<miền>/leak/...
#       label_folder : phần còn lại của thư mục nhãn    leak_16k/, noleak_tt/       -> "16k", "tt"
#       filename     : phần còn lại của tên tệp         lab_R001_leak01.wav         -> "lab"
#       bandwidth    : chuỗi thu đo từ phổ (năng lượng 5-7.5 kHz so với 2.5-3.5 kHz) -> chain_8k / chain_16k
#     cfg.domain_roots (mỗi dataset một miền) và "csv" (bảng gán miền theo tệp) luôn được ưu tiên.
# =========================================================
_REC_RE = re.compile(cfg.record_regex, re.IGNORECASE)
_DERIV = tuple(_norm_token(m).strip("_") for m in cfg.derivative_markers)

def extract_record_id(path):
    m = _REC_RE.search(path.stem)
    if m:
        return m.group(1).upper(), "regex:stem"
    m = _REC_RE.search(str(path.parent))
    if m:
        return m.group(1).upper(), "regex:parent"
    return path.stem, "fallback:stem"

def _residual(s, drop_rid=False):
    # Bỏ từ nhãn (leak/noleak...), mã Rxxx, số thứ tự, đánh dấu tệp dẫn xuất -> phần còn lại = ứng viên miền.
    s = _LEAK_RE.sub("_", _NOLEAK_RE.sub("_", _norm_token(s)))
    if drop_rid:
        s = _REC_RE.sub("_", s)
    for m in _DERIV:
        s = s.replace(m, "_")
    toks = []
    for t in re.split(r"[_.\s]+", s):
        if not t or t.isdigit():
            continue
        if re.fullmatch(r"[a-z]+\d+", t):            # seg01 -> seg ; giữ 16k, 8k
            t = re.sub(r"\d+$", "", t)
        toks.append(t)
    return "_".join(toks)

def _domain_dirs(rel):
    return [d for d in Path(rel).parts[:-1] if _label_of_token(d) is None]

def chain_hf_db(path, max_sec=20.0):
    # Trung vị PSD 5-7.5 kHz so với 2.5-3.5 kHz (dB) — tránh dải chuyển tiếp của bộ lọc chống chồng phổ
    # quanh 4 kHz. Thu 8 kHz rồi nâng lên 16 kHz: thường < -60 dB; thu gốc 16 kHz: thường > -10 dB.
    try:
        info = sf.info(str(path)); sr = int(info.samplerate)
        if sr < 12000:
            return -120.0
        x, _ = sf.read(str(path), frames=int(min(info.frames, max_sec * sr)), dtype="float32", always_2d=True)
        x = x.mean(1); x = x - x.mean()
        f, P = sps.welch(x, fs=sr, nperseg=min(4096, len(x)))
        hi = np.median(P[(f >= 5000) & (f <= min(7500, sr / 2 - 200))])
        lo = np.median(P[(f >= 2500) & (f <= 3500)])
        return float(10 * np.log10((hi + 1e-30) / (lo + 1e-30)))
    except Exception:
        return np.nan

# ---- Lượt 1: nhãn, mã bản ghi, thông tin tệp (chưa gán miền) ----
rows = []
for _name, _root, _fs in ROOTS:
    for p_ in _fs:
        rel = p_.relative_to(_root)
        lab, tok = label_from_rel(rel)
        if lab is None:
            continue
        try:
            info = sf.info(str(p_))
        except Exception as e:
            print("  [bỏ qua]", rel, type(e).__name__); continue
        rid, how = extract_record_id(p_)
        rows.append(dict(path=str(p_), root=str(_root), fixed_domain=_name, rel=str(rel), file_stem=p_.stem,
                         record_id=rid, rid_source=how, label=int(lab), cls="leak" if lab == 1 else "noleak",
                         label_token=tok, duration_s=float(info.duration), sr_native=int(info.samplerate),
                         channels=int(info.channels)))
files = pd.DataFrame(rows)
assert len(files), "Không đọc được tệp âm thanh nào."

# ---- Các cách gán miền ----
def cand_folder(df):
    dirs = [_domain_dirs(r) for r in df.rel]
    w = 0                                            # bỏ các thư mục "bọc" chung cho mọi tệp
    while dirs and all(len(d) > w + 1 for d in dirs) and len({d[w] for d in dirs}) == 1:
        w += 1
    return pd.Series(["/".join(d[w:w + cfg.domain_depth]) for d in dirs], index=df.index)

def cand_label_folder(df):
    out = []
    for r in df.rel:
        labdirs = [p for p in Path(r).parts[:-1] if _label_of_token(p) is not None]
        out.append(_residual(labdirs[-1]) if labdirs else "")
    return pd.Series(out, index=df.index)

def cand_filename(df):
    return pd.Series([_residual(Path(r).stem, drop_rid=True) for r in df.rel], index=df.index)

def cand_regex(df):
    def one(r):
        m = re.search(cfg.domain_regex, str(r), re.IGNORECASE)
        return m.group(cfg.domain_regex_group) if m else ""
    return pd.Series([one(r) for r in df.rel], index=df.index)

_HF_DB = {}
def cand_bandwidth(df):
    if not _HF_DB:
        t0 = time.time()
        for p in df.path:
            _HF_DB[p] = chain_hf_db(p)
        print(f"  (đo chuỗi thu của {len(df)} tệp trong {time.time() - t0:.0f}s)")
    v = df.path.map(_HF_DB)
    lab = pd.Series(np.where(v < cfg.bw_split_db, "chain_8k", "chain_16k"), index=df.index).where(v.notna(), "")
    # một bản ghi (gốc × lớp × Rxxx) phải thuộc MỘT chuỗi thu -> lấy đa số
    key = df.root + "|" + df.cls + "|" + df.record_id
    maj = lab.groupby(key).agg(lambda s: s.value_counts().index[0])
    n_mixed = int((lab.groupby(key).nunique() > 1).sum())
    if n_mixed:
        print(f"  !! {n_mixed} bản ghi có tệp thuộc hai chuỗi thu khác nhau -> gán theo đa số")
    return key.map(maj)

def cand_csv(df):
    dm = pd.read_csv(cfg.domain_csv)
    keycol = next((c for c in ("rel", "file", "file_stem", "record_id") if c in dm.columns), None)
    assert "domain" in dm.columns and keycol, "CSV miền cần cột 'domain' và một trong: rel, file, file_stem, record_id"
    dm = dm[dm.domain.notna() & (dm.domain.astype(str).str.strip() != "")]
    nk = lambda v: str(v).strip().replace("\\", "/").lower()
    M = {nk(k): str(d).strip() for k, d in zip(dm[keycol], dm.domain)}
    keys = {"rel": df.rel, "file": df.rel.map(lambda r: Path(r).name),
            "file_stem": df.file_stem, "record_id": df.record_id}[keycol]
    return keys.map(lambda k: M.get(nk(k), ""))

CANDIDATES = {"regex": cand_regex, "folder": cand_folder, "label_folder": cand_label_folder,
              "filename": cand_filename, "bandwidth": cand_bandwidth, "csv": cand_csv}

def assess(dom, df):
    d = dom.replace("", "unnamed")
    rec = (df.assign(d=d).groupby(["d", "label"]).record_id.nunique().unstack(fill_value=0)
           .reindex(columns=[0, 1], fill_value=0))
    both = int(((rec[0] >= 1) & (rec[1] >= 1)).sum())
    elig = int(((rec[0] >= cfg.min_records_per_class) & (rec[1] >= cfg.min_records_per_class)).sum())
    ok = (2 <= len(rec) <= cfg.max_auto_domains and both >= 2 and elig >= 1 and bool((rec.sum(1) >= 2).all()))
    return ok, d, rec

free = files.fixed_domain.isna()
files["domain"] = files.fixed_domain.fillna("")
files["domain_source"] = np.where(free, "", "root")
CAND_REPORT, chosen = [], None
if free.any():
    sub = files[free]
    if cfg.domain_source in ("csv", "regex", "folder", "label_folder", "filename", "bandwidth"):
        order = [cfg.domain_source]
    else:
        order = (["regex"] if cfg.domain_regex else []) + ["folder", "label_folder", "filename", "bandwidth"]
    for meth in order:
        dom = CANDIDATES[meth](sub)
        ok, d, rec = assess(dom, sub)
        vals = ", ".join(f"{k} ({int(r[0])} noleak/{int(r[1])} leak)" for k, r in rec.head(8).iterrows())
        CAND_REPORT.append(dict(method=meth, n_domains=len(rec), valid=ok, domains=vals))
        if ok or cfg.domain_source != "auto":
            chosen = meth
            # với cách khai báo tay, tệp không gán được miền bị LOẠI (không gộp vào "unnamed")
            files.loc[sub.index, "domain"] = (d if cfg.domain_source == "auto" else dom).values
            files.loc[sub.index, "domain_source"] = meth
            break
    if chosen is None:                                 # một miền duy nhất -> cờ "all" để mục chẩn đoán xử lý
        files.loc[sub.index, "domain"] = "all"; files.loc[sub.index, "domain_source"] = "none"
display(pd.DataFrame(CAND_REPORT)) if CAND_REPORT else None
print("Cách gán miền được dùng:", chosen or ("domain_roots" if not free.any() else "KHÔNG CÓ"))

n_drop = int((files.domain == "").sum())
if n_drop:
    print(f"!! {n_drop} tệp không gán được miền theo '{cfg.domain_source}' -> loại khỏi phân tích.")
files = files[files.domain != ""].copy()
files["domain"] = files.domain.map(lambda d: cfg.domain_alias.get(d, d))
if cfg.exclude_domains:
    files = files[~files.domain.isin(cfg.exclude_domains)]
# [V2] ghép LỚP vào mã bản ghi khi chuỗi đánh số lại trong mỗi lớp (S01_leak ≠ S01_noleak)
files["record_uid"] = files.domain + "|" + files.record_id + (("|" + files.cls) if cfg.record_id_by_class else "")
files = files.sort_values(["domain", "cls", "record_uid", "file_stem"]).reset_index(drop=True)
files["file_id"] = np.arange(len(files))
files.to_csv(OUT / "manifest_files.csv", index=False)

print(f"Số tệp: {len(files)} | số miền: {files.domain.nunique()} -> {sorted(files.domain.unique())}")
print("Trích mã bản ghi   :", files.rid_source.value_counts().to_dict())
print("Tần số lấy mẫu gốc :", files.groupby("domain").sr_native.agg(lambda s: str(sorted(set(s)))).to_dict())

if files.domain.nunique() < 2:
    # ---------------- CHẨN ĐOÁN: vì sao không tách được miền ----------------
    print("\n" + "=" * 100)
    print("!! CHỈ NHẬN RA MỘT MIỀN — thích ứng miền cần ít nhất 2 miền. Thông tin để quyết định cách gán miền:")
    print("=" * 100)
    print("\n(1) Ví dụ đường dẫn tương đối (12 tệp mỗi lớp):")
    for c in ("leak", "noleak"):
        for r in files.loc[files.cls == c, "rel"].head(12):
            print("     ", r)
    print("\n(2) Phần còn lại của TÊN TỆP sau khi bỏ Rxxx/nhãn/số thứ tự (nếu có nhóm rõ ràng -> đó là miền):")
    display(cand_filename(files).replace("", "(trống)").groupby(files.cls).value_counts().unstack(0).fillna(0)
            .astype(int).sort_values("leak", ascending=False).head(15))
    _db = cand_bandwidth(files)
    v = files.path.map(_HF_DB)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for c in ("noleak", "leak"):
        ax.hist(v[files.cls == c].dropna(), bins=40, alpha=.6, color=PAL[c], label=c)
    ax.axvline(cfg.bw_split_db, color="k", ls="--", label=f"ngưỡng {cfg.bw_split_db:g} dB")
    ax.set_xlabel("PSD 5–7.5 kHz so với 2.5–3.5 kHz (dB)"); ax.set_ylabel("số tệp"); ax.legend()
    ax.set_title("(3) Chuỗi thu: hai cụm tách biệt = hai chuỗi thu (8 kHz / 16 kHz)")
    savefig(fig, "domain_diagnostic_chain", "Diagnostic: acquisition-chain indicator per file.")
    tmpl = files[["rel", "file_stem", "record_id", "cls", "duration_s"]].assign(
        hf_db=v.round(1).values, suggested=_db.values, domain="")
    tmpl.to_csv(OUT / "domain_map_template.csv", index=False)
    print(f"\n(4) Đã ghi {OUT / 'domain_map_template.csv'}: điền cột 'domain' cho từng tệp nếu cần gán tay.")
    print("\n(5) Các thư mục trong /kaggle/input có leak/ + noleak/ (có thể dùng làm MIỀN THỨ HAI):")
    for d in candidate_dataset_roots():
        print("      ", d)
    print("""
CÁCH KHẮC PHỤC (chọn một, sửa ở mục 1.2 rồi chạy lại từ đầu):
  A. Tập Mix chỉ là MỘT miền -> thêm dataset thứ hai (ví dụ tập hiện trường v7) làm miền khác:
       cfg.domain_roots = {"mix": "<đường dẫn tapdulieuMix_16k>", "field": "<đường dẫn field_rec_8k_V7.2>"}
  B. Tên tệp có ký hiệu miền (ví dụ 'ptn_R001_leak01', 'R001_tt_leak01'):
       cfg.domain_source = "regex"; cfg.domain_regex = r"(?:^|/)(ptn|tt)_"   (nhóm 1 = tên miền)
  C. Biểu đồ (3) có hai cụm rõ ràng (dữ liệu 8 kHz và 16 kHz trộn lẫn):
       cfg.domain_source = "bandwidth"   (chỉnh cfg.bw_split_db theo khe giữa hai cụm)
  D. Không có quy luật nào ở trên: điền cột 'domain' trong domain_map_template.csv, tải lên Kaggle rồi
       cfg.domain_source = "csv"; cfg.domain_csv = "/kaggle/input/<dataset>/domain_map_template.csv"
""")
    raise ValueError("Cần >= 2 miền cho thích ứng miền — xem chẩn đoán (1)-(5) và cách khắc phục A-D ngay trên.")
files.head(6)
''')

code(r'''
# =========================================================
# 2.3b Kiểm tra chất lượng định danh / Identifier quality checks
# =========================================================
problems = []
conf = files.groupby("record_uid").label.nunique()
if (conf > 1).any():
    bad = conf[conf > 1].index.tolist()
    problems.append(f"{len(bad)} bản ghi mang hai nhãn")
    display(files[files.record_uid.isin(bad)][["record_uid", "cls", "rel"]])

n_fb = int((files.rid_source == "fallback:stem").sum())
if n_fb:
    print(f"!! {n_fb} tệp không khớp regex '{cfg.record_regex}' -> lấy tên tệp làm mã bản ghi "
          f"(mỗi tệp thành một bản ghi; kiểm tra lại nếu một lần thu bị cắt thành nhiều tệp).")

# [V2] Mã bản ghi xuất hiện ở CẢ HAI LỚP trong cùng miền: với cfg.record_id_by_class=True đây là các lần thu
#      khác nhau (chuỗi S đánh số lại trong mỗi lớp); với False thì là mâu thuẫn nhãn (đã bắt ở trên).
_both = files.groupby(["domain", "record_id"]).cls.nunique()
_both = _both[_both > 1]
if len(_both):
    print(f"Mã bản ghi có ở cả leak và noleak: {len(_both)} (ví dụ {[f'{d}:{r}' for d, r in _both.index[:5]]})"
          + (" -> tách thành hai bản ghi theo lớp (cfg.record_id_by_class=True)." if cfg.record_id_by_class else ""))
_tpr = files.groupby("domain").agg(n_rec=("record_uid", "nunique"), n_files=("path", "size"))
print("Bản ghi / tệp theo miền:", {d: f"{int(r.n_rec)} / {int(r.n_files)}" for d, r in _tpr.iterrows()})

# Mã Rxxx trùng giữa các miền: hợp lệ (đã tách bằng record_uid) nhưng phải nêu ra.
_shared = files.groupby("record_id").domain.nunique()
_shared = _shared[_shared > 1]
if len(_shared):
    print(f"Mã bản ghi xuất hiện ở NHIỀU miền: {len(_shared)} (ví dụ {list(_shared.index[:6])}).")
    print("  -> được coi là các lần thu KHÁC NHAU (record_uid = miền|Rxxx). Mục 3.6 kiểm tra bằng nội")
    print("     dung âm thanh xem có phải cùng một lần thu bị sao chép sang miền khác hay không.")

fpr_ = files.groupby("record_uid").size()
print(f"Tệp/bản ghi: trung vị {int(fpr_.median())}, min {fpr_.min()}, max {fpr_.max()} | "
      f"bản ghi nhiều tệp: {(fpr_ > 1).sum()}/{len(fpr_)}")
if problems:
    msg = "CẦN XỬ LÝ: " + "; ".join(problems)
    print("\n=== " + msg + " ===")
    if cfg.strict_checks:
        raise ValueError(msg)
else:
    print("✓ Nhãn nhất quán trong mọi bản ghi.")
''')

code(r'''
# =========================================================
# 2.4 Kiểm kê MỨC BẢN GHI theo miền × lớp + Hình 1 / Record-level inventory by domain
# =========================================================
records = (files.groupby("record_uid")
           .agg(domain=("domain", "first"), record_id=("record_id", "first"),
                label=("label", "first"), cls=("cls", "first"),
                n_files=("file_stem", "size"), duration_s=("duration_s", "sum"),
                sr_native=("sr_native", "min"))
           .reset_index())
records.to_csv(OUT / "manifest_records.csv", index=False)
DOMAINS_ALL = sorted(records.domain.unique())
DOMAIN_COLOR = {d: DOMAIN_PALETTE[i % len(DOMAIN_PALETTE)] for i, d in enumerate(DOMAINS_ALL)}

inv = (records.groupby(["domain", "cls"])
       .agg(n_records=("record_uid", "nunique"), n_files=("n_files", "sum"),
            total_min=("duration_s", lambda s: s.sum() / 60), mean_s=("duration_s", "mean"),
            min_s=("duration_s", "min"), max_s=("duration_s", "max"))
       .reset_index())
save_table(inv.round(2), "inventory_domain_class",
           "Recording-level inventory per acquisition domain and class (a recording may span several files).", 2)

prior = records.groupby("domain").label.mean().rename("leak_fraction")
print("Tỉ lệ bản ghi leak theo miền:", prior.round(3).to_dict())
if prior.max() - prior.min() > 0.15:
    print("!! DỊCH CHUYỂN NHÃN: tỉ lệ lớp khác nhau > 15 điểm % giữa các miền. Căn chỉnh biên P(f)")
    print("   (DANN) có thể bị ảnh hưởng (Zhao et al., 2019) -> nguồn được lấy mẫu cân bằng lớp,")
    print("   và CDAN+E (căn chỉnh có điều kiện) là đối chứng trực tiếp cho vấn đề này.")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
piv = inv.pivot(index="domain", columns="cls", values="n_records").reindex(columns=["noleak", "leak"]).fillna(0)
xx = np.arange(len(piv)); w = .38
for j, c in enumerate(["noleak", "leak"]):
    ax[0].bar(xx + (j - .5) * w, piv[c].values, w, color=PAL[c], label=c)
    for x0, v in zip(xx + (j - .5) * w, piv[c].values):
        ax[0].text(x0, v, f"{int(v)}", ha="center", va="bottom", fontsize=8)
ax[0].set_xticks(xx); ax[0].set_xticklabels(piv.index, rotation=20)
ax[0].set_title("(a) Independent recordings"); ax[0].legend()
for d in DOMAINS_ALL:
    ax[1].hist(records.loc[records.domain == d, "duration_s"], bins=15, alpha=.55,
               color=DOMAIN_COLOR[d], label=d)
ax[1].set_xlabel("recording duration (s)"); ax[1].set_title("(b) Duration by domain"); ax[1].legend()
ax[2].bar(prior.index, prior.values, color=[DOMAIN_COLOR[d] for d in prior.index], width=.5)
ax[2].axhline(.5, color="k", lw=.8, ls="--")
ax[2].set_ylim(0, 1); ax[2].set_title("(c) Leak fraction (label shift)")
for i, v in enumerate(prior.values):
    ax[2].text(i, v, f"{v:.2f}", ha="center", va="bottom")
fig.suptitle("Mixed dataset: acquisition domains and class balance", y=1.04)
savefig(fig, "domain_inventory",
        "Dataset composition per acquisition domain: (a) independent recordings per class, "
        "(b) recording duration, (c) leak fraction, which quantifies label shift between domains.")
''')

# =============================================================================
# 3. EDA DỊCH CHUYỂN MIỀN
# =============================================================================
md(r'''
## 3. Phân tích dịch chuyển miền TRƯỚC khi huấn luyện / Domain-shift analysis before training

Mục này trả lời **RQ1** mà không dùng mạng sâu: (i) băng thông hiệu dụng mỗi miền, (ii) PSD theo miền × lớp,
(iii) dấu hiệu rò rỉ có **cùng chiều** giữa các miền không (điều kiện cần để UDA có ý nghĩa), (iv) trùng lặp
nội dung giữa các miền, (v) mức tách được của miền và **suy giảm chéo miền** của một bộ phân lớp cổ điển.

Toàn bộ tệp được đọc **một lần**: cùng lượt đó sinh frame (mục 4) và thống kê mức tệp (PSD Welch, băng thông,
đường bao năng lượng), lưu cache theo dấu vân tay nội dung.
''')

code(r'''
# =========================================================
# 3.1 Nạp + tiền xử lý + MỘT lượt quét: frame & thống kê tệp / single pass: frames + file statistics
# =========================================================
_HP, _NOTCH = {}, {}
def _hp_coeffs(sr, fc):
    k = (int(sr), float(fc))
    if k not in _HP:
        _HP[k] = sps.butter(4, float(fc) / (sr / 2), btype="high")
    return _HP[k]

def _notches(sr):
    k = int(sr)
    if k not in _NOTCH:
        _NOTCH[k] = [sps.iirnotch(cfg.mains_hz * h / (sr / 2), cfg.notch_q)
                     for h in range(1, cfg.mains_harmonics + 1) if cfg.mains_hz * h < sr / 2 - 10]
    return _NOTCH[k]

def load_audio(path, sr=None):
    sr = sr or cfg.target_sr
    try:
        x, _ = librosa.load(str(path), sr=sr, mono=True)
    except Exception:
        y, sr0 = sf.read(str(path), always_2d=True, dtype="float32")
        y = np.nan_to_num(y).mean(axis=1)
        x = librosa.resample(y, orig_sr=sr0, target_sr=sr) if sr0 != sr else y
    x = np.nan_to_num(np.asarray(x, dtype=np.float32))
    x = x - x.mean()
    if cfg.highpass_hz and cfg.highpass_hz > 0:
        b, a = _hp_coeffs(sr, cfg.highpass_hz); x = sps.filtfilt(b, a, x).astype(np.float32)
    if cfg.notch_mains:
        for b, a in _notches(sr):
            x = sps.filtfilt(b, a, x).astype(np.float32)
    x = np.nan_to_num(x)
    m = float(np.max(np.abs(x)))
    return (x / m if m > 1e-9 else x).astype(np.float32)

WELCH_NPERSEG = 4096
def file_stats(x, sr):
    nper = min(WELCH_NPERSEG, len(x))
    f, P = sps.welch(x, fs=sr, nperseg=nper, noverlap=nper // 2)
    if len(f) != WELCH_NPERSEG // 2 + 1:                     # tệp quá ngắn -> nội suy về lưới chuẩn
        fg = np.fft.rfftfreq(WELCH_NPERSEG, 1 / sr); P = np.interp(fg, f, P); f = fg
    tot = P.sum() + 1e-30
    cum = np.cumsum(P) / tot
    bw = float(f[min(len(f) - 1, np.searchsorted(cum, cfg.bw_energy_frac))])
    hop = int(0.1 * sr)
    env = np.log10(np.array([np.mean(x[i:i + hop] ** 2) + 1e-12 for i in range(0, len(x) - hop + 1, hop)]))
    return dict(psd=P.astype(np.float32), bw=bw, hf_ratio=float(P[f >= 4000].sum() / tot),
                env=env.astype(np.float32), rms=float(np.sqrt(np.mean(x ** 2))))

def _fingerprint(df):
    h = hashlib.sha1()
    for _, r in df.sort_values("path").iterrows():
        try:
            st = os.stat(r.path); meta = f"{st.st_size}|{int(st.st_mtime)}"
        except Exception:
            meta = "NA"
        h.update(f"{r.path}|{meta}|{r.record_uid}|{r.label}|".encode())
    h.update(json.dumps(dict(sr=cfg.target_sr, hp=cfg.highpass_hz, n=cfg.notch_mains, mh=cfg.mains_hz,
                             k=cfg.mains_harmonics, q=cfg.notch_q, fs=cfg.frame_sec, hs=cfg.hop_sec,
                             bwf=cfg.bw_energy_frac), sort_keys=True).encode())
    return h.hexdigest()[:12]

DATA_FP = _fingerprint(files)
_xp, _mp, _sp = (CACHE_DIR / f"frames_{DATA_FP}.npy", CACHE_DIR / f"frames_{DATA_FP}.csv",
                 CACHE_DIR / f"filestats_{DATA_FP}.pkl")
if _xp.exists() and _mp.exists() and _sp.exists():
    X_ALL = np.load(_xp, mmap_mode="r"); frames = pd.read_csv(_mp)
    FILE_STATS = pickle.load(open(_sp, "rb"))
    print("Nạp cache:", _xp.name)
else:
    t0 = time.time(); chunks, metas, FILE_STATS = [], [], {}
    for _, r in files.iterrows():
        x = load_audio(r.path)
        FILE_STATS[int(r.file_id)] = file_stats(x, cfg.target_sr)
        if len(x) < FRAME_LEN:
            x = np.pad(x, (0, FRAME_LEN - len(x)))
        for s in range(0, len(x) - FRAME_LEN + 1, HOP_LEN):
            seg = x[s:s + FRAME_LEN]
            chunks.append(seg)
            metas.append(dict(file_id=int(r.file_id), record_uid=r.record_uid, domain=r.domain,
                              label=int(r.label), cls=r.cls, start_s=s / cfg.target_sr,
                              rms=float(np.sqrt(np.mean(seg ** 2)))))
    X_ALL = np.stack(chunks).astype(np.float32); del chunks
    frames = pd.DataFrame(metas)
    np.save(_xp, X_ALL); frames.to_csv(_mp, index=False)
    pickle.dump(FILE_STATS, open(_sp, "wb"))
    X_ALL = np.load(_xp, mmap_mode="r")
    print(f"Quét {len(files)} tệp trong {time.time() - t0:.1f}s")

assert np.isfinite(np.asarray(X_ALL[:min(len(X_ALL), 2048)])).all(), "NaN/Inf trong frame"
y_all = frames.label.values.astype(np.float32)
uid_all = frames.record_uid.values
dom_all = frames.domain.values
N_FRAMES = len(frames)
PSD_F = np.fft.rfftfreq(WELCH_NPERSEG, 1 / cfg.target_sr)
print(f"Frame: {N_FRAMES} | shape {X_ALL.shape} | {X_ALL.nbytes / 1024 ** 3:.2f} GB")
display(frames.groupby(["domain", "cls"]).agg(n_frames=("label", "size"),
                                              n_records=("record_uid", "nunique"),
                                              mean_rms=("rms", "mean")).round(4))
''')

code(r'''
# =========================================================
# 3.2 Băng thông hiệu dụng theo miền + HOÀ HỢP BĂNG THÔNG / Effective bandwidth and harmonisation
# =========================================================
files["bw_hz"] = files.file_id.map(lambda i: FILE_STATS[i]["bw"])
files["hf_ratio"] = files.file_id.map(lambda i: FILE_STATS[i]["hf_ratio"])
rec_bw = files.groupby("record_uid").agg(domain=("domain", "first"), cls=("cls", "first"),
                                         bw_hz=("bw_hz", "median"), hf_ratio=("hf_ratio", "median"))
bw_tab = (rec_bw.groupby("domain")
          .agg(n_records=("bw_hz", "size"), bw_median_hz=("bw_hz", "median"),
               bw_p10_hz=("bw_hz", lambda s: s.quantile(.1)), bw_p90_hz=("bw_hz", lambda s: s.quantile(.9)),
               energy_above_4k=("hf_ratio", "median"))
          .reset_index())
save_table(bw_tab, "bandwidth_by_domain",
           f"Effective bandwidth per domain (frequency below which {cfg.bw_energy_frac*100:.1f}% of the "
           "signal energy lies; median over recordings) and median energy fraction above 4 kHz.", 3)

NYQ = cfg.target_sr / 2
# ---- [V2] Hoà hợp HAI PHÍA: PSD trung bình mỗi miền (mọi bản ghi, cả hai lớp -> KHÔNG dùng nhãn), căn mức
#      ở 300-1500 Hz; dải thấp (< f_min_cap) và dải cao (> 2 kHz) nơi các miền lệch nhau > harmonise_tol_db
#      chỉ phản ánh chuỗi thu (cảm biến, bộ lọc của máy ghi, bộ lọc tái lấy mẫu) -> bị loại khỏi front-end.
_gc = np.geomspace(20.0, NYQ * 0.99, 96)
def _dom_curve(d):
    L = []
    for u, g in files[files.domain == d].groupby("record_uid"):
        P = np.mean([FILE_STATS[i]["psd"] for i in g.file_id], axis=0)
        L.append(10 * np.log10(np.interp(_gc, PSD_F, P) + 1e-30))
    c = np.mean(L, axis=0)
    return c - np.median(c[(_gc >= 300) & (_gc <= 1500)])
CURVES = {d: _dom_curve(d) for d in DOMAINS_ALL}
MISMATCH = np.zeros_like(_gc)
for _i, _a in enumerate(DOMAINS_ALL):
    for _b in DOMAINS_ALL[_i + 1:]:
        MISMATCH = np.maximum(MISMATCH, np.abs(CURVES[_a] - CURVES[_b]))
MISMATCH = pd.Series(MISMATCH).rolling(3, center=True, min_periods=1).median().values

F_MIN_EFF, F_MAX_EFF = float(max(cfg.f_min, 20.0)), float(min(cfg.f_max, NYQ))
if cfg.bandwidth_harmonize == "auto" and len(DOMAINS_ALL) >= 2:
    lo = np.where((_gc <= cfg.f_min_cap) & (MISMATCH > cfg.harmonise_tol_db))[0]
    if len(lo):
        F_MIN_EFF = float(max(F_MIN_EFF, min(cfg.f_min_cap, 5 * math.ceil(_gc[min(lo.max() + 1, len(_gc) - 1)] / 5))))
    hi = np.where((_gc >= 2000) & (MISMATCH > cfg.harmonise_tol_db))[0]
    if len(hi):
        F_MAX_EFF = float(min(F_MAX_EFF, 100 * math.floor(_gc[max(hi.min() - 1, 0)] / 100)))
    common = float(bw_tab.bw_p90_hz.min())            # quy tắc v1: chuỗi thu 8 kHz (năng lượng 99.5%)
    if common < 0.85 * NYQ and bw_tab.bw_p90_hz.max() > 1.15 * common:
        F_MAX_EFF = float(min(F_MAX_EFF, max(1000.0, math.floor(common / 100) * 100)))
elif isinstance(cfg.bandwidth_harmonize, (tuple, list)):
    F_MIN_EFF, F_MAX_EFF = map(float, cfg.bandwidth_harmonize)
HARMONISED = bool(F_MIN_EFF > max(cfg.f_min, 20.0) + 1e-6 or F_MAX_EFF < min(cfg.f_max, NYQ) - 1e-6)
if HARMONISED:
    print(f"[P] HOÀ HỢP CHUỖI THU: mọi front-end phổ dùng dải chung {F_MIN_EFF:.0f}-{F_MAX_EFF:.0f} Hz "
          f"(ngưỡng lệch {cfg.harmonise_tol_db:g} dB giữa PSD đã căn mức của các miền).")
    print("    Lý do: ngoài dải này phổ hai miền khác nhau do CHUỖI THU, không do rò rỉ -> dấu hiệu miền tầm")
    print("    thường và lối tắt tiềm năng. Viết rõ trong phần Methods.")
    if F_MIN_EFF >= cfg.f_min_cap - 1e-6:
        print(f"!! f_min chạm trần {cfg.f_min_cap:g} Hz: lệch phổ còn lớn ở dải thấp — xem Hình (d).")
else:
    print(f"Phổ các miền tương đương -> giữ {F_MIN_EFF:.0f}-{F_MAX_EFF:.0f} Hz.")

fig, ax = plt.subplots(2, 2, figsize=(13, 7))
ax = ax.ravel()
for i, d in enumerate(DOMAINS_ALL):
    v = rec_bw.loc[rec_bw.domain == d, "bw_hz"].values
    ax[0].scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-.15, .15, len(v)), v,
                  s=14, color=DOMAIN_COLOR[d], alpha=.7)
    ax[0].hlines(np.median(v), i - .3, i + .3, color="k", lw=2)
ax[0].axhline(F_MAX_EFF, color=PAL["warn"], ls="--", lw=1.2, label=f"front-end f_max = {F_MAX_EFF:.0f} Hz")
ax[0].set_xticks(range(len(DOMAINS_ALL))); ax[0].set_xticklabels(DOMAINS_ALL)
ax[0].set_ylabel("effective bandwidth (Hz)"); ax[0].set_title("(a) Effective bandwidth per recording")
ax[0].legend(loc="lower right")
for d in DOMAINS_ALL:
    v = rec_bw.loc[rec_bw.domain == d, "hf_ratio"].values
    ax[1].hist(np.log10(v + 1e-9), bins=20, alpha=.55, color=DOMAIN_COLOR[d], label=d)
ax[1].set_xlabel("log10(energy fraction above 4 kHz)"); ax[1].set_title("(b) High-frequency energy")
ax[1].legend()
for d in DOMAINS_ALL:
    ax[2].semilogx(_gc, CURVES[d], color=DOMAIN_COLOR[d], lw=1.5, label=d)
ax[3].semilogx(_gc, MISMATCH, color="k", lw=1.4)
ax[3].axhline(cfg.harmonise_tol_db, color=PAL["warn"], ls="--", label=f"tolerance {cfg.harmonise_tol_db:g} dB")
for a_ in (ax[2], ax[3]):
    a_.axvspan(20, F_MIN_EFF, color="#BBBBBB", alpha=.35)
    a_.axvspan(F_MAX_EFF, NYQ, color="#BBBBBB", alpha=.35)
    a_.set_xlabel("frequency (Hz)")
ax[2].set_ylabel("PSD aligned at 0.3–1.5 kHz (dB)"); ax[2].set_title("(c) Domain-average spectra (labels not used)")
ax[2].legend()
ax[3].set_ylabel("max |difference| between domains (dB)")
ax[3].set_title(f"(d) Acquisition-chain mismatch → common band {F_MIN_EFF:.0f}–{F_MAX_EFF:.0f} Hz"); ax[3].legend()
savefig(fig, "bandwidth_by_domain",
        "Acquisition-chain fingerprint and harmonisation: (a) effective bandwidth of every recording, (b) energy above "
        "4 kHz, (c) domain-average spectra aligned at 0.3–1.5 kHz (class labels not used) and (d) their maximum "
        "difference; grey bands are excluded from every spectral front-end.")
''')

code(r'''
# =========================================================
# 3.3 PSD theo miền × lớp (mức bản ghi) / Record-level PSD per domain and class
# =========================================================
def record_psd_table():
    rows = {}
    for uid, g in files.groupby("record_uid"):
        P = np.mean([FILE_STATS[i]["psd"] for i in g.file_id], axis=0)
        rows[uid] = P / (P[(PSD_F >= F_MIN_EFF) & (PSD_F <= F_MAX_EFF)].sum() + 1e-30)   # chuẩn hoá: bất biến gain
    return rows
REC_PSD = record_psd_table()
_band = (PSD_F >= 20) & (PSD_F <= NYQ)                 # vẽ toàn dải, tô xám phần bị loại

fig, axes = plt.subplots(1, len(DOMAINS_ALL) + 1, figsize=(4.2 * (len(DOMAINS_ALL) + 1), 3.6), sharey=True)
for ax, d in zip(axes, DOMAINS_ALL):
    for c in ["noleak", "leak"]:
        uids = records.loc[(records.domain == d) & (records.cls == c), "record_uid"]
        if not len(uids):
            continue
        L = 10 * np.log10(np.stack([REC_PSD[u] for u in uids])[:, _band] + 1e-20)
        mu = L.mean(0)
        se = L.std(0, ddof=1) / np.sqrt(len(L)) if len(L) > 1 else np.zeros(L.shape[1])
        ax.semilogx(PSD_F[_band], mu, color=PAL[c], lw=1.3, label=f"{c} (n={len(L)})")
        ax.fill_between(PSD_F[_band], mu - 1.96 * se, mu + 1.96 * se, color=PAL[c], alpha=.2)
    ax.set_title(f"domain: {d}"); ax.set_xlabel("frequency (Hz)"); ax.legend()
axes[0].set_ylabel("relative PSD (dB)")
ax = axes[-1]
for d in DOMAINS_ALL:
    for c, ls in [("noleak", "--"), ("leak", "-")]:
        uids = records.loc[(records.domain == d) & (records.cls == c), "record_uid"]
        if len(uids):
            L = 10 * np.log10(np.stack([REC_PSD[u] for u in uids])[:, _band] + 1e-20)
            ax.semilogx(PSD_F[_band], L.mean(0), color=DOMAIN_COLOR[d], ls=ls, lw=1.2, label=f"{d}/{c}")
ax.set_title("all domains overlaid"); ax.set_xlabel("frequency (Hz)"); ax.legend(fontsize=7)
for a_ in axes:
    a_.axvspan(20, F_MIN_EFF, color="#BBBBBB", alpha=.3); a_.axvspan(F_MAX_EFF, NYQ, color="#BBBBBB", alpha=.3)
savefig(fig, "psd_domain_class",
        "Record-level relative power spectral density (Welch) per domain and class (mean ± 95% CI over "
        "recordings). The right panel overlays domains: vertical offsets between domains of the same class "
        "visualise covariate shift.")
''')

code(r'''
# =========================================================
# 3.4-3.5 Năng lượng theo dải + AUC TRONG TỪNG MIỀN + kiểm tra CHIỀU dấu hiệu
#         Band energies, within-domain band AUC and cue-direction consistency (conditional-shift check)
# =========================================================
_edges = np.unique(np.round(np.geomspace(20, NYQ, 13)).astype(int))     # toàn dải: thấy cả dải bị loại
BANDS = [(int(a), int(b)) for a, b in zip(_edges[:-1], _edges[1:])]

def band_energy(P):
    return np.array([P[(PSD_F >= a) & (PSD_F < b)].sum() for a, b in BANDS])

REC_BAND = pd.DataFrame({u: 10 * np.log10(band_energy(P) + 1e-20) for u, P in REC_PSD.items()}).T
REC_BAND.columns = [f"{a}-{b}" for a, b in BANDS]
REC_BAND = REC_BAND.join(records.set_index("record_uid")[["domain", "label"]])

def _auc_ci(y, s, n=500, seed=0):
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    a = roc_auc_score(y, s); rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            v.append(roc_auc_score(y[i], s[i]))
    return a, np.percentile(v, 2.5), np.percentile(v, 97.5)

band_rows = []
for d in DOMAINS_ALL:
    g = REC_BAND[REC_BAND.domain == d]
    for col in REC_BAND.columns[:-2]:
        a, lo, hi = _auc_ci(g.label.values, g[col].values)
        band_rows.append(dict(domain=d, band_hz=col, auc=a, lo=lo, hi=hi,
                              sig=bool(np.isfinite(lo) and (lo > .5 or hi < .5))))
band_df = pd.DataFrame(band_rows)
A = band_df.pivot(index="domain", columns="band_hz", values="auc")[REC_BAND.columns[:-2]]
S = band_df.pivot(index="domain", columns="band_hz", values="sig")[REC_BAND.columns[:-2]]

# Chiều dấu hiệu: một dải "nhất quán" nếu mọi miền cùng phía 0.5; "ĐỔI CHIỀU" nếu có hai miền
# có ý nghĩa (CI không chứa 0.5) ở hai phía khác nhau -> dịch chuyển có điều kiện thật sự.
cons = []
for col in A.columns:
    v = A[col].values - .5; sg = S[col].values
    flip = bool(((v > 0) & sg).any() and ((v < 0) & sg).any())
    same = bool(np.all(v > 0) or np.all(v < 0))
    _lo, _hi = map(float, col.split("-"))
    cons.append(dict(band_hz=col, in_frontend=bool(_lo >= F_MIN_EFF - 1 and _hi <= F_MAX_EFF + 1),
                     **{f"AUC_{d}": A.loc[d, col] for d in A.index},
                     direction=("FLIP (significant)" if flip else ("consistent" if same else "mixed"))))
cons_df = pd.DataFrame(cons)
save_table(cons_df, "band_auc_within_domain",
           "Univariate record-level AUC of each band's relative energy for leak vs no-leak, computed "
           "separately inside each domain. 'FLIP' marks bands whose cue reverses direction between domains "
           "with bootstrap CIs excluding 0.5 (evidence of conditional shift).", 3)
n_flip = int((cons_df.direction == "FLIP (significant)").sum())
n_cons = int((cons_df.direction == "consistent").sum())
print(f"Dải nhất quán chiều: {n_cons}/{len(cons_df)} | dải ĐỔI CHIỀU có ý nghĩa: {n_flip}")
if n_flip:
    print("!! Có dấu hiệu DỊCH CHUYỂN CÓ ĐIỀU KIỆN: cùng một dải báo hiệu rò rỉ theo hai chiều ngược nhau ở")
    print("   hai miền. Giả thiết P(y|x) chung của UDA bị vi phạm một phần -> phải thảo luận trong bài báo.")

fig, ax = plt.subplots(1, 2, figsize=(16, 1.2 + 0.6 * len(DOMAINS_ALL) + 1.6),
                       gridspec_kw=dict(width_ratios=[3, 2]))
im = ax[0].imshow(A.values - .5, cmap="RdBu_r", vmin=-.5, vmax=.5, aspect="auto")
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        ax[0].text(j, i, f"{A.values[i, j]:.2f}" + ("*" if S.values[i, j] else ""),
                   ha="center", va="center", fontsize=7)
ax[0].set_xticks(range(A.shape[1])); ax[0].set_xticklabels(A.columns, rotation=45, ha="right", fontsize=8)
ax[0].set_yticks(range(A.shape[0])); ax[0].set_yticklabels(A.index); ax[0].grid(False)
ax[0].set_title("(a) Within-domain band AUC (red: leak louder; * CI excludes 0.5)")
plt.colorbar(im, ax=ax[0], fraction=.025, pad=.01).set_label("AUC − 0.5", labelpad=2)
prof = REC_BAND.groupby(["domain", "label"])[list(A.columns)].mean()
prof = prof.sub(prof.mean(axis=1), axis=0)
for (d, lab), row in prof.iterrows():
    ax[1].plot(range(len(row)), row.values, color=DOMAIN_COLOR[d], ls="-" if lab == 1 else "--",
               marker="o", ms=3, label=f"{d}/{'leak' if lab == 1 else 'noleak'}")
ax[1].set_xticks(range(len(A.columns))); ax[1].set_xticklabels(A.columns, rotation=45, ha="right", fontsize=8)
ax[1].set_ylabel("band energy − mean (dB)"); ax[1].set_title("(b) Spectral-tilt profile"); ax[1].legend(fontsize=7)
savefig(fig, "band_auc_by_domain",
        "Is the leak cue shared across domains? (a) Record-level AUC of each band's relative energy inside each "
        "domain; (b) mean band-energy profiles (mean-removed) per domain and class.")
''')

code(r'''
# =========================================================
# 3.6 Phát hiện bản ghi TRÙNG NỘI DUNG / Content-duplicate detection (leakage guard)
#     So khớp đường bao log-năng lượng 100 ms (bền với khác biệt tần số lấy mẫu/lọc) giữa các
#     tệp thuộc HAI BẢN GHI KHÁC NHAU có thời lượng gần bằng nhau.
# =========================================================
DUP_PAIRS = pd.DataFrame()
if cfg.check_duplicates:
    def _z(v):
        return (v - v.mean()) / (v.std() + 1e-9)
    fl = files[["file_id", "record_uid", "domain", "duration_s", "rel"]].sort_values("duration_s").values
    dups = []
    for i in range(len(fl)):
        j = i + 1
        while j < len(fl) and fl[j][3] - fl[i][3] <= 1.0:
            if fl[i][1] != fl[j][1]:
                a, b = FILE_STATS[int(fl[i][0])]["env"], FILE_STATS[int(fl[j][0])]["env"]
                L = min(len(a), len(b))
                if L >= 20:
                    za, zb = _z(a[:L]), _z(b[:L])
                    r = max(float(np.dot(za[k:], zb[:L - k]) / (L - k)) for k in range(0, min(10, L - 10)))
                    r = max(r, max(float(np.dot(zb[k:], za[:L - k]) / (L - k)) for k in range(1, min(10, L - 10))))
                    if r >= 0.97:
                        dups.append(dict(file_a=fl[i][4], uid_a=fl[i][1], file_b=fl[j][4], uid_b=fl[j][1],
                                         cross_domain=fl[i][2] != fl[j][2], envelope_r=round(r, 4)))
            j += 1
    DUP_PAIRS = pd.DataFrame(dups)
    if len(DUP_PAIRS):
        save_table(DUP_PAIRS, "duplicate_pairs",
                   "Pairs of files from different recordings whose 100-ms energy envelopes correlate at r ≥ 0.97 "
                   "(likely the same acquisition stored twice).", 4)
        n_x = int(DUP_PAIRS.cross_domain.sum())
        print(f"!! {len(DUP_PAIRS)} cặp nghi trùng ({n_x} cặp GIỮA hai miền).")
        print("   Trùng giữa nguồn và đích = mô hình đã 'thấy' bản ghi đích có nhãn -> kết quả UDA bị thổi")
        print("   phồng. Hãy loại một bản trong mỗi cặp (cfg.exclude_domains hoặc xoá tệp) rồi chạy lại.")
    else:
        print("✓ Không phát hiện bản ghi trùng nội dung (r < 0.97 với mọi cặp ứng viên).")
''')

code(r'''
# =========================================================
# 3.7 Mức tách được của MIỀN và suy giảm CHÉO MIỀN với đặc trưng cổ điển
#     Domain separability (input-level PAD) and cross-domain degradation of a classical baseline
#     Đặc trưng: thống kê log-Mel mỗi frame (trung bình theo thời gian đã trừ mức trung bình + độ
#     lệch chuẩn theo thời gian) -> bất biến với gain, cùng dải f_max đã hoà hợp.
# =========================================================
_mel_cpu = torchaudio.transforms.MelSpectrogram(sample_rate=cfg.target_sr, n_fft=cfg.n_fft,
                                                hop_length=cfg.hop_length, n_mels=cfg.n_mels,
                                                f_min=F_MIN_EFF, f_max=F_MAX_EFF, power=2.0)
_mel_dev = copy.deepcopy(_mel_cpu).to(DEVICE)          # .to() của nn.Module là tại chỗ -> phải sao chép

def logmel_stats(idx, bs=512):
    out = []
    for s in range(0, len(idx), bs):
        xb = torch.from_numpy(np.asarray(X_ALL[np.asarray(idx[s:s + bs])])).to(DEVICE)
        with torch.no_grad():
            L = torch.log10(_mel_dev(xb) + 1e-10)
            mu = L.mean(-1); sd = L.std(-1)
            out.append(torch.cat([mu - mu.mean(1, keepdim=True), sd], 1).cpu().numpy())
    return np.concatenate(out)

FEAT_IN = logmel_stats(np.arange(N_FRAMES))
REC_FEAT = pd.DataFrame(FEAT_IN).groupby(uid_all).mean()
REC_FEAT = REC_FEAT.join(records.set_index("record_uid")[["domain", "label"]])
_fcols = [c for c in REC_FEAT.columns if c not in ("domain", "label")]

def lr_pipe():
    return LogisticRegression(C=0.1, max_iter=2000)

def cv_proba(Xm, y, k=5, seed=0, strat=None):
    # Xác suất out-of-fold mức bản ghi (phân tầng theo y hoặc theo strat)
    strat = y if strat is None else strat
    k = int(min(k, pd.Series(strat).value_counts().min()))
    p = np.full(len(y), np.nan)
    if k < 2:
        return p
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(Xm, strat):
        sc = StandardScaler().fit(Xm[tr])
        p[te] = lr_pipe().fit(sc.transform(Xm[tr]), y[tr]).predict_proba(sc.transform(Xm[te]))[:, 1]
    return p

def pad_from_err(err):
    return 2 * (1 - 2 * min(err, 0.5))

# (a) PAD đầu vào cho từng cặp miền (mức bản ghi)
pad_rows = []
for i, a in enumerate(DOMAINS_ALL):
    for b in DOMAINS_ALL[i + 1:]:
        g = REC_FEAT[REC_FEAT.domain.isin([a, b])]
        yd = (g.domain == b).astype(int).values
        p = cv_proba(g[_fcols].values, yd)
        err = float(np.mean((p >= .5) != yd))
        pad_rows.append(dict(domain_a=a, domain_b=b, n_a=int((yd == 0).sum()), n_b=int((yd == 1).sum()),
                             domain_clf_auc=roc_auc_score(yd, p), domain_clf_error=err, PAD=pad_from_err(err)))
PAD_INPUT = pd.DataFrame(pad_rows)
save_table(PAD_INPUT, "pad_input",
           "Input-level domain separability: a logistic-regression domain classifier on gain-invariant log-Mel "
           "statistics (record-level, stratified CV). PAD = 2(1 − 2ε); PAD → 2 means fully separable domains.", 3)

# (b) Ma trận huấn luyện miền A -> kiểm thử miền B (đường chéo = CV trong miền)
xd_rows = []
for a in DOMAINS_ALL:
    ga = REC_FEAT[REC_FEAT.domain == a]
    for b in DOMAINS_ALL:
        gb = REC_FEAT[REC_FEAT.domain == b]
        if ga.label.nunique() < 2 or gb.label.nunique() < 2:
            continue
        if a == b:
            p = cv_proba(ga[_fcols].values, ga.label.values)
            auc = roc_auc_score(ga.label.values, p) if np.isfinite(p).all() else np.nan
        else:
            sc = StandardScaler().fit(ga[_fcols].values)
            m = lr_pipe().fit(sc.transform(ga[_fcols].values), ga.label.values)
            auc = roc_auc_score(gb.label.values, m.predict_proba(sc.transform(gb[_fcols].values))[:, 1])
        xd_rows.append(dict(train=a, test=b, auc=auc))
XD = pd.DataFrame(xd_rows).pivot(index="train", columns="test", values="auc")
save_table(XD.reset_index().rename(columns={"train": "train \\ test"}), "cross_domain_matrix_classical",
           "Record-level AUC of a classical log-Mel-statistics + logistic-regression detector trained on one domain "
           "and tested on another (diagonal: within-domain stratified CV). Off-diagonal drops quantify the "
           "domain gap that motivates adaptation.", 3)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw=dict(width_ratios=[1, 1.3, 1.3]))
im = ax[0].imshow(XD.values, cmap="viridis", vmin=.5, vmax=1.0)
for i in range(XD.shape[0]):
    for j in range(XD.shape[1]):
        ax[0].text(j, i, f"{XD.values[i, j]:.2f}", ha="center", va="center",
                   color="w" if XD.values[i, j] < .8 else "k")
ax[0].set_xticks(range(XD.shape[1])); ax[0].set_xticklabels(XD.columns)
ax[0].set_yticks(range(XD.shape[0])); ax[0].set_yticklabels(XD.index)
ax[0].set_xlabel("test domain"); ax[0].set_ylabel("train domain"); ax[0].grid(False)
ax[0].set_title("(a) Classical detector AUC")
plt.colorbar(im, ax=ax[0], fraction=.046)
Z = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(REC_FEAT[_fcols].values))
for d in DOMAINS_ALL:
    for lab, mk in [(0, "o"), (1, "^")]:
        m = (REC_FEAT.domain.values == d) & (REC_FEAT.label.values == lab)
        ax[1].scatter(Z[m, 0], Z[m, 1], s=22, marker=mk, color=DOMAIN_COLOR[d], alpha=.75,
                      label=f"{d}/{'leak' if lab else 'noleak'}")
ax[1].set_title("(b) PCA of record features (colour = domain)"); ax[1].legend(fontsize=7)
_sub = np.random.default_rng(0).choice(N_FRAMES, min(N_FRAMES, cfg.tsne_max_points), replace=False)
Zt = TSNE(2, init="pca", random_state=0, perplexity=min(30, max(5, len(_sub) // 4))).fit_transform(
    StandardScaler().fit_transform(FEAT_IN[_sub]))
for d in DOMAINS_ALL:
    for lab, mk in [(0, "o"), (1, "^")]:
        m = (dom_all[_sub] == d) & (y_all[_sub] == lab)
        ax[2].scatter(Zt[m, 0], Zt[m, 1], s=5, marker=mk, color=DOMAIN_COLOR[d], alpha=.5)
ax[2].set_title("(c) t-SNE of frame features (o noleak, ^ leak)"); ax[2].set_xticks([]); ax[2].set_yticks([])
savefig(fig, "input_domain_shift",
        "Input-level domain shift: (a) train-on-A/test-on-B AUC of a classical detector, (b) PCA of record-level "
        "log-Mel statistics and (c) t-SNE of frame-level statistics, coloured by domain.")
''')

code(r'''
# =========================================================
# 3.8 Spectrogram đại diện mỗi miền × lớp / Representative log-Mel spectrograms
# =========================================================
_rng = np.random.default_rng(1)
fig, axes = plt.subplots(2, len(DOMAINS_ALL), figsize=(4.4 * len(DOMAINS_ALL), 5.6), squeeze=False)
for j, d in enumerate(DOMAINS_ALL):
    for i, c in enumerate(["noleak", "leak"]):
        ax = axes[i, j]
        cand = frames.index[(frames.domain == d) & (frames.cls == c)].values
        if not len(cand):
            ax.axis("off"); continue
        k = int(_rng.choice(cand))
        with torch.no_grad():
            S = 10 * torch.log10(_mel_cpu(torch.from_numpy(np.asarray(X_ALL[k])).float()) + 1e-10).numpy()
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, cfg.frame_sec, 0, cfg.n_mels], vmin=S.max() - 70, vmax=S.max())
        ax.set_title(f"{d} / {c}  ({frames.record_uid[k]})", fontsize=9); ax.grid(False)
        ax.set_xlabel("time (s)"); ax.set_ylabel("mel band")
savefig(fig, "spectrogram_examples",
        f"Representative log-Mel spectrograms ({F_MIN_EFF:.0f}–{F_MAX_EFF:.0f} Hz, {cfg.n_mels} bands) of one random frame "
        "per domain and class.")
''')

code(r'''
# =========================================================
# 3.9 [V2] KIỂM TRA LỐI TẮT: dải nào phân biệt TRONG miền, dải nào CHUYỂN GIAO được?
#     Within-domain vs cross-domain informativeness of each frequency region (no deep network, no target tuning).
#     Một vùng tần số phân biệt rất tốt TRONG một miền nhưng không chuyển sang miền kia là ĐẶC THÙ MIỀN
#     (đáp ứng thủy lực của mô hình thí nghiệm, bơm, chuỗi thu) chứ không phải chữ ký rò rỉ.
#     Đặc trưng: năng lượng log theo dải 1/3 octave, trừ trung bình của chính bản ghi (bất biến gain), mức BẢN GHI.
# =========================================================
SHORTCUT = pd.DataFrame()
if cfg.run_shortcut_check:
    _fc = 1000 * 2 ** (np.arange(-16, 10) / 3)                       # 25 Hz ... 6.3 kHz
    _fc = _fc[(_fc >= 25) & (_fc * 2 ** (1 / 6) <= NYQ)]
    def _third_oct(P):
        return np.array([P[(PSD_F >= f / 2 ** (1 / 6)) & (PSD_F < f * 2 ** (1 / 6))].sum() for f in _fc])
    TOB = {}
    for u, g in files.groupby("record_uid"):
        P = np.mean([FILE_STATS[i]["psd"] for i in g.file_id], axis=0)
        L = 10 * np.log10(_third_oct(P) + 1e-30)
        TOB[u] = L - L.mean()
    TOB = pd.DataFrame(TOB, index=[f"{f:.0f}" for f in _fc]).T.join(records.set_index("record_uid")[["domain", "label"]])
    TOB["impulsiveness"] = files.groupby("record_uid").file_id.agg(
        lambda ids: float(np.mean([FILE_STATS[i]["rms"] for i in ids])))   # RMS/đỉnh (đã chuẩn hoá đỉnh)
    REGIONS = [("< 100 Hz", 20, 100), ("100–300 Hz", 100, 300), ("300–1800 Hz", 300, 1800),
               ("1.8–4 kHz", 1800, 4000), ("> 4 kHz", 4000, NYQ),
               (f"front-end {F_MIN_EFF:.0f}–{F_MAX_EFF:.0f} Hz", F_MIN_EFF, F_MAX_EFF), ("full band", 20, NYQ)]
    def _cols(lo, hi):
        return [c for c, f in zip(TOB.columns[:len(_fc)], _fc) if lo <= f < hi]
    rows = []
    for name, lo, hi in REGIONS + [("impulsiveness (RMS/peak)", None, None)]:
        cols = ["impulsiveness"] if lo is None else _cols(lo, hi)
        if not cols:
            continue
        row = dict(region=name, n_features=len(cols))
        for d in DOMAINS_ALL:
            g = TOB[TOB.domain == d]
            p = cv_proba(g[cols].values, g.label.values)
            row[f"within {d}"] = (roc_auc_score(g.label.values, p)
                                  if np.isfinite(p).all() and g.label.nunique() == 2 else np.nan)
        for a in DOMAINS_ALL:
            for b in DOMAINS_ALL:
                if a == b:
                    continue
                ga, gb = TOB[TOB.domain == a], TOB[TOB.domain == b]
                sc = StandardScaler().fit(ga[cols].values)
                m = lr_pipe().fit(sc.transform(ga[cols].values), ga.label.values)
                row[f"{a} → {b}"] = roc_auc_score(gb.label.values, m.predict_proba(sc.transform(gb[cols].values))[:, 1])
        rows.append(row)
    SHORTCUT = pd.DataFrame(rows)
    save_table(SHORTCUT, "band_transferability",
               "Shortcut check: record-level AUC of a logistic-regression detector on gain-invariant 1/3-octave band "
               "energies restricted to each frequency region, within each domain (stratified CV) and across domains "
               "(train on one, test on the other). High within-domain but chance-level cross-domain AUC marks "
               "domain-specific cues; regions that stay informative across domains carry the transferable signature.", 3)
    wcols = [c for c in SHORTCUT.columns if c.startswith("within")]
    xcols = [c for c in SHORTCUT.columns if "→" in c]
    for _, r in SHORTCUT.iterrows():
        w = np.nanmax(r[wcols].values.astype(float))
        x = np.nanmin(r[xcols].values.astype(float)) if xcols else np.nan
        tag = ("không đủ dữ liệu" if not (np.isfinite(w) and np.isfinite(x)) else
               "ĐẶC THÙ MIỀN (nghi lối tắt)" if w >= 0.8 and x <= 0.6 else
               "chuyển giao được" if np.nanmin(r[wcols].values.astype(float)) >= 0.65 and x >= 0.65 else
               "ít thông tin" if w < 0.65 else "chuyển giao một phần")
        print(f"  {r.region:<28s} trong miền max {w:.2f} | chéo miền min {x:.2f} -> {tag}")
    fig, ax = plt.subplots(figsize=(13, 4))
    xx = np.arange(len(SHORTCUT)); cols_ = wcols + xcols; w_ = .8 / len(cols_)
    shades = [DOMAIN_COLOR.get(c.replace("within ", ""), "#999999") for c in wcols] + ["#444444", "#AAAAAA"][:len(xcols)]
    for k, c in enumerate(cols_):
        ax.bar(xx + (k - (len(cols_) - 1) / 2) * w_, SHORTCUT[c], w_, color=shades[k % len(shades)],
               hatch="" if c in wcols else "//", label=c)
    ax.axhline(.5, color="k", lw=.8, ls=":")
    ax.set_xticks(xx); ax.set_xticklabels(SHORTCUT.region, rotation=15); ax.set_ylim(.3, 1.02)
    ax.set_ylabel("record-level AUC"); ax.legend(ncol=2, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title("Within-domain (solid) vs cross-domain (hatched) informativeness of each frequency region")
    savefig(fig, "band_transferability",
            "Shortcut check: within-domain (solid) and cross-domain (hatched) AUC of a classical detector restricted to "
            "each frequency region. Regions informative only within one domain are acquisition- or rig-specific cues.")
''')

md(r'''
## 4. Phân bố frame / Frame distribution

Frame `cfg.frame_sec` giây, bước `cfg.hop_sec` giây; **mọi frame mang `record_uid` và `domain`**. Số frame không làm tăng
cỡ mẫu hiệu dụng — đơn vị thống kê vẫn là bản ghi (v7, mục 4).
''')

code(r'''
# =========================================================
# 4.1 Hình: frame theo miền × lớp, mức RMS theo miền / frames per domain and RMS level
# =========================================================
fig, ax = plt.subplots(1, 3, figsize=(15, 3.4))
fc = frames.groupby(["domain", "cls"]).size().unstack().reindex(columns=["noleak", "leak"]).fillna(0)
fc.plot.bar(ax=ax[0], color=[PAL["noleak"], PAL["leak"]], rot=20, width=.7)
ax[0].set_title("(a) Frames per domain and class"); ax[0].set_xlabel("")
nfr = frames.groupby(["record_uid", "domain"]).size().reset_index(name="n")
for d in DOMAINS_ALL:
    ax[1].hist(nfr.loc[nfr.domain == d, "n"], bins=15, alpha=.55, color=DOMAIN_COLOR[d], label=d)
ax[1].set_title("(b) Frames per recording"); ax[1].legend()
for d in DOMAINS_ALL:
    for c, ls in [("noleak", "--"), ("leak", "-")]:
        v = 20 * np.log10(frames.loc[(frames.domain == d) & (frames.cls == c), "rms"] + 1e-9)
        if len(v) > 5:
            kde = spstats.gaussian_kde(v); xs = np.linspace(v.min(), v.max(), 200)
            ax[2].plot(xs, kde(xs), color=DOMAIN_COLOR[d], ls=ls, label=f"{d}/{c}")
ax[2].set_xlabel("frame RMS (dBFS, after per-file peak normalisation)"); ax[2].set_title("(c) Level")
ax[2].legend(fontsize=7)
savefig(fig, "frame_distribution",
        "Frame-level composition: (a) frames per domain and class, (b) frames per recording, "
        "(c) frame RMS level by domain and class. All networks normalise each input, so level is not usable.")
''')
