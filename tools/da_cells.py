# -*- coding: utf-8 -*-
"""Nội dung notebook thích nghi miền PVC -> HDPE (1D/2D-CNN + DANN/GRL + Deep CORAL).

Mỗi phần tử của CELLS là ("markdown" | "code", nguồn). Dựng notebook bằng:

    python tools/build_da.py notebooks/leak-da-pvc-hdpe-dann-coral-kaggle.ipynb

Cơ sở: tài liệu "domain_shift.docx" (vật lý lệch miền PVC/HDPE, DANN, Deep CORAL, AdaBN,
fine-tuning chọn lọc, giả lập vật lý, giao thức đánh giá) và sơ đồ tương tác
"Domain Adaptation cho Chẩn đoán Rò rỉ Nước" (4 giai đoạn: lệch miền -> VMD + CWT + CNN/CBAM
-> GRL/CORAL -> không gian bất biến).
"""

CELLS = []


def md(s):
    CELLS.append(("markdown", s.strip("\n")))


def code(s):
    CELLS.append(("code", s.strip("\n")))


# =============================================================================
# 0. Tiêu đề
# =============================================================================
md(r'''
# Thích nghi miền PVC → HDPE cho nhận dạng rò rỉ nước bằng âm thanh: 1D/2D-CNN + DANN (GRL) + Deep CORAL

**Cross-material Unsupervised Domain Adaptation for Acoustic Leak Detection in Low-Pressure Plastic Pipes — 1D/2D-CNN with a Gradient Reversal Layer (DANN) and Deep CORAL**

---

### Bài toán / Problem

Mô hình học sâu huấn luyện trên ống **PVC** (miền nguồn $\mathcal{D}_S$, có nhãn) thường mất mạnh độ chính xác khi chuyển sang ống **HDPE** (miền đích $\mathcal{D}_T$). HDPE mềm hơn nhiều ($E \approx 0{,}7$–$1{,}1$ GPa so với $2{,}8$–$3{,}4$ GPa của PVC), nên theo phương trình Korteweg hiệu chỉnh

$$c = c_0\left[1 + \frac{B\,D}{E\,e}\right]^{-1/2}$$

vận tốc sóng âm trong cột nước giảm (≈ 380–480 m/s → 180–260 m/s). Phổ rò rỉ vì thế **dịch xuống** (≈ 80–280 Hz → 20–120 Hz) và suy hao phụ thuộc tần số **dốc hơn**. Hệ quả là *covariate shift* $P_S(X) \neq P_T(X)$.

Notebook hiện thực trọn chuỗi xử lý của sơ đồ tương tác kèm theo và tài liệu "domain_shift":

| Giai đoạn (sơ đồ) | Nội dung | Mục |
|---|---|---|
| **1. Lệch miền vật lý** | Korteweg $c_{PVC}$ vs $c_{HDPE}$; PSD, tần số trọng tâm, tâm mode VMD theo miền | 3, 4, 5 |
| **2. Tiền xử lý & CNN** | Lọc dải 15–350 Hz, notch 50 Hz, **VMD** khử nhiễu, **CWT** Morlet (thang tần số log) hoặc log-Mel, **CMVN theo miền**; **1D-CNN** (sóng thô) và **2D-CNN + CBAM** dùng chung trọng số $G_f$ | 3, 5, 6 |
| **3. Căn chỉnh miền** | **GRL/DANN**: $G_d$ phân biệt PVC/HDPE, GRL nhân gradient với $-\lambda_p$; **Deep CORAL**: $\frac{1}{4d^2}\lVert C_S - C_T\rVert_F^2$ | 6, 8 |
| **4. Hội tụ bất biến** | t-SNE, proxy $\mathcal{A}$-distance, khoảng cách CORAL/MMD trong không gian ẩn | 11 |

Hàm mục tiêu (minimax, hiện thực bằng GRL):

$$
\min_{\theta_f,\theta_y}\max_{\theta_d}\;
\underbrace{\frac{1}{N_S}\sum_{i}\mathcal{L}_y\big(G_y(G_f(x_i^S)),y_i^S\big)}_{\text{chỉ PVC có nhãn}}
-\lambda_p\underbrace{\Big[\frac{1}{N_S}\sum_i\mathcal{L}_d\big(G_d(f_i^S),0\big)+\frac{1}{N_T}\sum_j\mathcal{L}_d\big(G_d(f_j^T),1\big)\Big]}_{\text{DANN}}
+\beta_p\underbrace{\frac{1}{4d^2}\lVert C_S-C_T\rVert_F^2}_{\text{Deep CORAL}}
$$

với $\lambda_p=\lambda_{\max}\big(\frac{2}{1+e^{-\gamma p}}-1\big)$, $p\in[0,1]$ là tiến độ huấn luyện (Ganin & Lempitsky, 2015); $\beta_p$ dùng cùng lịch tăng dần.

### Thiết kế thực nghiệm (mục 5 tài liệu) / Experimental design

| Phương pháp | Dữ liệu HDPE dùng khi huấn luyện | Vai trò |
|---|---|---|
| Source-only | không | cận dưới, dùng để chứng minh lệch miền |
| AdaBN | không nhãn (chỉ thống kê BN + CMVN) | thích nghi thống kê, rẻ nhất |
| Deep CORAL | không nhãn | căn chỉnh hiệp phương sai (bậc 2) |
| DANN (GRL) | không nhãn | căn chỉnh đối kháng |
| **DANN + CORAL** | không nhãn | **phương pháp đề xuất** |
| DANN + CORAL + PhysAug | không nhãn | thêm giả lập vật lý PVC→HDPE (mục 4 tài liệu) |
| Few-shot FT | ~10 % bản ghi có nhãn | tinh chỉnh chọn lọc theo tầng (LR 1e-4 / 1e-5 / 1e-3) |
| Oracle | toàn bộ phần thích nghi, **có nhãn** | cận trên lý thuyết |

**Chống rò rỉ dữ liệu:** mọi phân chia đều theo **bản ghi** (`Rxxx`). Bản ghi HDPE được chia K-fold thành *phần thích nghi* (không nhãn) và *phần kiểm thử* độc lập. Nhãn HDPE **chỉ** dùng để đánh giá, và điều này được kiểm tra bằng assertion. Không bước chọn epoch hay ngưỡng nào dùng nhãn HDPE: mô hình lấy ở **epoch cuối** (quy tắc đăng ký trước), ngưỡng cố định 0,5.

### Bố trí dữ liệu / Data layout

```
/kaggle/input/<dataset>/
├── PVC/                    (hoặc chỉ định cfg.source_root)
│   ├── leak/    R001_seg01.wav, R001_seg02.wav, ...
│   └── noleak/  R002_seg01.wav, ...
└── HDPE/                   (hoặc chỉ định cfg.target_root)
    ├── leak/    ...        <- nhãn chỉ dùng để ĐÁNH GIÁ
    ├── noleak/  ...
    └── unlabeled/ ...      <- tuỳ chọn: bản ghi chưa biết nhãn (chỉ dùng để thích nghi)
```

Notebook tự nhận diện một trong ba chế độ chạy:

1. **`real`**: có cả PVC và HDPE thật. Kết quả là bằng chứng thực nghiệm về chuyển giao PVC→HDPE.
2. **`physics_sim`**: chỉ có PVC thật (ví dụ bộ `field_rec_8k_V7.2` của notebook v7). Một phần bản ghi PVC **tách rời** được biến đổi vật lý (dịch tần theo $c_{HDPE}/c_{PVC}$, suy hao nhớt đàn hồi $e^{-\alpha(f)x}$, nhiễu nền) thành miền "giả lập HDPE". Chế độ này chỉ kiểm chứng pipeline, **không** phải bằng chứng về ống HDPE thật.
3. **`synthetic`**: không tìm thấy dữ liệu nào. Notebook tự sinh dữ liệu mô phỏng hai miền.

> **Cách dùng nhanh / Quick start**
> 1. `Add Data`: gắn dataset PVC (và HDPE nếu có). Sửa `cfg.source_root` / `cfg.target_root` hoặc `cfg.data_root` ở mục 1.2.
> 2. `Settings → Accelerator = GPU T4 x2`. Không cần Internet.
> 3. `Run All`: khoảng 30–60 phút với cấu hình mặc định (2 kiến trúc × 6 phương pháp huấn luyện × 3 fold đích, chạy song song trên 2 GPU).
> 4. Chạy thử nhanh: biến môi trường `LEAK_SMOKE=1` (3 epoch, 2 fold).
> 5. Kịch bản triển khai thật (HDPE chưa có nhãn): đặt tệp HDPE vào `unlabeled/` hoặc đặt `cfg.hide_target_labels=True`. Notebook bỏ phần đánh giá, huấn luyện mô hình cuối và xuất **bảng xếp hạng bản ghi HDPE nghi rò** để khảo sát hiện trường.
''')

# =============================================================================
# 1. Môi trường
# =============================================================================
md(r'''
## 1. Thiết lập môi trường / Environment setup
''')

code(r'''
# =========================================================
# 1.1 Imports & phụ thuộc / Imports and dependencies
#     Không phụ thuộc torchaudio: STFT/Mel/CWT viết bằng torch thuần -> chạy được
#     trên mọi phiên bản torch của Kaggle.
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
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from IPython.display import display, Markdown

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_curve, average_precision_score, balanced_accuracy_score,
                             f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def amp_autocast(device, enabled):
    # API AMP tương thích nhiều phiên bản PyTorch
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
#     D_S = miền nguồn (ống PVC, CÓ nhãn)
#     D_T = miền đích  (ống HDPE, nhãn CHỈ dùng để đánh giá)
# =========================================================
@dataclass
class CFG:
    # ---- Dữ liệu hai miền / Two-domain data --------------------------------------
    # Thứ tự ưu tiên: source_root/target_root (nếu tồn tại) > data_root/{PVC,HDPE}
    # > tự dò trong /kaggle/input theo tên thư mục (pvc/upvc | hdpe/pe100/pe80).
    # source_root mặc định = bộ dữ liệu PVC hiện trường của notebook v7.
    data_root: str = ""
    source_root: str = "/kaggle/input/datasets/nguyennguyenkhac/tapdulieutt-8k-v7-2/field_rec_8k_V7.2"
    target_root: str = ""
    source_name: str = "PVC"
    target_name: str = "HDPE"
    source_keys: tuple = ("pvc", "upvc", "u_pvc")
    target_keys: tuple = ("hdpe", "pe100", "pe80", "pe_100", "pe_80", "polyethylene")
    leak_keys: tuple = ("leak", "ro_ri", "rori")
    noleak_keys: tuple = ("noleak", "no_leak", "non_leak", "nonleak", "normal", "background",
                          "khong_ro_ri", "khongroli")
    unlabeled_keys: tuple = ("unlabeled", "unlabelled", "unknown", "chua_gan_nhan", "khong_nhan")
    audio_ext: tuple = (".wav", ".flac", ".ogg", ".mp3")
    record_regex: str = r"(R\d{1,4})"      # mã bản ghi = đơn vị nhóm khi chia dữ liệu
    strict_checks: bool = True             # DỪNG khi có mâu thuẫn nhãn / trùng tệp giữa hai miền
    # Chỉ có PVC thật -> tạo miền đích "giả lập HDPE" từ các bản ghi PVC TÁCH RỜI (mục 2.3)
    target_fallback: str = "physics_sim"   # "physics_sim" | "stop"
    sim_target_frac: float = 0.5           # tỉ lệ bản ghi PVC (mỗi lớp) chuyển sang miền đích giả lập
    sim_snr_db: tuple = (5.0, 15.0)        # nhiễu nền của miền giả lập
    hide_target_labels: bool = False       # True = kịch bản triển khai: HDPE hoàn toàn KHÔNG nhãn

    # ---- Tín hiệu / Signal (mục 3.1 tài liệu: tập trung dải thấp) ----------------
    target_sr: int = 2000                  # Nyquist 1 kHz: đủ cho dải 15-350 Hz, nhẹ hơn 8 kHz 4 lần
    band_hz: tuple = (15.0, 350.0)         # cắt > 400 Hz: tránh học vẹt thành phần chỉ có ở PVC
    filter_order: int = 4
    notch_mains: bool = True
    mains_hz: float = 50.0
    notch_q: float = 30.0
    frame_sec: float = 2.0                 # 2 s = 30 chu kỳ ở 15 Hz
    hop_sec: float = 1.0
    eda_max_records: int = 40              # số bản ghi tối đa / (miền, lớp) cho phân tích phổ (mục 4)

    # ---- VMD (Variational Mode Decomposition, sơ đồ - nút 3) ---------------------
    use_vmd: bool = True
    vmd_K: int = 4
    vmd_alpha: float = 1500.0
    vmd_tau: float = 0.0                   # 0 = chịu nhiễu (không ép tổng mode = tín hiệu)
    vmd_iters: int = 150
    vmd_tol: float = 1e-6
    vmd_batch: int = 256
    vmd_mains_tol_hz: float = 2.0          # bỏ mode có tâm trùng hài lưới điện (thành phần tonal)

    # ---- Biểu diễn thời gian-tần số / Time-frequency front-end ------------------
    tf_repr: str = "cwt"                   # "cwt" (Morlet, tần số log) | "logmel"
    n_freqs: int = 64
    tf_hop: int = 20                       # 10 ms @ 2 kHz
    cwt_w0: float = 6.0
    n_fft: int = 1024                      # chỉ dùng cho "logmel"
    input_norm: str = "domain_cmvn"        # "domain_cmvn" (z-score theo dải, theo MIỀN) | "frame"
    specaug_freq_mask: int = 6
    specaug_time_mask: int = 12

    # ---- Vật lý / Physics (Korteweg + suy hao nhớt đàn hồi) ----------------------
    c0: float = 1480.0                     # m/s, âm trong nước tự do
    K_water: float = 2.2e9                 # Pa, mô-đun đàn hồi thể tích của nước (B)
    E_pvc: float = 3.0e9
    E_hdpe: float = 0.9e9
    sdr_pvc: float = 17.0                  # SDR = D/e
    sdr_hdpe: float = 17.0
    phys_warp_range: tuple = (0.50, 0.65)  # c_HDPE / c_PVC (tài liệu mục 4)
    phys_alpha: float = 4.6e-4             # Np/(m.Hz): thêm ~ -6 dB ở 100 Hz sau 15 m
    phys_dist_m: tuple = (5.0, 25.0)
    phys_noise_snr_db: tuple = (-5.0, 5.0)
    phys_aug_p: float = 0.5
    phys_noise_p: float = 0.5

    # ---- Mô hình / Models ----------------------------------------------------------
    backbones: tuple = ("CNN2D", "CNN1D")  # G_f: 2D-CNN+CBAM trên CWT | 1D-CNN trên sóng thô
    main_backbone: str = "CNN2D"           # cố định TRƯỚC cho hình không gian ẩn & mô hình xuất
    emb_dim: int = 128                     # d trong 1/(4 d^2) của CORAL
    use_cbam: bool = True
    disc_hidden: int = 128
    dropout: float = 0.3

    # ---- Thích nghi miền / Domain adaptation ---------------------------------------
    methods: tuple = ("source_only", "adabn", "coral", "dann", "dann_coral", "dann_coral_phys",
                      "fewshot_src", "fewshot_da", "oracle")   # thêm "mmd" nếu muốn Deep MMD
    lambda_dann: float = 1.0               # lambda_max của GRL
    grl_gamma: float = 10.0                # lambda_p = lambda_max * (2/(1+exp(-gamma p)) - 1)
    coral_weight: float = 1.0
    mmd_weight: float = 1.0
    ramp_alignment: bool = True            # CORAL/MMD dùng cùng lịch tăng dần với GRL
    final_method: str = "dann_coral"       # mô hình xuất triển khai - ĐĂNG KÝ TRƯỚC

    # ---- Huấn luyện / Training -------------------------------------------------------
    epochs: int = 30
    batch_size: int = 64                   # số mẫu MỖI MIỀN trong một bước
    eval_batch: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.1
    label_smoothing: float = 0.05
    grad_clip: float = 5.0
    amp: bool = True
    aug_noise_p: float = 0.5
    aug_noise_snr_db: tuple = (10.0, 30.0)
    aug_shift_p: float = 0.5
    # Few-shot: tinh chỉnh chọn lọc theo tầng (mục 3.2 tài liệu)
    ft_epochs: int = 15
    ft_lr_early: float = 1e-4              # tầng đầu: KHÔNG đóng băng (phổ HDPE dịch xuống)
    ft_lr_mid: float = 1e-5                # tầng giữa: LR rất nhỏ
    ft_lr_head: float = 1e-3               # đầu phân loại: LR lớn
    fewshot_frac: float = 0.10
    fewshot_min_records_per_class: int = 1

    # ---- Chia dữ liệu / Splits ----------------------------------------------------------
    n_target_folds: int = 3
    source_val_frac: float = 0.2
    seed: int = 42
    seeds: tuple = (42,)                   # nhiều seed = lặp toàn bộ với phân hoạch + khởi tạo mới

    # ---- Đánh giá / Evaluation ----------------------------------------------------------
    decision_threshold: float = 0.5        # cố định: không có nhãn HDPE để chọn ngưỡng
    bootstrap_n: int = 2000
    tsne_max_points: int = 2000

    # ---- Hạ tầng / Infrastructure -----------------------------------------------------
    parallel_jobs: bool = True             # chạy song song các job trên nhiều GPU (nhanh ~2x, nhưng dropout dùng
                                           # RNG toàn cục -> không tái lập bit-exact; đặt False khi chốt số liệu)
    max_parallel: int = 2
    gpu_resident_max_gb: float = 4.0
    verbose_epochs: bool = False
    train_final_model: bool = True
    out_dir: str = "/kaggle/working"
    fig_dpi: int = 300
    smoke_test: bool = False


cfg = CFG()
# ---- Ghi đè bằng biến môi trường (chạy ngoài Kaggle / smoke test) ----
if os.environ.get("LEAK_SMOKE") == "1":
    cfg.smoke_test = True
if os.environ.get("LEAK_DATA") is not None:          # thư mục chứa PVC/ và HDPE/ ("" = tự dò)
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

# ---- Đặc tả các phương pháp (mục 5 tài liệu) ----
#   kind="train": huấn luyện đầy đủ | kind="post": dựng trên mô hình đã huấn luyện (base)
#   src/tgt: dùng PVC có nhãn / HDPE KHÔNG nhãn khi huấn luyện; label_dom: miền của tập có nhãn
METHOD_INFO = {
    "source_only":     dict(kind="train", src=True,  tgt=False, dann=False, coral=False, mmd=False, phys=False, label_dom=0),
    "coral":           dict(kind="train", src=True,  tgt=True,  dann=False, coral=True,  mmd=False, phys=False, label_dom=0),
    "mmd":             dict(kind="train", src=True,  tgt=True,  dann=False, coral=False, mmd=True,  phys=False, label_dom=0),
    "dann":            dict(kind="train", src=True,  tgt=True,  dann=True,  coral=False, mmd=False, phys=False, label_dom=0),
    "dann_coral":      dict(kind="train", src=True,  tgt=True,  dann=True,  coral=True,  mmd=False, phys=False, label_dom=0),
    "dann_coral_phys": dict(kind="train", src=True,  tgt=True,  dann=True,  coral=True,  mmd=False, phys=True,  label_dom=0),
    "oracle":          dict(kind="train", src=False, tgt=False, dann=False, coral=False, mmd=False, phys=False, label_dom=1,
                            needs_target_labels=True),
    "adabn":           dict(kind="post", base="source_only", finetune=False),
    "fewshot_src":     dict(kind="post", base="source_only", finetune=True, needs_target_labels=True),
    "fewshot_da":      dict(kind="post", base="dann_coral",  finetune=True, needs_target_labels=True),
}
METHOD_LABEL = {"source_only": "Source-only", "adabn": "AdaBN", "coral": "Deep CORAL", "mmd": "Deep MMD",
                "dann": "DANN (GRL)", "dann_coral": "DANN + CORAL", "dann_coral_phys": "DANN + CORAL + PhysAug",
                "fewshot_src": "Few-shot FT (từ Source-only)", "fewshot_da": "Few-shot FT (từ DANN+CORAL)",
                "oracle": "Oracle (HDPE có nhãn)"}
UDA_METHODS = ("adabn", "coral", "mmd", "dann", "dann_coral", "dann_coral_phys")

_unknown = [m for m in cfg.methods if m not in METHOD_INFO]
assert not _unknown, f"Phương pháp không xác định: {_unknown}"
_m = list(cfg.methods)
for m in list(_m):                       # tự thêm mô hình nền mà phương pháp "post" cần
    base = METHOD_INFO[m].get("base")
    if base and base not in _m:
        _m.insert(0, base)
        print(f"  [cfg] '{m}' cần '{base}' -> tự thêm '{base}' vào cfg.methods")
cfg.methods = tuple(dict.fromkeys(_m))
TRAIN_METHODS = [m for m in cfg.methods if METHOD_INFO[m]["kind"] == "train"]
POST_METHODS = [m for m in cfg.methods if METHOD_INFO[m]["kind"] == "post"]
assert METHOD_INFO[cfg.final_method]["kind"] == "train" and METHOD_INFO[cfg.final_method]["src"], \
    "cfg.final_method phải là phương pháp huấn luyện dùng PVC có nhãn"
assert cfg.main_backbone in cfg.backbones
assert cfg.band_hz[1] < cfg.target_sr / 2, "band_hz vượt Nyquist"

OUT = Path(cfg.out_dir)
FIG_DIR, TAB_DIR = OUT / "figures", OUT / "tables"
CACHE_DIR, EXPORT_DIR = OUT / "cache", OUT / "export"
for _d in (FIG_DIR, TAB_DIR, CACHE_DIR, EXPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
FRAME_LEN = int(round(cfg.frame_sec * cfg.target_sr))
HOP_LEN = int(round(cfg.hop_sec * cfg.target_sr))

print(json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
                 indent=1, ensure_ascii=False))
print(f"\nFRAME_LEN = {FRAME_LEN} mẫu ({cfg.frame_sec}s @ {cfg.target_sr} Hz) | HOP = {HOP_LEN}")
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
    torch.backends.cudnn.benchmark = True      # nhanh hơn; kết quả không tất định tuyệt đối

def make_generator(device, seed):
    # Generator RIÊNG cho từng job/thiết bị (cuda:1 cần generator của cuda:1)
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
PAL = {"src": "#2E6F95", "tgt": "#C8553D", "leak": "#C8553D", "noleak": "#2E6F95",
       "acc": "#3F7D20", "warn": "#E0A458", "aux": "#8C6BB1", "grey": "#6E7B8B"}
# 4 nhóm của sơ đồ: PVC Leak / PVC Safe / HDPE Leak / HDPE Safe  (khoá = (miền, nhãn))
GROUP_STYLE = {
    (0, 1): dict(label=f"{cfg.source_name} Leak", color="#B2182B", marker="o"),
    (0, 0): dict(label=f"{cfg.source_name} Safe", color="#2166AC", marker="o"),
    (1, 1): dict(label=f"{cfg.target_name} Leak", color="#E08214", marker="^"),
    (1, 0): dict(label=f"{cfg.target_name} Safe", color="#35978F", marker="^"),
    (1, -1): dict(label=f"{cfg.target_name} (không nhãn)", color="#8C8C8C", marker="^"),
}
METHOD_COLORS = {"source_only": "#6E7B8B", "adabn": "#A6761D", "coral": "#1B9E77", "mmd": "#66A61E",
                 "dann": "#7570B3", "dann_coral": "#D95F02", "dann_coral_phys": "#E7298A",
                 "fewshot_src": "#8DA0CB", "fewshot_da": "#FC8D62", "oracle": "#222222"}

FIG_INDEX = {}
def savefig(fig, name, caption=""):
    # Lưu PNG (xem nhanh) + PDF (đưa vào bản thảo)
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

Quy ước nhãn: `leak` = **1**, `noleak` / `normal` / `background` = **0**, `unlabeled` = **−1** (chỉ cho miền đích).
Mã bản ghi được **gắn tiền tố miền** (`PVC:R001`, `HDPE:R001`) nên hai bản ghi trùng số ở hai miền không
bao giờ bị gộp. Bản ghi là đơn vị chia dữ liệu và là đơn vị thống kê.
''')

code(r'''
# =========================================================
# 2.1 Dò thư mục hai miền / Locate the PVC and HDPE roots
# =========================================================
def _norm_token(s):
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.lower().replace("-", "_").replace(" ", "_")

def _key_regex(keys):
    ks = sorted({_norm_token(k) for k in keys}, key=len, reverse=True)
    # khoá phải đứng đầu một "từ" (đầu chuỗi hoặc sau "_" / "."), sau nó là hết chuỗi, "_", "." hoặc chữ số
    return re.compile(r"(?:^|[_.])(?:" + "|".join(map(re.escape, ks)) + r")(?=$|[_.\d])")

_RX = dict(noleak=_key_regex(cfg.noleak_keys), leak=_key_regex(cfg.leak_keys),
           unlab=_key_regex(cfg.unlabeled_keys),
           src=_key_regex(cfg.source_keys), tgt=_key_regex(cfg.target_keys))

def label_of_token(s):
    s = _norm_token(s)
    if _RX["noleak"].search(s):         # kiểm tra 'noleak' TRƯỚC 'leak'
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
for _t, _want in [("PVC", 0), ("uPVC", 0), ("HDPE", 1), ("pe100", 1), ("pvc_vs_hdpe", None),
                  ("pipe", None), ("field_rec_8k_V7.2", None)]:
    assert domain_of_token(_t) == _want, f"bộ nhận miền sai với '{_t}'"

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
    # Thư mục có TÊN thuộc miền `dom` và chứa âm thanh; nông nhất trước.
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
                    out.append(k)                 # không đi sâu hơn: mọi thứ bên trong thuộc miền này
                elif depth < max_depth:
                    stack.append((k, depth + 1))
    out = [d for d in dict.fromkeys(out) if _has_audio(d)]
    return sorted(out, key=lambda p: (len(p.parts), str(p)))

def dirs_with_leak_noleak(base, max_depth=6):
    # Bộ dữ liệu MỘT miền kiểu v7 (leak/ + noleak/) -> dùng làm miền nguồn
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
    src = tgt = None
    how = {}
    if cfg.source_root:
        if Path(cfg.source_root).exists():
            src, how["source"] = Path(cfg.source_root), "cfg.source_root"
        else:
            print(f"!! cfg.source_root không tồn tại: {cfg.source_root} -> tự dò.")
    if cfg.target_root:
        if Path(cfg.target_root).exists():
            tgt, how["target"] = Path(cfg.target_root), "cfg.target_root"
        else:
            print(f"!! cfg.target_root không tồn tại: {cfg.target_root} -> tự dò.")
    if cfg.data_root and not Path(cfg.data_root).exists():
        print(f"!! cfg.data_root không tồn tại: {cfg.data_root} -> tự dò trong /kaggle/input.")
    bases = ([Path(cfg.data_root)] if (cfg.data_root and Path(cfg.data_root).exists())
             else [Path("/kaggle/input"), Path("./data")])
    if tgt is None:
        c = [d for d in find_domain_dirs(bases, 1) if not _is_within(d, src)]
        if c:
            tgt, how["target"] = c[0], "tự dò theo tên thư mục"
            if len(c) > 1:
                print(f"!! {len(c)} thư mục miền đích; dùng {c[0]}. Các ứng viên khác: {c[1:6]}")
    if src is None:
        c = [d for d in find_domain_dirs(bases, 0) if not _is_within(d, tgt)]
        if c:
            src, how["source"] = c[0], "tự dò theo tên thư mục"
        else:
            c = [d for b in bases for d in dirs_with_leak_noleak(b)
                 if not (_is_within(d, tgt) or _is_within(tgt, d))]
            if c:
                src, how["source"] = c[0], "bộ leak/noleak một miền -> coi là PVC"
    if src is not None and tgt is not None and (_is_within(src, tgt) or _is_within(tgt, src)):
        raise RuntimeError(f"Thư mục nguồn {src} và đích {tgt} lồng nhau - hãy chỉ định riêng từng miền.")
    return src, tgt, how

SRC_ROOT, TGT_ROOT, ROOT_HOW = resolve_domain_roots()
if SRC_ROOT is None:
    DATA_MODE = "synthetic"
elif TGT_ROOT is None:
    if cfg.target_fallback != "physics_sim":
        raise RuntimeError("Không tìm thấy dữ liệu HDPE (cfg.target_fallback='stop').")
    DATA_MODE = "physics_sim"
else:
    DATA_MODE = "real"

print(f"Nguồn ({cfg.source_name}) : {SRC_ROOT}  [{ROOT_HOW.get('source', '-')}]")
print(f"Đích  ({cfg.target_name}) : {TGT_ROOT}  [{ROOT_HOW.get('target', '-')}]")
print(f"CHẾ ĐỘ DỮ LIỆU   : {DATA_MODE}")
if DATA_MODE == "physics_sim":
    print("!! Chỉ có dữ liệu PVC thật -> miền đích sẽ là 'giả lập HDPE' từ các bản ghi PVC tách rời.")
    print("   Mọi con số trên miền đích chỉ kiểm chứng PIPELINE, không phải kết quả trên ống HDPE thật.")
''')

code(r'''
# =========================================================
# 2.2 Dữ liệu mô phỏng hai miền / Synthetic two-domain data
#     Chỉ dùng khi KHÔNG tìm thấy dữ liệu thật.
# =========================================================
def _pink_np(rng, n, sr, slope=1.0):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / sr); f[0] = f[1]
    y = np.fft.irfft(X / f ** (slope / 2.0), n)
    return y / (np.std(y) + 1e-12)

def synthesise_two_domains(root, n_rec_per_class=10, files_per_rec=2, dur=20.0, sr=4000, seed=0,
                           witness=0.7):
    """Sinh dữ liệu MÔ PHỎNG theo vật lý của tài liệu:
       - PVC : rò rỉ = nhiễu dải ~80-290 Hz; nền = nhiễu hồng (độ dốc 1)
       - HDPE: cùng cơ chế nhưng mọi tần số nhân r = c_HDPE/c_PVC in cfg.phys_warp_range,
               thêm suy hao e^{-alpha f x} (x = 5-25 m) và SNR thấp hơn (áp lực thấp); nền dốc hơn
       - mỗi bản ghi có một cộng hưởng tĩnh riêng ở CẢ HAI lớp ("dấu vân tay" đoạn ống/cảm biến)
       - tiếng rò chỉ hiện diện ~witness thời lượng; có hài 50 Hz, tiếng bơm và transient xe cộ."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    n = int(dur * sr); t = np.arange(n) / sr
    rid = 0
    for dom, dname in ((0, cfg.source_name), (1, cfg.target_name)):
        for cls in ("leak", "noleak"):
            (root / dname / cls).mkdir(parents=True, exist_ok=True)
            for _ in range(n_rec_per_class):
                rid += 1
                warp = rng.uniform(*cfg.phys_warp_range) if dom == 1 else 1.0
                dist = rng.uniform(*cfg.phys_dist_m) if dom == 1 else rng.uniform(1.0, 5.0)
                f_res, q_res = rng.uniform(40, 300) * warp, rng.uniform(4, 10)
                f0, bw = rng.uniform(140, 220), rng.uniform(50, 70)
                lo, hi = max(10.0, (f0 - bw) * warp), (f0 + bw) * warp
                gain_db = rng.uniform(2, 10) if dom == 0 else rng.uniform(-2, 6)
                f_pump = rng.uniform(22, 45)
                b_res, a_res = sps.iirpeak(f_res / (sr / 2), q_res)
                for seg in range(1, files_per_rec + 1):
                    x = _pink_np(rng, n, sr, 1.0 if dom == 0 else 1.6)
                    res = sps.lfilter(b_res, a_res, rng.standard_normal(n))
                    x = x + 0.5 * res / (np.std(res) + 1e-12)
                    for h in (1, 2, 3):
                        x += 0.08 * rng.uniform(.5, 1.) * np.sin(2 * np.pi * cfg.mains_hz * h * t + rng.uniform(0, 6.28))
                    x += 0.15 * np.sin(2 * np.pi * f_pump * t + rng.uniform(0, 6.28))
                    if cls == "leak":
                        sos = sps.butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band", output="sos")
                        lk = sps.sosfilt(sos, rng.standard_normal(n))
                        if dom == 1:
                            fq = np.fft.rfftfreq(n, 1.0 / sr)
                            lk = np.fft.irfft(np.fft.rfft(lk) * np.exp(-cfg.phys_alpha * fq * dist), n)
                        lk /= (np.std(lk) + 1e-12)
                        mask = np.zeros(n); k_int = int(rng.integers(2, 5))
                        for _ in range(k_int):
                            L = int(witness * n / k_int); s0 = int(rng.integers(0, max(1, n - L)))
                            mask[s0:s0 + L] = 1.0
                        win = np.hanning(int(.3 * sr)); mask = np.convolve(mask, win / win.sum(), "same")
                        x = x + lk * mask * 10 ** (gain_db / 20.0)
                    for _ in range(int(rng.integers(2, 6))):
                        s0 = int(rng.integers(0, n - sr))
                        x[s0:s0 + sr] += 3.0 * np.exp(-np.arange(sr) / (.05 * sr)) * rng.standard_normal(sr)
                    x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
                    sf.write(root / dname / cls / f"R{rid:03d}_seg{seg:02d}.wav", x.astype(np.float32), sr)
    return root

if DATA_MODE == "synthetic":
    _kw = (dict(n_rec_per_class=6, files_per_rec=1, dur=12.0) if cfg.smoke_test
           else dict(n_rec_per_class=10, files_per_rec=2, dur=20.0))
    SYN_ROOT = synthesise_two_domains(OUT / "synthetic_da", seed=cfg.seed, **_kw)
    SRC_ROOT, TGT_ROOT = SYN_ROOT / cfg.source_name, SYN_ROOT / cfg.target_name
    ROOT_HOW = dict(source="synthetic", target="synthetic")
    print("!! SYNTHETIC DEMO MODE:", SYN_ROOT, "|", len(audio_files_in(SYN_ROOT)), "tệp")
''')

code(r'''
# =========================================================
# 2.3 Kiểm kê tệp + mã bản ghi + miền giả lập (nếu cần)
# =========================================================
_REC_RE = re.compile(cfg.record_regex, re.IGNORECASE)

def extract_record_id(path):
    m = _REC_RE.search(path.stem)
    if m:
        return m.group(1).upper(), "regex:stem"
    m = _REC_RE.search(path.parent.name)
    if m:
        return m.group(1).upper(), "regex:parent"
    return path.stem, "fallback:stem"

def label_from_path(p, root):
    # Chỉ xét phần đường dẫn BÊN DƯỚI thư mục miền (tên dataset phía trên không được gán nhãn)
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
            skipped.append((str(p), "PVC không nhãn")); continue
        try:
            info = sf.info(str(p)); dur, sr0 = float(info.duration), int(info.samplerate)
        except Exception:
            try:
                dur, sr0 = float(librosa.get_duration(path=str(p))), int(librosa.get_samplerate(str(p)))
            except Exception as e:
                skipped.append((str(p), f"không đọc được ({type(e).__name__})")); continue
        rid, how = extract_record_id(p)
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

# ---- Chế độ physics_sim: tách bản ghi PVC thành miền đích giả lập ----
SIM_PARAMS = {}
if DATA_MODE == "physics_sim":
    _rec0 = files.groupby("record_id").label.first()
    _rng = np.random.default_rng(cfg.seed + 2024)
    _pick = []
    for lab in (0, 1):
        r = sorted(_rec0.index[_rec0 == lab])
        k = max(min(int(round(cfg.sim_target_frac * len(r))), len(r) - 2), 0)
        if k < 2:
            raise RuntimeError(f"Lớp {lab} chỉ có {len(r)} bản ghi PVC - cần >= 4 để tách miền đích giả lập.")
        _pick += list(_rng.choice(r, size=k, replace=False))
    _msk = files.record_id.isin(_pick)
    files.loc[_msk, "domain"] = 1
    files.loc[_msk, "dom_name"] = cfg.target_name
    files.loc[_msk, "record_id"] = cfg.target_name + "sim:" + files.loc[_msk, "record_raw"]
    for rid in sorted(files.loc[_msk, "record_id"].unique()):
        rr = np.random.default_rng(int(hashlib.sha1(rid.encode()).hexdigest()[:8], 16) ^ cfg.seed)
        SIM_PARAMS[rid] = dict(warp=float(rr.uniform(*cfg.phys_warp_range)),
                               dist=float(rr.uniform(*cfg.phys_dist_m)),
                               snr_db=float(rr.uniform(*cfg.sim_snr_db)), seed=int(rr.integers(1 << 30)))
    print(f"physics_sim: {len(SIM_PARAMS)} bản ghi PVC -> '{cfg.target_name}sim' "
          f"(r = {cfg.phys_warp_range}, x = {cfg.phys_dist_m} m, SNR = {cfg.sim_snr_db} dB)")

if cfg.hide_target_labels:
    files.loc[files.domain == 1, "label"] = -1
    print("!! cfg.hide_target_labels=True -> mọi tệp miền đích coi là KHÔNG nhãn.")
files["cls"] = files.label.map({1: "leak", 0: "noleak", -1: "unlabeled"})
files = files.sort_values(["domain", "cls", "record_id", "file_stem"]).reset_index(drop=True)
files.to_csv(OUT / "manifest_files.csv", index=False)

print(f"Số tệp: {len(files)} | bỏ qua {len(SKIPPED)}")
for p_, why in SKIPPED[:8]:
    print("   [bỏ qua]", Path(p_).name, "-", why)
print("Trích mã bản ghi:", files.rid_source.value_counts().to_dict())
files.head(6)
''')

code(r'''
# =========================================================
# 2.4 Kiểm tra toàn vẹn + bảng mức BẢN GHI (đơn vị thống kê)
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
    # Sao chép nhầm tệp PVC vào thư mục HDPE = rò rỉ dữ liệu giữa hai miền
    files["qhash"] = files.path.map(_quick_hash)
    _cross = set(files.qhash[files.domain == 0]) & set(files.qhash[files.domain == 1])
    if _cross:
        problems.append(f"{len(_cross)} tệp có NỘI DUNG trùng nhau giữa {cfg.source_name} và {cfg.target_name}")
_nfb = int((files.rid_source == "fallback:stem").sum())
if _nfb:
    notes.append(f"{_nfb} tệp không khớp regex '{cfg.record_regex}' -> lấy tên tệp làm mã bản ghi "
                 f"(các đoạn cắt từ cùng một bản ghi có thể bị tách - hãy kiểm tra)")

records = (files.groupby("record_id")
           .agg(domain=("domain", "first"), dom_name=("dom_name", "first"), label=("label", "first"),
                n_files=("path", "size"), duration_s=("duration_s", "sum"))
           .reset_index())
records["cls"] = records.label.map({1: "leak", 0: "noleak", -1: "unlabeled"})
records.to_csv(OUT / "manifest_records.csv", index=False)
REC_LABEL = dict(zip(records.record_id, records.label.astype(int)))
REC_DOM = dict(zip(records.record_id, records.domain.astype(int)))

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
                 f"(cần >= 2 bản ghi mỗi lớp). Notebook sẽ chỉ huấn luyện mô hình cuối + xếp hạng.")

for s in notes:
    print("  [lưu ý]", s)
if problems:
    for s in problems:
        print("  !! LỖI:", s)
    if cfg.strict_checks:
        raise RuntimeError("Dữ liệu có mâu thuẫn - xem danh sách LỖI ở trên (hoặc đặt cfg.strict_checks=False).")

summary = (records.groupby(["dom_name", "cls"])
           .agg(n_records=("record_id", "nunique"), n_files=("n_files", "sum"),
                total_min=("duration_s", lambda s: s.sum() / 60), mean_s=("duration_s", "mean"))
           .round(2))
display(Markdown("### Table 1. Recording-level inventory per domain"))
display(summary)
summary.to_csv(TAB_DIR / "table1_inventory.csv")
print(f"Chế độ: {DATA_MODE} | bản ghi {cfg.source_name}: {int((records.domain == 0).sum())} | "
      f"bản ghi {cfg.target_name}: {int((records.domain == 1).sum())} "
      f"(có nhãn {TGT_LAB_COUNTS}, không nhãn {N_TGT_UNLAB}) | TARGET_HAS_LABELS = {TARGET_HAS_LABELS}")
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
savefig(fig, "inventory", f"Dữ liệu theo miền và lớp (chế độ `{DATA_MODE}`). Nhãn {cfg.target_name} "
        f"chỉ được dùng để đánh giá.")
''')

# =============================================================================
# 3. Vật lý & tiền xử lý
# =============================================================================
md(r'''
## 3. Vật lý lệch miền và tiền xử lý / Physics of the shift and pre-processing

**Giai đoạn 1 của sơ đồ.** Phương trình Korteweg hiệu chỉnh cho ống đàn hồi:
$c = c_0\,[1 + B D/(E e)]^{-1/2}$ ($c_0 \approx 1480$ m/s, $B \approx 2{,}2$ GPa, $D/e$ = SDR).
Các tần số cộng hưởng của đoạn ống $f_n = n c / 2L$ tỉ lệ thuận với $c$, nên toàn bộ phổ rò rỉ bị
**nhân với** $r = c_{HDPE}/c_{PVC}$. Trên trục tần số **log**, phép nhân này trở thành một phép
**tịnh tiến** cố định $\log r$. Một CNN 2D trên ảnh CWT có trục tần số log, cộng với gộp toàn cục,
vì vậy có sẵn thiên hướng (inductive bias) bất biến với lệch miền này. Đây là lý do mặc định dùng `tf_repr="cwt"`.

Tiền xử lý (mục 3.1 tài liệu): tái lấy mẫu 2 kHz → khử DC → lọc dải **15–350 Hz** (Butterworth, pha 0)
→ notch 50 Hz × các hài trong dải → chuẩn hoá đỉnh theo tệp. Ở chế độ `physics_sim`, bản ghi miền đích
còn được biến đổi theo mục 4 tài liệu: **dịch tần** (tái lấy mẫu với tốc độ giả định $f_s \cdot r$),
**suy hao nhớt đàn hồi** $e^{-\alpha f x}$ và **nhiễu nền** trong dải.
''')

code(r'''
# =========================================================
# 3.1 Vận tốc sóng theo Korteweg / Wave speed (Korteweg)
# =========================================================
def wave_speed(E, sdr, c0=None, K=None):
    c0 = cfg.c0 if c0 is None else c0
    K = cfg.K_water if K is None else K
    return c0 / np.sqrt(1.0 + K * sdr / E)

_rows = []
for mat, E in [("PVC", 2.8e9), ("PVC", 3.4e9), ("HDPE", 0.7e9), ("HDPE", 1.1e9)]:
    for sdr in (11, 17, 21, 26):
        _rows.append(dict(material=mat, E_GPa=E / 1e9, SDR=sdr, c_ms=round(float(wave_speed(E, sdr)), 1)))
korteweg = pd.DataFrame(_rows).pivot_table(index=["material", "E_GPa"], columns="SDR", values="c_ms")
display(Markdown("### Table 2. Korteweg wave speed c (m/s) vs. Young modulus and SDR = D/e"))
display(korteweg)
korteweg.to_csv(TAB_DIR / "table2_korteweg.csv")

C_PVC = float(wave_speed(cfg.E_pvc, cfg.sdr_pvc))
C_HDPE = float(wave_speed(cfg.E_hdpe, cfg.sdr_hdpe))
R_KORTEWEG = C_HDPE / C_PVC
print(f"c_PVC  (E={cfg.E_pvc/1e9:.1f} GPa, SDR {cfg.sdr_pvc:g}) = {C_PVC:6.1f} m/s")
print(f"c_HDPE (E={cfg.E_hdpe/1e9:.1f} GPa, SDR {cfg.sdr_hdpe:g}) = {C_HDPE:6.1f} m/s")
print(f"r = c_HDPE / c_PVC = {R_KORTEWEG:.3f}  -> dải rò PVC 80-280 Hz dự kiến dịch về "
      f"{80 * R_KORTEWEG:.0f}-{280 * R_KORTEWEG:.0f} Hz (chưa kể suy hao)")
print(f"Khoảng giả lập dùng cho PhysAug: r in {cfg.phys_warp_range}")
''')

code(r'''
# =========================================================
# 3.2 Nạp & tiền xử lý một tệp / Load and pre-process one file
# =========================================================
_FILT = {}
def band_sos(sr):
    key = ("band", int(sr), tuple(cfg.band_hz), int(cfg.filter_order))
    if key not in _FILT:
        lo, hi = cfg.band_hz
        _FILT[key] = sps.butter(cfg.filter_order, [lo / (sr / 2), hi / (sr / 2)], btype="band", output="sos")
    return _FILT[key]

def mains_notches(sr):
    key = ("notch", int(sr), cfg.mains_hz, cfg.notch_q, tuple(cfg.band_hz))
    if key not in _FILT:
        fm = [cfg.mains_hz * h for h in range(1, 64)
              if cfg.mains_hz * h <= min(cfg.band_hz[1] + 25.0, sr / 2 - 5.0)]
        _FILT[key] = [sps.iirnotch(f / (sr / 2), cfg.notch_q) for f in fm]
    return _FILT[key]

def viscoelastic_attenuation(x, sr, dist_m, alpha=None):
    # Bộ lọc suy hao nhớt đàn hồi e^{-alpha(f) x}, alpha(f) = alpha * f (tài liệu mục 4)
    alpha = cfg.phys_alpha if alpha is None else alpha
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-alpha * f * dist_m), len(x))

def _read_any(path):
    try:
        y, fs = sf.read(str(path), always_2d=True, dtype="float32")
        return y.mean(axis=1), int(fs)
    except Exception:
        y, fs = librosa.load(str(path), sr=None, mono=True)
        return y.astype(np.float32), int(fs)

LOAD_REPAIRED = []
def load_audio(path, sim=None, apply_band=True, sr=None):
    """(sim) dịch tần vật lý: coi mẫu gốc được lấy ở tốc độ fs*r rồi tái lấy mẫu -> mọi tần số nhân r
       -> tái lấy mẫu về cfg.target_sr (chống chồng phổ) -> khử DC
       -> (sim) suy hao e^{-alpha f x} + nhiễu hồng trong dải ở SNR định trước
       -> lọc dải 15-350 Hz (pha 0) -> notch 50 Hz x hài trong dải -> chuẩn hoá đỉnh theo tệp."""
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
        x = viscoelastic_attenuation(x, sr, sim["dist"])
        rng = np.random.default_rng(sim["seed"])
        nz = sps.sosfiltfilt(band_sos(sr), _pink_np(rng, len(x), sr, 1.0))
        ps = float(np.mean(sps.sosfiltfilt(band_sos(sr), x) ** 2))
        pn = float(np.mean(nz ** 2)) + 1e-20
        x = x + nz * np.sqrt(ps / (pn * 10 ** (sim["snr_db"] / 10.0)))
    if apply_band:
        x = sps.sosfiltfilt(band_sos(sr), x)
    if cfg.notch_mains:
        for b, a in mains_notches(sr):
            x = sps.filtfilt(b, a, x)
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    if m > 1e-12:
        x = x / m
    return x.astype(np.float32)

_t0 = time.time()
_r0 = files.iloc[0]
_probe = load_audio(_r0.path, sim=SIM_PARAMS.get(_r0.record_id))
print(f"Thử nạp 1 tệp: {len(_probe) / cfg.target_sr:.1f} s @ {cfg.target_sr} Hz trong {time.time() - _t0:.2f} s")
print(f"Notch tại: {[round(float(cfg.mains_hz * (i + 1))) for i in range(len(mains_notches(cfg.target_sr)))]} Hz")
''')

# =============================================================================
# 4. EDA - Giai đoạn 1
# =============================================================================
md(r'''
## 4. Giai đoạn 1 — đo lệch miền trên dữ liệu / Measuring the domain shift

Phổ Welch trung bình **mức bản ghi** của từng (miền, lớp), tần số trọng tâm trong dải 15–350 Hz, và
tỉ số dịch tần đo được $r_{emp}$. $r_{emp}$ là tỉ số **trung vị** tần số trọng tâm của hai miền, tính trên
phân phối **biên**, không dùng nhãn, nên vẫn hợp lệ khi HDPE không có nhãn. Giá trị này được so với tỉ số
Korteweg $r$ và với khoảng `phys_warp_range` mà PhysAug dùng.

> Đường phổ theo lớp của miền đích chỉ để **mô tả**. Nhãn HDPE không đi vào bất kỳ bước học nào.
''')

code(r'''
# =========================================================
# 4.1 Phổ mức bản ghi theo miền × lớp + tần số trọng tâm
# =========================================================
NPERSEG = 1024
def record_spectrum(rid, max_files=4):
    Ps, f = [], None
    for p in files.path[files.record_id == rid].head(max_files):
        x = load_audio(p, sim=SIM_PARAMS.get(rid), apply_band=False)
        if len(x) < NPERSEG:
            continue
        f, P = sps.welch(x, fs=cfg.target_sr, nperseg=NPERSEG, noverlap=NPERSEG // 2)
        Ps.append(P)
    return (f, np.mean(Ps, axis=0)) if Ps else (None, None)

_rng = np.random.default_rng(cfg.seed)
eda_recs = []
for _, g in records.groupby(["domain", "label"]):
    r = g.record_id.values
    eda_recs += list(_rng.choice(r, cfg.eda_max_records, replace=False) if len(r) > cfg.eda_max_records else r)

SPEC, F_W = {}, None
_t0 = time.time()
for rid in eda_recs:
    f, P = record_spectrum(rid)
    if P is not None:
        SPEC[rid], F_W = P, f
print(f"Phổ của {len(SPEC)} bản ghi trong {time.time() - _t0:.1f} s")

_lo, _hi = cfg.band_hz
INB = (F_W >= _lo) & (F_W <= _hi)
SPEC_DB = {}
for rid, P in SPEC.items():
    d = 10 * np.log10(P + 1e-20)
    SPEC_DB[rid] = d - d[INB].mean()          # chỉ giữ HÌNH DẠNG phổ (bất biến với mức to nhỏ)

def _centroid(P):
    return float((F_W[INB] * P[INB]).sum() / (P[INB].sum() + 1e-30))

spec_df = records[records.record_id.isin(list(SPEC))].copy()
spec_df["centroid_hz"] = spec_df.record_id.map(lambda r: _centroid(SPEC[r]))
spec_df["peak_hz"] = spec_df.record_id.map(lambda r: float(F_W[INB][np.argmax(SPEC[r][INB])]))
def _bandfrac(P, a, b):
    m = (F_W >= a) & (F_W <= b)
    return float(P[m].sum() / (P[INB].sum() + 1e-30))
spec_df["frac_80_280"] = spec_df.record_id.map(lambda r: _bandfrac(SPEC[r], 80, 280))
spec_df["frac_20_120"] = spec_df.record_id.map(lambda r: _bandfrac(SPEC[r], 20, 120))

_cen = spec_df.groupby("domain").centroid_hz.median()
R_EMP = float(_cen.get(1, np.nan) / _cen.get(0, np.nan))
tab_shift = (spec_df.groupby(["dom_name", "cls"])
             .agg(n=("record_id", "size"), centroid_hz=("centroid_hz", "median"), peak_hz=("peak_hz", "median"),
                  frac_80_280=("frac_80_280", "median"), frac_20_120=("frac_20_120", "median")).round(3))
display(Markdown("### Table 3. Spectral descriptors per domain × class (record-level medians, band 15–350 Hz)"))
display(tab_shift)
tab_shift.to_csv(TAB_DIR / "table3_spectral_shift.csv")
print(f"Tần số trọng tâm (trung vị, biên): {cfg.source_name} {_cen.get(0, np.nan):.1f} Hz | "
      f"{cfg.target_name} {_cen.get(1, np.nan):.1f} Hz")
print(f"r_emp = {R_EMP:.3f}  vs  r_Korteweg = {R_KORTEWEG:.3f}  vs  phys_warp_range = {cfg.phys_warp_range}")
if np.isfinite(R_EMP) and not (0.85 * cfg.phys_warp_range[0] <= R_EMP <= 1.15 * cfg.phys_warp_range[1]):
    print("  [lưu ý] r_emp nằm ngoài khoảng PhysAug -> cân nhắc chỉnh cfg.phys_warp_range quanh r_emp.")
''')

code(r'''
# =========================================================
# 4.2 Hình: lệch miền vật lý (Giai đoạn 1 của sơ đồ)
# =========================================================
fig, ax = plt.subplots(1, 3, figsize=(16, 4.1), gridspec_kw=dict(width_ratios=[1.7, 1, 1]))
fz = (F_W >= 5) & (F_W <= min(600, cfg.target_sr / 2))
for key, st in GROUP_STYLE.items():
    rids = [r for r in SPEC if REC_DOM[r] == key[0] and REC_LABEL[r] == key[1]]
    if not rids:
        continue
    M = np.stack([SPEC_DB[r] for r in rids])
    q1, med, q3 = np.percentile(M, [25, 50, 75], axis=0)
    ax[0].plot(F_W[fz], med[fz], color=st["color"], lw=1.6, ls="-" if key[0] == 0 else "--",
               label=f"{st['label']} (n={len(rids)})")
    ax[0].fill_between(F_W[fz], q1[fz], q3[fz], color=st["color"], alpha=.10)
ax[0].axvspan(80, 280, color=PAL["src"], alpha=.06, lw=0)
ax[0].axvspan(20, 120, color=PAL["tgt"], alpha=.06, lw=0)
for v in cfg.band_hz:
    ax[0].axvline(v, color="k", lw=.7, ls=":")
ax[0].set_xscale("log"); ax[0].set_xlabel("Tần số (Hz, log)"); ax[0].set_ylabel("PSD chuẩn hoá (dB)")
ax[0].set_title("(a) Phổ trung vị mức bản ghi (vùng tô: 80-280 Hz PVC, 20-120 Hz HDPE)")
ax[0].legend(fontsize=8, loc="lower left")

data = [spec_df.centroid_hz[spec_df.domain == d].values for d in (0, 1)]
ax[1].boxplot(data, widths=.5, showfliers=False)
for d in (0, 1):
    sub = spec_df[spec_df.domain == d]
    xs = d + 1 + np.random.default_rng(d).uniform(-.12, .12, len(sub))
    ax[1].scatter(xs, sub.centroid_hz, s=14, c=[GROUP_STYLE[(d, int(l))]["color"] for l in sub.label],
                  marker=GROUP_STYLE[(d, 0)]["marker"], zorder=3)
ax[1].set_xticks([1, 2]); ax[1].set_xticklabels([cfg.source_name, cfg.target_name])
ax[1].set_ylabel("Tần số trọng tâm 15-350 Hz (Hz)"); ax[1].set_title("(b) Phân phối biên (không dùng nhãn)")

vals = [R_EMP, R_KORTEWEG]
ax[2].bar(["r_emp\n(đo được)", "r Korteweg\n(lý thuyết)"], vals, color=[PAL["aux"], PAL["acc"]])
ax[2].axhspan(*cfg.phys_warp_range, color=PAL["warn"], alpha=.25, label="phys_warp_range")
for i, v in enumerate(vals):
    if np.isfinite(v):
        ax[2].text(i, v + .02, f"{v:.2f}", ha="center")
ax[2].set_ylim(0, 1.15); ax[2].axhline(1, color="k", lw=.6, ls=":")
ax[2].set_ylabel(f"tỉ số tần số {cfg.target_name}/{cfg.source_name}")
ax[2].set_title("(c) Tỉ số dịch tần"); ax[2].legend(fontsize=8)
savefig(fig, "domain_shift_physics",
        f"Giai đoạn 1: lệch miền vật lý. (a) Phổ Welch trung vị theo miền × lớp (nét đứt: {cfg.target_name}); "
        f"(b) tần số trọng tâm mức bản ghi; (c) tỉ số dịch tần đo được so với Korteweg.")
''')

# =============================================================================
# 5. Frame + VMD
# =============================================================================
md(r'''
## 5. Cắt frame và khử nhiễu VMD / Framing and VMD denoising

Mỗi tệp được cắt thành frame `frame_sec` = 2 s, bước 1 s. Mỗi frame mang mã bản ghi mẹ để
mọi phép chia và đánh giá sau này đều tính theo bản ghi.

**VMD** (Dragomiretskiy & Zosso, 2014; nút 3 của sơ đồ) phân rã mỗi frame thành $K$ mode băng hẹp
$u_k$ với tần số tâm $\omega_k$, bằng ADMM trong miền tần số:

$$\hat u_k^{n+1}(\omega)=\frac{\hat f(\omega)-\sum_{i\neq k}\hat u_i(\omega)+\hat\lambda(\omega)/2}{1+2\alpha(\omega-\omega_k)^2},\qquad
\omega_k^{n+1}=\frac{\int_0^\infty\omega|\hat u_k(\omega)|^2d\omega}{\int_0^\infty|\hat u_k(\omega)|^2d\omega}$$

Tín hiệu khử nhiễu là tổng các mode có tâm **trong dải 15–350 Hz** và **không trùng hài lưới điện**
(thành phần tonal). Với $\tau=0$, VMD hoạt động như một dãy bộ lọc Wiener thích nghi đặt quanh các đỉnh phổ.
VMD được hiện thực **theo lô trên GPU** (torch), chạy một lần rồi lưu cache. Phân bố $\omega_k$ theo miền
là một thước đo lệch miền **không cần nhãn**.
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
    h.update(json.dumps(dict(sr=cfg.target_sr, band=list(cfg.band_hz), order=cfg.filter_order,
                             notch=cfg.notch_mains, mh=cfg.mains_hz, q=cfg.notch_q, fl=FRAME_LEN, hl=HOP_LEN,
                             alpha=cfg.phys_alpha), sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:12]

def build_frames():
    chunks, metas = [], []
    for r in files.itertuples():
        x = load_audio(r.path, sim=SIM_PARAMS.get(r.record_id))
        if len(x) < FRAME_LEN:
            x = np.pad(x, (0, FRAME_LEN - len(x)))
        for s in range(0, len(x) - FRAME_LEN + 1, HOP_LEN):
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
    xm = torch.cat([x[:, :h].flip(1), x, x[:, N - h:].flip(1)], dim=1)     # phản xạ biên
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
        for k in range(K):                                   # Gauss-Seidel theo mode
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
    uh[:, :, torch.arange(T // 2, 0, -1, device=dev)] = u[:, :, T // 2:].conj()   # đối xứng Hermite
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

def vmd_denoise_frames(X, device):
    n = len(X)
    out = np.empty_like(X); OM = np.zeros((n, cfg.vmd_K), np.float32)
    KEEP = np.zeros((n, cfg.vmd_K), bool); EF = np.zeros(n, np.float32); its = []
    for s in range(0, n, cfg.vmd_batch):
        xb = torch.from_numpy(np.ascontiguousarray(X[s:s + cfg.vmd_batch])).to(device)
        modes, om, it = vmd_torch(xb, cfg.vmd_K, cfg.vmd_alpha, cfg.vmd_tau, cfg.vmd_iters, cfg.vmd_tol)
        fhz = om * cfg.target_sr
        keep = vmd_keep_mask(fhz)
        rec = (modes * keep[:, :, None].float()).sum(1)
        rec = torch.where(keep.any(1)[:, None], rec, xb)     # không mode nào hợp lệ -> giữ nguyên
        e = (rec ** 2).sum(1) / (xb ** 2).sum(1).clamp_min(1e-12)
        sl = slice(s, s + len(xb))
        out[sl] = rec.cpu().numpy(); OM[sl] = fhz.cpu().numpy(); KEEP[sl] = keep.cpu().numpy()
        EF[sl] = e.cpu().numpy(); its.append(it)
    return out, OM, KEEP, EF, its

# ---- kiểm thử VMD: tổng 2 sin + nhiễu -> 2 mode đúng tần số ----
_sr = cfg.target_sr; _tt = np.arange(FRAME_LEN) / _sr
_sig = np.sin(2 * np.pi * 40 * _tt) + .7 * np.sin(2 * np.pi * 180 * _tt)
_m, _o, _ = vmd_torch(torch.from_numpy(_sig[None].astype(np.float32)), K=2, alpha=2000, iters=300)
_fo = sorted((_o[0] * _sr).tolist())
assert abs(_fo[0] - 40) < 3 and abs(_fo[1] - 180) < 5, f"VMD tìm sai tần số tâm: {_fo}"
print(f"Kiểm thử VMD: tâm mode {[round(v, 1) for v in _fo]} Hz (kỳ vọng 40, 180)")

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
              f"(số vòng lặp trung bình {np.mean(_its):.0f}/{cfg.vmd_iters})")
    for k in range(cfg.vmd_K):
        frames[f"vmd_f{k}"] = VMD_OM[:, k]
    frames["vmd_energy_kept"] = VMD_EF
else:
    X_ALL = X_PRE
    VMD_OM = VMD_KEEP = VMD_EF = None
    print("cfg.use_vmd=False -> dùng frame sau lọc dải.")
X_ALL = np.ascontiguousarray(X_ALL, dtype=np.float32)
assert np.isfinite(X_ALL).all()

# ---- Mảng toàn cục dùng cho mọi bước sau ----
Y_ALL = frames.label.values.astype(np.float32)          # -1 = không nhãn
DOM_ALL = frames.domain.values.astype(np.int64)
REC_ALL = frames.record_id.values.astype(str)
REC2IDX = {r: np.asarray(v, dtype=np.int64) for r, v in frames.groupby("record_id").indices.items()}
# Nhãn mà thuật toán UDA được phép thấy: CHỈ miền nguồn (miền đích = NaN)
Y_SRC_VISIBLE = np.where(DOM_ALL == 0, Y_ALL, np.nan).astype(np.float32)
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
            f_, P_ = sps.welch(modes[0, k].numpy(), fs=cfg.target_sr, nperseg=512)
            a1.semilogy(f_, P_ + 1e-14, lw=1.1, ls="-" if keep[k] else ":",
                        label=f"mode {k}: {fhz[k]:.0f} Hz {'(giữ)' if keep[k] else '(bỏ)'}")
        a1.set_xlim(0, 500); a1.set_xlabel("Hz"); a1.set_title("PSD từng mode VMD"); a1.legend(fontsize=7)
    a2 = fig.add_subplot(gs[:, 2])
    bins = np.linspace(0, min(600, cfg.target_sr / 2), 61)
    for d in (0, 1):
        a2.hist(VMD_OM[DOM_ALL == d].ravel(), bins=bins, density=True, histtype="step", lw=1.6,
                color=PAL["tgt"] if d else PAL["src"], label=DOM_NAMES[d])
    for v in cfg.band_hz:
        a2.axvline(v, color="k", ls=":", lw=.7)
    a2.set_xlabel("tần số tâm mode (Hz)"); a2.set_ylabel("mật độ")
    a2.set_title("(c) Phân bố tần số tâm VMD theo miền\n(không dùng nhãn)"); a2.legend()
    savefig(fig, "vmd", "VMD theo lô: (a, b) frame đại diện của mỗi miền trước/sau VMD và PSD từng mode "
            "(nét chấm = mode bị loại vì ngoài dải hoặc trùng hài 50 Hz); (c) phân bố tần số tâm mode theo miền.")
''')

# =============================================================================
# 6. Kiến trúc
# =============================================================================
md(r'''
## 6. Giai đoạn 2–3 — kiến trúc 1D/2D-CNN + GRL + Deep CORAL / Architectures

```
            x (sóng 2 s @ 2 kHz, đã lọc dải + VMD)
                        │
        ┌───────────────┴────────────────┐
   [CNN2D]  CWT Morlet 64 tần số log 15-350 Hz      [CNN1D]  chuẩn hoá theo mẫu
        → dB → SpecAugment → chuẩn hoá frame              → Conv1d k=65 (bộ lọc dải học được)
        → CMVN THEO MIỀN (z-score từng dải)                → 3 khối Conv1d + CBAM-1D
        → 4 khối Conv2d + CBAM                             │
        └───────────────┬────────────────┘
               gộp TB + cực đại → bottleneck → f ∈ R^d   (= G_f, dùng chung cho PVC & HDPE)
                 ┌──────┴──────────────┬────────────────────────┐
             G_y: FC → leak/no-leak    GRL(−λ_p) → G_d: PVC/HDPE   Deep CORAL(f_S, f_T)
             (chỉ PVC có nhãn)          (BCE miền)                  (1/4d²·‖C_S − C_T‖²_F)
```

- **GRL**: lan truyền xuôi là ánh xạ đồng nhất; lan truyền ngược nhân gradient với $-\lambda_p$.
  $G_d$ cực tiểu $\mathcal{L}_d$, còn $G_f$ nhận $-\lambda_p\,\partial\mathcal{L}_d/\partial f$, tức là học
  cách "đánh lừa" $G_d$.
- **Deep CORAL** căn chỉnh thống kê **bậc 2**: bất biến với dịch trung bình, nên bổ sung cho DANN
  (đối kháng, căn chỉnh toàn bộ phân phối nhưng kém ổn định). Hiệp phương sai luôn tính ở fp32.
- **CMVN theo miền** (mục 3.1 tài liệu): trừ phổ log trung bình và chia độ lệch chuẩn **của từng dải** ước lượng
  trên frame của chính miền đó (PVC: tập huấn luyện; HDPE: phần thích nghi không nhãn). Bước này loại bỏ hàm
  truyền tĩnh $H(\omega)$ của ống/đất/cảm biến. Không chuẩn hoá theo từng frame vì rò rỉ là nhiễu
  **dừng**: trừ trung bình theo thời gian của chính frame sẽ xoá luôn dấu hiệu rò.
- **AdaBN**: ước lượng lại thống kê BatchNorm (và CMVN) trên HDPE không nhãn.
''')

code(r'''
# =========================================================
# 6.1 Front-end thời gian-tần số + SpecAugment + CMVN theo miền
# =========================================================
def frame_norm(s):
    # Trừ mức trung bình (bất biến độ to) và chia độ lệch chuẩn TOÀN ẢNH -> giữ hình dạng phổ
    s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=-100.0)
    mu = s.mean(dim=(2, 3), keepdim=True)
    sd = s.std(dim=(2, 3), keepdim=True).clamp_min(1e-3)
    return (s - mu) / sd

class CWTFrontend(nn.Module):
    """CWT Morlet phức, tần số chia đều theo LOG trong cfg.band_hz, tính bằng conv1d (GPU).
    Mỗi nhân chuẩn hoá để có độ lợi 1 tại tần số tâm. Ra: dB [B, 1, F, T]."""
    def __init__(self, sr=None, band=None, n_freqs=None, hop=None, w0=None):
        super().__init__()
        sr = int(sr or cfg.target_sr); lo, hi = band or cfg.band_hz
        n = int(n_freqs or cfg.n_freqs); w0 = float(w0 or cfg.cwt_w0)
        self.hop = int(hop or cfg.tf_hop)
        self.freqs = np.geomspace(lo, hi, n)
        sig = w0 / (2 * np.pi * self.freqs)                    # độ rộng Gauss (s)
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
            db = torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - 80.0)      # top_db = 80
            return db.unsqueeze(1)

class LogMelFrontend(nn.Module):
    # log-Mel dải thấp bằng torch.stft + bộ lọc Mel của librosa (không cần torchaudio)
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
    # Che ngẫu nhiên dải tần / đoạn thời gian, MỖI MẪU một mặt nạ, chỉ khi training,
    # áp TRƯỚC chuẩn hoá với giá trị che = mức nhỏ nhất của chính mẫu ("im lặng").
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
    """z-score THEO DẢI, thống kê RIÊNG cho từng miền (0 = PVC, 1 = HDPE).
    Loại hàm truyền tĩnh H(w) của ống/đất/cảm biến mà vẫn giữ khác biệt leak/no-leak trong miền."""
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
    # Convolutional Block Attention Module (Woo et al., 2018): kênh -> không gian
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
    """G_f 2D: CWT (hoặc log-Mel) dải 15-350 Hz -> CMVN theo miền -> 4 khối Conv2d + CBAM.
    Trục tần số log + gộp toàn cục -> phép nhân tần số r (PVC -> HDPE) thành phép tịnh tiến."""
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
    def tf_raw(self, x):                    # ảnh đã chuẩn hoá frame, CHƯA CMVN (để ước lượng thống kê miền)
        return frame_norm(self.front(x))
    def tf(self, x, dom=None):
        return self.cmvn(frame_norm(self.specaug(self.front(x))), dom)
    def forward(self, x, dom=None):
        return self.bottleneck(gpool(self.mid(self.stem(self.tf(x, dom)))))
    def ft_groups(self):
        return dict(early=list(self.stem.parameters()), middle=list(self.mid.parameters()),
                    head=list(self.bottleneck.parameters()))

class CNN1D(nn.Module):
    """G_f 1D: sóng thô (chuẩn hoá theo mẫu) -> Conv1d k=65 (dãy bộ lọc dải HỌC ĐƯỢC, ~32 ms)
    -> 3 khối Conv1d + CBAM-1D. Thích nghi miền nhờ BatchNorm/AdaBN + GRL/CORAL (không có CMVN)."""
    kind = "1d"
    def __init__(self, emb_dim=None, ch=(32, 64, 128, 128), p_drop=0.1):
        super().__init__()
        emb_dim = int(emb_dim or cfg.emb_dim)
        self.stem = nn.Sequential(nn.Conv1d(1, ch[0], 65, stride=2, padding=32, bias=False),
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
# 6.3 GRL, G_y, G_d, mạng DANN đầy đủ + hàm mất mát CORAL / MMD
# =========================================================
class GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer (Ganin & Lempitsky 2015): xuôi = đồng nhất, ngược = -lambda * grad."""
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
        self.lambd = 0.0                     # được đặt lại mỗi bước theo lịch lambda_p
    def forward(self, x):
        return GradReverse.apply(x, self.lambd)

class LabelPredictor(nn.Module):           # G_y
    def __init__(self, d, hidden=64, p=None):
        super().__init__()
        p = cfg.dropout if p is None else p
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(inplace=True), nn.Dropout(p), nn.Linear(hidden, 1))
    def forward(self, f):
        return self.net(f).squeeze(-1)

class DomainDiscriminator(nn.Module):      # G_d
    def __init__(self, d, hidden=None, p=None):
        super().__init__()
        h = int(hidden or cfg.disc_hidden); p = cfg.dropout if p is None else p
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(p),
                                 nn.Linear(h, h), nn.ReLU(inplace=True), nn.Dropout(p), nn.Linear(h, 1))
    def forward(self, f):
        return self.net(f).squeeze(-1)

class DANNNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone_name = backbone
        self.G_f = BACKBONES[backbone]()
        d = self.G_f.out_dim
        self.G_y, self.grl, self.G_d = LabelPredictor(d), GRL(), DomainDiscriminator(d)
    def forward(self, x, dom=None):            # suy luận: logit rò rỉ
        return self.G_y(self.G_f(x, dom))
    def forward_all(self, x, dom):
        f = self.G_f(x, dom)
        return self.G_y(f), self.G_d(self.grl(f)), f

def grl_schedule(p, gamma=None):
    gamma = cfg.grl_gamma if gamma is None else gamma
    return 2.0 / (1.0 + math.exp(-gamma * float(p))) - 1.0

def coral_loss(fs, ft):
    """Deep CORAL (Sun & Saenko 2016): 1/(4 d^2) ||C_S - C_T||_F^2, luôn tính ở fp32."""
    with amp_autocast(fs.device, False):
        fs, ft = fs.float(), ft.float()
        d = fs.shape[1]
        def _cov(f):
            fc = f - f.mean(0, keepdim=True)
            return fc.t() @ fc / max(f.shape[0] - 1, 1)
        return ((_cov(fs) - _cov(ft)) ** 2).sum() / (4.0 * d * d)

def mmd_rbf(fs, ft, n_kernels=5, mul=2.0):
    """MMD^2 đa nhân Gauss (DAN, Long et al. 2015), băng thông theo khoảng cách trung bình."""
    with amp_autocast(fs.device, False):
        x = torch.cat([fs, ft], 0).float(); n, ns = x.shape[0], fs.shape[0]
        d2 = torch.cdist(x, x).pow(2)
        bw = d2.detach().sum() / max(n * n - n, 1)
        K = sum(torch.exp(-d2 / (bw * mul ** (i - n_kernels // 2)).clamp_min(1e-8)) for i in range(n_kernels))
        return K[:ns, :ns].mean() + K[ns:, ns:].mean() - 2 * K[:ns, ns:].mean()
''')

code(r'''
# =========================================================
# 6.4 Kiểm thử đơn vị + độ phức tạp mô hình
# =========================================================
# (1) GRL: xuôi = đồng nhất, ngược = -lambda * grad
_x = torch.randn(5, 3, requires_grad=True)
_y = GradReverse.apply(_x, 0.7)
assert torch.equal(_y.detach(), _x.detach())
_y.sum().backward()
assert torch.allclose(_x.grad, torch.full_like(_x, -0.7)), "GRL sai dấu/độ lớn"
# (2) Qua GRL, G_f nhận gradient NGƯỢC DẤU với gradient mà G_d nhận -> minimax đúng nghĩa
_net = DomainDiscriminator(8).eval(); _f = torch.randn(16, 8, requires_grad=True); _dl = torch.randint(0, 2, (16,)).float()
_g1, = torch.autograd.grad(F.binary_cross_entropy_with_logits(_net(_f), _dl), _f)
_g2, = torch.autograd.grad(F.binary_cross_entropy_with_logits(_net(GradReverse.apply(_f, 1.0)), _dl), _f)
assert torch.allclose(_g1, -_g2, atol=1e-6)
# (3) CORAL: = 0 khi trùng phân phối, khớp công thức numpy, bất biến với dịch trung bình
_a = torch.randn(64, 16, dtype=torch.float64).float(); _b = _a * 2.0 + 0.3 * torch.randn(64, 16)
_ref = ((np.cov(_a.numpy(), rowvar=False) - np.cov(_b.numpy(), rowvar=False)) ** 2).sum() / (4 * 16 ** 2)
assert coral_loss(_a, _a).item() < 1e-10
assert abs(coral_loss(_a, _b).item() - _ref) < 1e-4 * max(1.0, _ref), (coral_loss(_a, _b).item(), _ref)
assert coral_loss(_a, _a + 5.0).item() < 1e-8, "CORAL phải bất biến với dịch trung bình (chỉ bậc 2)"
# (4) MMD: ~0 cho cùng phân phối, lớn hơn rõ khi lệch
_m0 = mmd_rbf(torch.randn(200, 8), torch.randn(200, 8)).item()
_m1 = mmd_rbf(torch.randn(200, 8), torch.randn(200, 8) + 1.5).item()
assert _m1 > 5 * max(_m0, 1e-4), (_m0, _m1)
# (5) Lịch lambda_p
assert abs(grl_schedule(0.0)) < 1e-9 and grl_schedule(1.0) > 0.99
print(f"GRL / CORAL / MMD / lịch lambda: OK  (CORAL ví dụ = {coral_loss(_a, _b).item():.4f}, MMD {_m0:.4f} vs {_m1:.4f})")

rows = []
with torch.no_grad():
    _xb = torch.from_numpy(X_ALL[:4].copy()); _dom = torch.tensor([0, 0, 1, 1])
    for bb in cfg.backbones:
        net = DANNNet(bb).eval()
        yl, dl, f = net.forward_all(_xb, _dom)
        assert yl.shape == (4,) and dl.shape == (4,) and f.shape == (4, cfg.emb_dim)
        npar = lambda m: sum(p.numel() for p in m.parameters())
        info = dict(backbone=bb, G_f=npar(net.G_f), G_y=npar(net.G_y), G_d=npar(net.G_d))
        if bb == "CNN2D":
            info["tf_shape"] = str(tuple(net.G_f.tf_raw(_xb).shape))
        rows.append(info)
param_table = pd.DataFrame(rows)
param_table["total_M"] = ((param_table.G_f + param_table.G_y + param_table.G_d) / 1e6).round(3)
display(Markdown("### Table 4. Model complexity (parameters)"))
display(param_table)
param_table.to_csv(TAB_DIR / "table4_model_complexity.csv", index=False)
''')

code(r'''
# =========================================================
# 6.5 Hình: ảnh CWT theo miền × lớp (Giai đoạn 2) + lịch lambda_p
# =========================================================
fe = make_tf_frontend()
ex = {}
for d in (0, 1):
    for lab in ((1, 0) if (d == 0 or TARGET_HAS_LABELS) else (-1,)):
        cand = np.where((DOM_ALL == d) & (Y_ALL == lab))[0]
        if len(cand):
            ex[(d, lab)] = int(cand[np.argsort(frames.rms.values[cand])[len(cand) // 2]])
fig, axes = plt.subplots(1, len(ex) + 1, figsize=(3.6 * (len(ex) + 1), 3.4))
yt_hz = [f for f in (20, 50, 100, 200, 350) if cfg.band_hz[0] <= f <= cfg.band_hz[1]]
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
savefig(fig, "tf_inputs", f"Giai đoạn 2: ảnh {cfg.tf_repr.upper()} (dB, trục tần số log {cfg.band_hz[0]:g}-"
        f"{cfg.band_hz[1]:g} Hz) của frame đại diện mỗi (miền, lớp) và lịch λ_p = 2/(1+e^(-γp)) - 1.")
''')

# =============================================================================
# 7. Chia dữ liệu
# =============================================================================
md(r'''
## 7. Giao thức chia dữ liệu chống rò rỉ / Leakage-free split protocol

- **Nguồn (PVC):** bản ghi chia thành *train* và *val* (phân tầng theo lớp). Val dùng để theo dõi và để báo
  hiệu năng trên chính PVC (DA không được phá hiệu năng miền nguồn).
- **Đích (HDPE):** bản ghi **có nhãn** chia `n_target_folds` fold phân tầng. Ở fold *k*: *kiểm thử* = fold
  *k*; *thích nghi* = các fold còn lại **cộng mọi bản ghi không nhãn**, đưa vào huấn luyện **không kèm nhãn**.
  *Few-shot* = khoảng 10 % bản ghi của phần thích nghi (≥ 1 bản ghi/lớp), có nhãn, chỉ cho Few-shot FT.
  *Oracle* dùng nhãn của toàn bộ phần thích nghi.
- Gộp dự đoán kiểm thử qua các fold: mỗi bản ghi HDPE có nhãn được dự đoán **đúng một lần** bởi một mô hình
  chưa từng thấy nó, kể cả không nhãn.
''')

code(r'''
# =========================================================
# 7.1 Dựng phân hoạch + assertion cứng
# =========================================================
def idx_of(recs):
    recs = sorted(recs)
    return np.sort(np.concatenate([REC2IDX[r] for r in recs])) if recs else np.zeros(0, np.int64)

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
        n_f = int(min(cfg.n_target_folds, tl.label.value_counts().min()))
        skf = StratifiedKFold(n_splits=n_f, shuffle=True, random_state=seed)
        for k, (a, b) in enumerate(skf.split(tl.record_id, tl.label)):
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
    assert set(Y_ALL[sp["src_train_idx"]]) == {0.0, 1.0}, "PVC train phải có cả hai lớp"
    tested = []
    for fd in sp["folds"]:
        assert not (fd["test_recs"] & fd["adapt_recs"]), "RÒ RỈ: bản ghi HDPE vừa thích nghi vừa kiểm thử"
        assert fd["fewshot_recs"] <= fd["adapt_lab_recs"] <= fd["adapt_recs"]
        assert all(REC_DOM[r] == 1 for r in fd["test_recs"] | fd["adapt_recs"])
        assert all(REC_LABEL[r] >= 0 for r in fd["test_recs"])
        assert np.intersect1d(fd["t_test_idx"], fd["t_adapt_idx"]).size == 0
        assert set(Y_ALL[fd["t_test_idx"]]) == {0.0, 1.0}, "fold kiểm thử phải có cả hai lớp"
        tested += list(fd["test_recs"])
    if sp["folds"]:
        want = sorted(r for r in REC_LABEL if REC_DOM[r] == 1 and REC_LABEL[r] >= 0)
        assert sorted(tested) == want, "mỗi bản ghi HDPE có nhãn phải được kiểm thử ĐÚNG MỘT lần"

SPLITS = {s: build_splits(s) for s in cfg.seeds}
for s, sp in SPLITS.items():
    check_splits(sp)
EVAL_ON = bool(TARGET_HAS_LABELS and all(len(sp["folds"]) >= 2 for sp in SPLITS.values()))
sp0 = SPLITS[cfg.seeds[0]]
print(f"seed {cfg.seeds[0]}: PVC train {len(sp0['src_train_recs'])} bản ghi ({len(sp0['src_train_idx'])} frame) | "
      f"PVC val {len(sp0['src_val_recs'])} ({len(sp0['src_val_idx'])} frame)")
for fd in sp0["folds"]:
    print(f"  fold {fd['fold']}: HDPE kiểm thử {len(fd['test_recs'])} bản ghi ({len(fd['t_test_idx'])} frame) | "
          f"thích nghi {len(fd['adapt_recs'])} ({len(fd['t_adapt_idx'])} frame, không nhãn) | "
          f"few-shot {len(fd['fewshot_recs'])} bản ghi "
          f"({100 * len(fd['t_fewshot_idx']) / max(1, (DOM_ALL == 1).sum()):.1f}% frame HDPE)")
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
fig, ax = plt.subplots(figsize=(1.3 * len(cols) + 3.5, min(9, 1.2 + .09 * len(recs_sorted))))
ax.imshow(M, aspect="auto", cmap=cmap, vmin=-.5, vmax=5.5, interpolation="nearest")
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols); ax.set_ylabel("bản ghi (theo miền, lớp)"); ax.grid(False)
nd0 = int((records.domain == 0).sum()); ax.axhline(nd0 - .5, color="k", lw=1)
ax.legend(handles=[Patch(color=cmap(i), label=n) for i, n in enumerate(names) if i], loc="center left",
          bbox_to_anchor=(1.01, .5), fontsize=8)
savefig(fig, "split_map", "Vai trò của từng bản ghi trong mỗi fold. Nhãn HDPE chỉ dùng ở ô 'kiểm thử' "
        "(đánh giá) và 'few-shot' (riêng phương pháp Few-shot FT).")
''')

# =============================================================================
# 8. Bộ máy huấn luyện
# =============================================================================
md(r'''
## 8. Bộ máy huấn luyện / Training engine

Mỗi bước huấn luyện UDA làm các việc sau:

1. lấy một lô **PVC có nhãn** $(x^S, y^S)$ và một lô **HDPE không nhãn** $x^T$ cùng kích thước;
2. tăng cường trên GPU: dịch vòng, nhiễu trong dải SNR 10–30 dB, và với `dann_coral_phys` thêm **PhysAug**
   cho lô PVC (dịch tần $r\sim\mathcal U(0{,}5;0{,}65)$, suy hao $e^{-\alpha f x}$, nhiễu nền SNR −5…+5 dB);
3. một lần lan truyền chung qua $G_f$ (BatchNorm thấy cả hai miền), sau đó tính
   $\mathcal L=\mathcal L_y(f^S)+\mathcal L_d\big(G_d(\mathrm{GRL}_{\lambda_p}(f))\big)+\beta_p\,\mathcal L_{CORAL}(f^S,f^T)$.

Các quy tắc cố định để đảm bảo đánh giá trung thực:

- Mô hình lấy ở **epoch cuối** và ngưỡng cố định 0,5. AUC trên HDPE kiểm thử được *ghi lại mỗi epoch* chỉ để
  vẽ đường học (cột `tgt_auc_monitor`) và **không bao giờ** dùng để chọn.
- Mọi phương pháp của cùng (seed, kiến trúc, fold) dùng **cùng khởi tạo** và cùng chuỗi lô PVC, nên so sánh
  là **ghép cặp** (paired).
- Có thể chạy song song: mỗi GPU nhận một job từ hàng đợi.
''')

code(r'''
# =========================================================
# 8.1 Tiện ích: dữ liệu trên GPU, tăng cường, đánh giá, CMVN, AdaBN
# =========================================================
_DEV_DATA, _DEV_LOCK, _INIT_LOCK = {}, threading.Lock(), threading.Lock()

def use_amp(device):
    return bool(cfg.amp and torch.device(device).type == "cuda")

def dev_data(device):
    # Toàn bộ frame thường trú trên GPU (nhỏ: 2 s @ 2 kHz = 16 KB/frame)
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

def band_noise(n, L, g, device, pink=False):
    W = torch.fft.rfft(torch.randn(n, L, generator=g, device=device), dim=1)
    f = torch.fft.rfftfreq(L, 1.0 / cfg.target_sr).to(device)
    shape = ((f >= cfg.band_hz[0]) & (f <= cfg.band_hz[1])).float()
    if pink:
        shape = shape / torch.sqrt(f.clamp_min(1.0))
    return torch.fft.irfft(W * shape, n=L, dim=1)

def add_noise_snr(x, snr_db, sel, noise):
    ps = x.pow(2).mean(1, keepdim=True)
    pn = noise.pow(2).mean(1, keepdim=True).clamp_min(1e-12)
    k = torch.sqrt(ps / (pn * 10 ** (snr_db[:, None] / 10.0)))
    return torch.where(sel[:, None], x + k * noise, x)

def phys_aug(x, g):
    """PhysAug (mục 4 tài liệu) trên GPU: biến PVC thành 'giả lập HDPE', giữ nguyên nhãn.
    (1) dịch tần r: y[n] = x(off + n*r) (nội suy tuyến tính) -> mọi tần số nhân r
    (2) suy hao nhớt đàn hồi e^{-alpha f x}, x ~ U(phys_dist_m)
    (3) nhiễu hồng trong dải ở SNR ~ U(phys_noise_snr_db) (thay cho nhiễu nền thực địa HDPE)."""
    B, L = x.shape; dev = x.device
    sel = torch.rand(B, generator=g, device=dev) < cfg.phys_aug_p
    n = int(sel.sum())
    if n == 0:
        return x
    xs = x[sel].float()
    r = _uniform(n, *cfg.phys_warp_range, g, dev)
    off = torch.rand(n, generator=g, device=dev) * (L - 1) * (1 - r)
    pos = off[:, None] + torch.arange(L, device=dev, dtype=torch.float32)[None, :] * r[:, None]
    i0 = pos.floor().long().clamp(0, L - 2); w = pos - i0.float()
    y = torch.gather(xs, 1, i0) * (1 - w) + torch.gather(xs, 1, i0 + 1) * w
    dist = _uniform(n, *cfg.phys_dist_m, g, dev)
    f = torch.fft.rfftfreq(L, 1.0 / cfg.target_sr).to(dev)
    y = torch.fft.irfft(torch.fft.rfft(y, dim=1) * torch.exp(-cfg.phys_alpha * f[None, :] * dist[:, None]), n=L, dim=1)
    nsel = torch.rand(n, generator=g, device=dev) < cfg.phys_noise_p
    y = add_noise_snr(y, _uniform(n, *cfg.phys_noise_snr_db, g, dev), nsel, band_noise(n, L, g, dev, pink=True))
    out = x.clone(); out[sel] = y.to(x.dtype)
    return out

def augment(x, g, phys_g=None):
    B, L = x.shape; dev = x.device
    if cfg.aug_shift_p > 0:
        do = (torch.rand(B, generator=g, device=dev) < cfg.aug_shift_p).long()
        sh = (torch.rand(B, generator=g, device=dev) * L).long() * do
        x = torch.gather(x, 1, (torch.arange(L, device=dev)[None, :] + sh[:, None]) % L)
    if phys_g is not None:
        x = phys_aug(x, phys_g)
    if cfg.aug_noise_p > 0:
        sel = torch.rand(B, generator=g, device=dev) < cfg.aug_noise_p
        x = add_noise_snr(x, _uniform(B, *cfg.aug_noise_snr_db, g, dev), sel, band_noise(B, L, g, dev))
    return x

# ---- chỉ số ----
def fast_auc(y, p):
    y = np.asarray(y) > 0.5; p = np.asarray(p, dtype=float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

def bce_np(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def rec_scores(idx, p):
    idx = np.asarray(idx)
    return (pd.DataFrame(dict(rec=REC_ALL[idx], y=Y_ALL[idx], p=np.asarray(p, float)))
            .groupby("rec").agg(y=("y", "first"), p=("p", "mean")))     # gộp p_mean (đăng ký trước)

def record_auc(idx, p):
    if len(idx) == 0:
        return np.nan
    r = rec_scores(idx, p)
    return fast_auc(r.y.values, r.p.values)

@torch.no_grad()
def predict(model, idx, device, dom=None, return_emb=False):
    """Xác suất rò rỉ cho các frame idx. dom=None -> dùng miền thật của từng frame."""
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
    """AdaBN (Li et al. 2016): ước lượng lại thống kê BatchNorm của G_f trên dữ liệu miền đích không nhãn."""
    bns = [m for m in model.G_f.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    idx = np.random.default_rng(seed).permutation(np.asarray(idx))
    if not bns or len(idx) < 2:
        return 0
    mom = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats(); m.momentum = None          # trung bình tích luỹ
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
    # Mở nhãn HDPE cho ĐÚNG các frame được phép (few-shot / oracle); mọi chỗ khác chỉ thấy Y_SRC_VISIBLE
    v = np.full(len(Y_ALL), np.nan, np.float32); v[np.asarray(idx)] = Y_ALL[np.asarray(idx)]
    return v

def pos_weight(y):
    pos = float(np.sum(y)); neg = float(len(y) - pos)
    return float(np.clip(neg / max(pos, 1.0), 0.1, 10.0))

def make_model(backbone, seed, device):
    with _INIT_LOCK:                      # khởi tạo trọng số tất định
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
        p = predict(model, ti, device)        # CHỈ để vẽ đường học - KHÔNG dùng để chọn
        out["tgt_auc_monitor"], out["tgt_rec_auc_monitor"] = fast_auc(Y_ALL[ti], p), record_auc(ti, p)
    return out

print("Tiện ích huấn luyện: OK")
''')

code(r'''
# =========================================================
# 8.2 Huấn luyện một job: Source-only / CORAL / MMD / DANN / DANN+CORAL(+PhysAug) / Oracle
# =========================================================
def train_da(job, device):
    device = torch.device(device)
    seed, bb, method = int(job["seed"]), job["backbone"], job["method"]
    info = METHOD_INFO[method]; sp = SPLITS[seed]; fd = _fold_of(job)
    js = _job_seed(job)
    model = make_model(bb, js, device)
    set_specaug_generators(model, torch.Generator().manual_seed(js + 4))
    g_src, g_tgt, g_phys = (make_generator(device, js + o) for o in (1, 2, 3))
    rng_src, rng_tgt = np.random.default_rng(js + 5), np.random.default_rng(js + 6)

    # ---- tập CÓ NHÃN và tập KHÔNG NHÃN ----
    if info["label_dom"] == 1:                                  # Oracle: HDPE có nhãn
        lab_idx = np.asarray(fd["t_adapt_lab_idx"]); y_lab = labels_for(lab_idx, reveal_target_labels(lab_idx))
    else:                                                       # mọi phương pháp UDA: CHỈ PVC có nhãn
        lab_idx = np.asarray(sp["src_train_idx"]); y_lab = labels_for(lab_idx, Y_SRC_VISIBLE)
        assert (DOM_ALL[lab_idx] == 0).all()
    tgt_idx = np.asarray(fd["t_adapt_idx"]) if info["tgt"] else None
    if tgt_idx is not None:
        assert len(tgt_idx) > 1 and (DOM_ALL[tgt_idx] == 1).all()
        assert np.intersect1d(np.unique(REC_ALL[tgt_idx]), np.unique(REC_ALL[fd["t_test_idx"]])).size == 0

    # ---- CMVN theo miền ----
    if info["label_dom"] == 1:
        fit_cmvn(model, lab_idx, 1, device); copy_cmvn(model, 1, 0)
    else:
        fit_cmvn(model, lab_idx, 0, device)
        if tgt_idx is not None:
            fit_cmvn(model, tgt_idx, 1, device)                 # thống kê HDPE KHÔNG cần nhãn
        else:
            copy_cmvn(model, 0, 1)                              # Source-only: nhìn HDPE bằng thống kê PVC

    params = list(model.G_f.parameters()) + list(model.G_y.parameters())
    if info["dann"]:
        params += list(model.G_d.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    bs = int(cfg.batch_size)
    spe = max(1, int(math.ceil(len(lab_idx) / bs)))
    total = spe * int(cfg.epochs)
    sched = make_sched(opt, total)
    amp_on = use_amp(device); scaler = make_grad_scaler(amp_on)
    pw = torch.tensor(pos_weight(y_lab), device=device, dtype=torch.float32)
    eps = float(cfg.label_smoothing)
    hist = defaultdict(list); step = 0; n_skip = 0; t0 = time.time(); lam = 0.0

    for ep in range(int(cfg.epochs)):
        model.train()
        perm = rng_src.permutation(len(lab_idx))
        logs = defaultdict(list)
        for s in range(spe):
            pos = perm[s * bs:(s + 1) * bs]
            if len(pos) < 2:                                    # BatchNorm cần >= 2 mẫu
                pos = np.concatenate([pos, rng_src.integers(0, len(lab_idx), 2 - len(pos))])
            ns = len(pos)
            xs = augment(fetch(device, lab_idx[pos]), g_src, phys_g=(g_phys if info["phys"] else None))
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
                l_y = F.binary_cross_entropy_with_logits(model.G_y(fs).float(), ys * (1 - eps) + eps / 2,
                                                         pos_weight=pw)
                loss = l_y
                if info["dann"]:
                    model.grl.lambd = float(cfg.lambda_dann) * lam
                    dl = model.G_d(model.grl(f)).float()
                    l_d = (F.binary_cross_entropy_with_logits(dl[:ns], torch.zeros_like(dl[:ns])) +
                           F.binary_cross_entropy_with_logits(dl[ns:], torch.ones_like(dl[ns:])))
                    loss = loss + l_d
                    logs["l_d"].append(l_d.detach())
                    logs["acc_d"].append(0.5 * ((dl[:ns] < 0).float().mean() + (dl[ns:] > 0).float().mean()))
                if tgt_idx is not None:
                    ft = f[ns:]
                    if info["coral"]:
                        l_c = coral_loss(fs, ft); loss = loss + cfg.coral_weight * ramp * l_c
                    else:                                       # chỉ đo (chẩn đoán), không tối ưu
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
                  f"acc_d {hist['acc_d'][-1]:.2f} CORAL {hist['l_coral'][-1]:.4f} | AUC PVC-val "
                  f"{hist['src_val_auc'][-1]:.3f} | HDPE (monitor) {hist['tgt_auc_monitor'][-1]:.3f}")

    # ---- mô hình = EPOCH CUỐI (đăng ký trước) ----
    p_test = predict(model, fd["t_test_idx"], device)
    sv = sp["src_val_idx"] if info["src"] else np.zeros(0, np.int64)
    p_sv = predict(model, sv, device)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out = dict(job=dict(job), hist=pd.DataFrame(hist), state=state, test_idx=np.asarray(fd["t_test_idx"]),
               p_test=p_test, srcval_idx=np.asarray(sv), p_srcval=p_sv, train_time_s=time.time() - t0,
               n_skip=n_skip, n_labeled=len(lab_idx), n_unlabeled=0 if tgt_idx is None else len(tgt_idx))
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
    # Thống kê miền đích (KHÔNG nhãn): CMVN + BatchNorm
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
        spe = max(4, int(math.ceil(len(fs_idx) / bs)))
        for ep in range(int(cfg.ft_epochs)):
            model.train()
            for m in bns:
                m.eval()                          # giữ thống kê AdaBN: vài bản ghi không đủ để ước lượng BN
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
               n_unlabeled=len(fd["t_adapt_idx"]))
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
    """Mô hình CUỐI: toàn bộ PVC train + TOÀN BỘ HDPE (không nhãn) - dùng để xuất và xếp hạng."""
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

`len(seeds) × len(backbones) × |phương pháp huấn luyện| × n_fold` job huấn luyện, sau đó đến các job dựng sau
(AdaBN, Few-shot). Nếu HDPE không có nhãn (`EVAL_ON = False`), mục này được bỏ qua và notebook chuyển sang
huấn luyện mô hình cuối (mục 11–12).
''')

code(r'''
# =========================================================
# 9.1 Huấn luyện toàn bộ / Train everything
# =========================================================
RESULTS = {}
if EVAL_ON:
    _tm = [m for m in TRAIN_METHODS if not METHOD_INFO[m].get("needs_target_labels") or TARGET_HAS_LABELS]
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
                              n_labeled=r["n_labeled"], n_unlabeled=r["n_unlabeled"], n_skip=r["n_skip"],
                              frame_auc=fast_auc(Y_ALL[r["test_idx"]], r["p_test"]),
                              record_auc=record_auc(r["test_idx"], r["p_test"]))
                         for k, r in RESULTS.items()])
    _log.to_csv(TAB_DIR / "job_log.csv", index=False)
    print(f"Tổng thời gian huấn luyện: {_log.time_s.sum() / 60:.1f} phút-thiết bị")
else:
    print("!! EVAL_ON = False: HDPE không đủ nhãn để đánh giá -> bỏ qua ma trận thí nghiệm.")
    print("   Chuyển sang mục 11-12: huấn luyện mô hình cuối và xếp hạng bản ghi HDPE.")
''')

# =============================================================================
# 10. Kết quả
# =============================================================================
md(r'''
## 10. Kết quả trên miền đích / Results on the target domain

- **Chỉ số chính:** AUC **mức bản ghi** trên HDPE kiểm thử (gộp `p_mean`, đăng ký trước). Dự đoán được gộp qua
  các fold (mỗi bản ghi đúng một lần) và trung bình qua các seed.
- **CI 95 %:** bootstrap **theo cụm bản ghi**. Frame của cùng một bản ghi không độc lập, nên bootstrap theo frame
  sẽ cho CI hẹp giả tạo.
- **So sánh với Source-only:** bootstrap **ghép cặp** của ΔAUC (cùng các bản ghi được lấy lại cho mọi phương
  pháp), p hai phía, hiệu chỉnh **Holm** trong từng kiến trúc.
- **Khoảng cách chuyển giao đã thu hẹp:** $\frac{AUC_m - AUC_{src}}{AUC_{oracle} - AUC_{src}}$.
- Accuracy/F1 dùng ngưỡng cố định 0,5. Khi điểm số trên miền đích bị lệch toàn cục, AUC vẫn có thể cao trong khi
  độ chính xác cân bằng thấp. Đây cũng là một biểu hiện của lệch miền.
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
        praw = {}
        for j, m in enumerate(names):
            if j != i0:
                d = (BR[:, j] - BR[:, i0]); d = d[np.isfinite(d)]
                praw[m] = (max(1.0 / len(d), min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
                           if len(d) else np.nan)
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
            # chỉ có nghĩa khi Oracle vượt Source-only một khoảng đáng kể (tránh chia cho ~0)
            gap = (100 * (e["record_auc"] - a_src) / (a_or - a_src)
                   if (np.isfinite(a_or) and a_or - a_src >= 0.05 and m not in ("oracle", "source_only"))
                   else np.nan)
            rows.append(dict(backbone=bb, method=m, label=METHOD_LABEL[m], **e,
                             frame_auc_lo=cf[0], frame_auc_hi=cf[1], record_auc_lo=cr[0], record_auc_hi=cr[1],
                             delta_rec_auc=e["record_auc"] - a_src, delta_lo=dd[0], delta_hi=dd[1],
                             p_raw=praw.get(m, np.nan), p_holm=padj.get(m, np.nan), gap_closed_pct=gap,
                             fold_rec_auc_mean=np.nanmean(fold_auc) if fold_auc else np.nan,
                             fold_rec_auc_sd=np.nanstd(fold_auc) if fold_auc else np.nan,
                             src_val_auc=np.nanmean(sv) if sv else np.nan))
    RES = pd.DataFrame(rows)
    RES.to_csv(TAB_DIR / "table5_target_results_full.csv", index=False)

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
    display(Markdown(f"### Table 5. {cfg.source_name} → {cfg.target_name} transfer: target-domain performance "
                     f"({int(RES.n_records.iloc[0])} bản ghi kiểm thử, {len(cfg.seeds)} seed)"))
    display(show)
    show.to_csv(TAB_DIR / "table5_target_results.csv", index=False)
    try:
        (TAB_DIR / "table5_target_results.tex").write_text(
            show.to_latex(index=False, escape=True, caption=f"{cfg.source_name} to {cfg.target_name} transfer",
                          label="tab:da_results"), encoding="utf-8")
    except Exception as e:
        print(f"  (bỏ qua LaTeX: {type(e).__name__})")
else:
    print("EVAL_ON = False -> không có bảng kết quả miền đích.")
''')

code(r'''
# =========================================================
# 10.2 Hình: forest plot AUC mức bản ghi trên miền đích
# =========================================================
if EVAL_ON and len(RES):
    bbs = list(RES.backbone.unique())
    _order = [m for m in cfg.methods if m in set(RES.method)]
    fig, axes = plt.subplots(1, len(bbs), figsize=(4.6 * len(bbs) + 2.4, 0.5 * len(_order) + 1.8), squeeze=False,
                             sharey=True, constrained_layout=True)
    ypos = {m: len(_order) - 1 - j for j, m in enumerate(_order)}      # cùng hàng cho mọi kiến trúc
    for ia, (a, bb) in enumerate(zip(axes[0], bbs)):
        g = RES[RES.backbone == bb].reset_index(drop=True)
        for r in g.itertuples():
            y, c = ypos[r.method], METHOD_COLORS.get(r.method, "k")
            a.errorbar(r.record_auc, y, xerr=[[max(0, r.record_auc - r.record_auc_lo)], [max(0, r.record_auc_hi - r.record_auc)]],
                       fmt="o", color=c, capsize=3, ms=6)
            a.plot(r.frame_auc, y, marker="x", color=c, ls="none", ms=6)
        src = float(g.record_auc[g.method == "source_only"].iloc[0])
        a.axvline(src, color=METHOD_COLORS["source_only"], ls="--", lw=1, label="Source-only")
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
            f"Nét đứt: Source-only (cận dưới); gạch chấm: Oracle (cận trên).")
''')

code(r'''
# =========================================================
# 10.3 Hình: ROC trên miền đích (kiến trúc chính)
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
    savefig(fig, "target_roc", f"ROC trên {cfg.target_name} kiểm thử, kiến trúc {cfg.main_backbone} (dự đoán gộp qua các fold).")
''')

code(r'''
# =========================================================
# 10.4 Hình: động học huấn luyện - Giai đoạn 3 (đối kháng & căn chỉnh)
# =========================================================
if EVAL_ON:
    s0, bb = cfg.seeds[0], cfg.main_backbone
    show = [m for m in ("source_only", "coral", "mmd", "dann", "dann_coral", "dann_coral_phys") if (s0, bb, m, 0) in RESULTS]
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
    ax[1].axhline(2 * np.log(2), color="k", ls=":", lw=.8, label="2·ln2 (G_d đoán mò)")
    ax[2].axhline(0.5, color="k", ls=":", lw=.8, label="0,5 (không phân biệt được miền)")
    ttl = ["(a) L_y trên PVC", "(b) L_d (BCE miền, sau GRL)", "(c) độ chính xác G_d",
           "(d) CORAL(f_S, f_T) theo lô\n(nét chấm: chỉ đo, không tối ưu)", f"(e) AUC {cfg.source_name}-val",
           f"(f) AUC bản ghi {cfg.target_name}\nCHỈ THEO DÕI - không dùng để chọn"]
    for a, t in zip(ax, ttl):
        a.set_title(t, fontsize=9.5); a.set_xlabel("epoch")
    ax[0].legend(fontsize=7)
    for a in (ax[1], ax[2]):
        if a.lines:
            a.legend(fontsize=7)
    savefig(fig, "training_dynamics",
            f"Giai đoạn 3: động học huấn luyện ({bb}, fold 0). Với DANN, L_d tiến về 2·ln2 và độ chính xác G_d về 0,5 "
            f"khi G_f tạo được đặc trưng không phân biệt được miền; CORAL giảm khi hiệp phương sai hai miền khớp nhau.")
''')

code(r'''
# =========================================================
# 10.5 Hình: hiệu năng miền nguồn vs miền đích (DA không được phá PVC)
# =========================================================
if EVAL_ON and len(RES):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for bb, mk in zip(RES.backbone.unique(), ("o", "s", "^")):
        g = RES[(RES.backbone == bb) & RES.src_val_auc.notna()]
        for r in g.itertuples():
            ax[0].scatter(r.src_val_auc, r.record_auc, color=METHOD_COLORS.get(r.method), marker=mk, s=55, zorder=3)
            ax[0].annotate(r.method, (r.src_val_auc, r.record_auc), fontsize=7, xytext=(3, 3), textcoords="offset points")
    lims = [min(.4, np.nanmin(RES[["src_val_auc", "record_auc"]].values) - .02), 1.01]
    ax[0].plot(lims, lims, "k:", lw=.7); ax[0].set_xlim(lims); ax[0].set_ylim(lims)
    ax[0].set_xlabel(f"AUC {cfg.source_name}-val (frame)"); ax[0].set_ylabel(f"AUC bản ghi {cfg.target_name}")
    ax[0].set_title("(a) Nguồn vs đích (o = " + ", ".join(RES.backbone.unique()) + ")")
    g = RES[RES.backbone == cfg.main_backbone]
    ax[1].barh(g.label, g.delta_rec_auc, color=[METHOD_COLORS.get(m) for m in g.method],
               xerr=[(g.delta_rec_auc - g.delta_lo).clip(lower=0).fillna(0), (g.delta_hi - g.delta_rec_auc).clip(lower=0).fillna(0)],
               capsize=3)
    ax[1].axvline(0, color="k", lw=.8); ax[1].invert_yaxis()
    ax[1].set_xlabel("ΔAUC bản ghi so với Source-only (CI 95% ghép cặp)"); ax[1].set_title(f"(b) {cfg.main_backbone}")
    savefig(fig, "source_vs_target", "(a) Hiệu năng giữ trên PVC so với hiệu năng chuyển sang HDPE; "
            "(b) cải thiện ghép cặp so với Source-only.")
''')

# =============================================================================
# 11. Không gian ẩn
# =============================================================================
md(r'''
## 11. Giai đoạn 4 — không gian đặc trưng bất biến miền / Domain-invariant latent space

Không gian ẩn $f = G_f(x)$ được so sánh **trước** (Source-only) và **sau** căn chỉnh (CORAL, DANN,
DANN+CORAL), trên PVC-val và HDPE kiểm thử. Đây là hai tập dữ liệu không mô hình nào được huấn luyện trên đó.

| Chỉ số (đo trong không gian GỐC d chiều, đã chuẩn hoá chung) | Ý nghĩa | Mong muốn |
|---|---|---|
| **Proxy $\mathcal A$-distance** $=2(1-2\epsilon)$ | $\epsilon$ = lỗi của hồi quy logistic phân biệt miền (CV **theo bản ghi**) | → 0 |
| **CORAL** $\frac{1}{4d^2}\lVert C_S-C_T\rVert_F^2$ | lệch hiệp phương sai | ↓ |
| **MMD²** (RBF đa nhân) | lệch phân phối tổng quát | ↓ |
| **cos(hướng rò)** $\cos(\mu^S_1-\mu^S_0,\;\mu^T_1-\mu^T_0)$ | hướng "leak − no-leak" có giữ nguyên qua miền? (nhãn HDPE chỉ để đánh giá) | → 1 |
| **AUC chuyển giao tuyến tính** | logistic học trên đặc trưng PVC, thử trên HDPE | ↑ |

> Căn chỉnh **biên** (A-distance → 0) mà không giữ **hướng lớp** (cos thấp) là dấu hiệu *negative transfer*:
> hai miền chồng lên nhau nhưng leak HDPE nằm cạnh no-leak PVC. Bản đồ t-SNE chỉ minh hoạ; kết luận dựa trên các chỉ số.
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
    # CV theo BẢN GHI, chia fold RIÊNG trong từng miền -> mọi fold kiểm thử đều có cả hai miền
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
_cands = ("source_only", "coral", "mmd", "dann", "dann_coral", "dann_coral_phys")
if EVAL_ON:
    s0, bb = cfg.seeds[0], cfg.main_backbone
    fd0 = SPLITS[s0]["folds"][0]
    lat_src = {m: RESULTS[(s0, bb, m, 0)]["state"] for m in _cands if (s0, bb, m, 0) in RESULTS}
    i_s = _subsample(SPLITS[s0]["src_val_idx"], cfg.tsne_max_points // 2, 1)
    i_t = _subsample(fd0["t_test_idx"], cfg.tsne_max_points // 2, 2)
    lat_scope = f"{bb}, fold 0: {cfg.source_name}-val vs {cfg.target_name} kiểm thử"
else:
    # Không có nhãn HDPE: so sánh mô hình CUỐI Source-only vs phương pháp đề xuất (chỉ số không cần nhãn)
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
display(Markdown(f"### Table 6. Latent-space alignment ({lat_scope}; {len(i_s)} + {len(i_t)} frame)"))
display(LATENT_TABLE)
LATENT_TABLE.to_csv(TAB_DIR / "table6_latent_alignment.csv", index=False)
''')

code(r'''
# =========================================================
# 11.2 Hình: t-SNE trước / sau căn chỉnh (4 nhóm của sơ đồ)
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
    ns = len(L["Es"])
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
        f"Giai đoạn 4: t-SNE của không gian đặc trưng f = G_f(x) ({lat_scope}). ○ {cfg.source_name}, △ {cfg.target_name}. "
        f"Căn chỉnh tốt: hai miền chồng lên nhau (A-distance → 0) trong khi Leak/Safe vẫn tách rời (cos hướng rò → 1).")
''')

# =============================================================================
# 12. Mô hình cuối & xuất
# =============================================================================
md(r'''
## 12. Mô hình triển khai / Deployment model

Phương pháp xuất là `cfg.final_method` (**DANN + CORAL**, chọn **trước** khi xem kết quả, nên không thiên lệch
do lựa chọn). Mô hình cuối được huấn luyện trên toàn bộ PVC-train có nhãn và **toàn bộ** HDPE không nhãn, sau đó
xuất các thành phần sau:

| Tệp | Vai trò |
|---|---|
| `leak_da_<backbone>_<method>.ts` | TorchScript: sóng 2 s @ 2 kHz (đã tiền xử lý + VMD) → xác suất rò, CMVN miền HDPE đã nhúng |
| `..._state.pt` | state_dict + cấu hình |
| `manifest.json` | toàn bộ tham số tiền xử lý cần để tái lập đúng đầu vào, cùng kết quả đã đo |
| `target_record_ranking.csv` | xếp hạng bản ghi HDPE theo xác suất rò (gộp `p_mean`), dùng làm danh sách ưu tiên khảo sát |
''')

code(r'''
# =========================================================
# 12.1 Huấn luyện mô hình cuối + xuất TorchScript
# =========================================================
class DeployNet(nn.Module):
    # Bọc DANNNet cho triển khai trên MỘT miền cố định (mặc định HDPE = 1)
    def __init__(self, net, dom=1):
        super().__init__()
        self.net, self.dom = net, int(dom)
    def forward(self, x):
        d = torch.zeros_like(x[:, 0]).long() + self.dom
        return torch.sigmoid(self.net(x, d))

FINAL = None
if cfg.train_final_model:
    FINAL = get_final(cfg.final_method)
    print(f"Mô hình cuối {cfg.main_backbone}/{cfg.final_method}: {FINAL['n_labeled']} frame PVC có nhãn + "
          f"{FINAL['n_unlabeled']} frame {cfg.target_name} không nhãn, huấn luyện {FINAL['train_time_s']:.0f} s")
    print(f"Kiểm tra hợp lý: AUC {cfg.source_name}-val của mô hình cuối = "
          f"{fast_auc(Y_ALL[FINAL['srcval_idx']], FINAL['p_srcval']):.3f}")
    net = DANNNet(cfg.main_backbone); net.load_state_dict(FINAL["state"]); net.eval()
    dep = DeployNet(net, 1).eval()
    _ex = torch.from_numpy(X_ALL[np.where(DOM_ALL == 1)[0][:4]].copy())
    with torch.no_grad():
        _ref = dep(_ex)
    stem = f"leak_da_{cfg.main_backbone}_{cfg.final_method}"
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
    manifest = dict(
        model=f"{stem}.ts", input=dict(shape=["B", FRAME_LEN], sr=cfg.target_sr, frame_sec=cfg.frame_sec,
                                       hop_sec=cfg.hop_sec),
        preprocessing=dict(resample_to=cfg.target_sr, dc_removal=True, bandpass_hz=list(cfg.band_hz),
                           butter_order=cfg.filter_order, zero_phase=True, notch_mains_hz=cfg.mains_hz if cfg.notch_mains else None,
                           notch_q=cfg.notch_q, peak_normalise="per file (OFFLINE)",
                           vmd=(dict(K=cfg.vmd_K, alpha=cfg.vmd_alpha, tau=cfg.vmd_tau, iters=cfg.vmd_iters,
                                     keep="modes with centre in band_hz and >%.1f Hz from mains harmonics" % cfg.vmd_mains_tol_hz)
                                if cfg.use_vmd else None)),
        output="P(leak) per 2-s frame; record decision = mean over frames (p_mean) >= threshold",
        threshold=cfg.decision_threshold, domain_embedded=cfg.target_name, tf_repr=cfg.tf_repr,
        data_mode=DATA_MODE, source=str(SRC_ROOT), target=str(TGT_ROOT), performance=_perf,
        warnings=([f"Miền đích là PVC biến đổi vật lý (physics_sim) - CHƯA kiểm chứng trên HDPE thật."]
                  if DATA_MODE == "physics_sim" else
                  ["Dữ liệu mô phỏng - chỉ để kiểm thử pipeline."] if DATA_MODE == "synthetic" else []))
    (EXPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Đã xuất:", sorted(p.name for p in EXPORT_DIR.iterdir()))
''')

code(r'''
# =========================================================
# 12.2 Xếp hạng bản ghi HDPE theo xác suất rò (danh sách ưu tiên khảo sát hiện trường)
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
    rank["label_if_known"] = rank.record_id.map(lambda r: {1: "leak", 0: "noleak"}.get(REC_LABEL[r], "?"))
    rank.to_csv(EXPORT_DIR / "target_record_ranking.csv", index=False)
    display(Markdown(f"### Table 7. {cfg.target_name} records ranked by leak probability (mô hình cuối, top 15)"))
    display(rank.head(15).round(4))
    if TARGET_HAS_LABELS:
        _known = rank[rank.label_if_known != "?"]
        print("Lưu ý: mô hình cuối ĐÃ thấy các bản ghi này (không nhãn) khi huấn luyện -> AUC dưới đây là "
              f"transductive, chỉ để đối chiếu: {fast_auc((_known.label_if_known == 'leak').values, _known.p_mean.values):.3f}")
    del net
''')

code(r'''
# =========================================================
# 12.3 Suy luận trọn tuyến trên một tệp HDPE thô (mô phỏng triển khai)
# =========================================================
TEST_FILE = ""          # để trống = tự chọn tệp của bản ghi HDPE có điểm cao nhất
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
    ts_model = torch.jit.load(str(EXPORT_DIR / f"leak_da_{cfg.main_backbone}_{cfg.final_method}.ts"))
    with torch.no_grad():
        pf = ts_model(torch.from_numpy(Xf)).numpy()
    fig = plt.figure(figsize=(13.5, 6.4))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1.4, 1], hspace=.45)
    a0 = fig.add_subplot(gs[0]); tt = np.arange(len(x)) / cfg.target_sr
    a0.plot(tt, x, lw=.4, color=PAL["grey"]); a0.set_title(f"{Path(TEST_FILE).name} - sau tiền xử lý"); a0.set_xlim(0, tt[-1])
    a1 = fig.add_subplot(gs[1], sharex=a0)
    with torch.no_grad():
        S = make_tf_frontend()(torch.from_numpy(x[None].copy()))[0, 0].numpy()
    fe_ = make_tf_frontend()
    a1.imshow(S, origin="lower", aspect="auto", cmap="magma", extent=[0, tt[-1], -.5, fe_.n_bands - .5])
    yt = [f for f in (20, 50, 100, 200, 350) if cfg.band_hz[0] <= f <= cfg.band_hz[1]]
    a1.set_yticks(np.interp(np.log(yt), np.log(fe_.freqs), np.arange(fe_.n_bands))); a1.set_yticklabels(yt)
    a1.set_ylabel("Hz"); a1.grid(False)
    a2 = fig.add_subplot(gs[2], sharex=a0)
    a2.plot(starts / cfg.target_sr + cfg.frame_sec / 2, pf, "o-", color=PAL["tgt"], ms=3)
    a2.axhline(cfg.decision_threshold, color="k", ls=":"); a2.set_ylim(-.02, 1.02)
    a2.set_ylabel("P(leak)"); a2.set_xlabel("thời gian (s)")
    a2.set_title(f"p_mean = {pf.mean():.3f} -> {'NGHI RÒ' if pf.mean() >= cfg.decision_threshold else 'bình thường'}"
                 + (f" | nhãn thật: {'leak' if REC_LABEL[_rid] == 1 else 'noleak'}"
                    if _rid is not None and REC_LABEL[_rid] >= 0 else ""))
    savefig(fig, "field_inference", "Suy luận trọn tuyến (tiền xử lý → VMD → TorchScript) trên một tệp miền đích.")
''')

# =============================================================================
# 13. Đoạn kết quả tự sinh + kết luận
# =============================================================================
md(r'''
## 13. Đoạn kết quả tự sinh và kết luận / Auto-generated results and conclusions

Đoạn dưới đây được **sinh từ số liệu** của lần chạy này. Mọi câu khẳng định đều đi kèm CI, và
notebook tự liệt kê những điều **không được** khẳng định.
''')

code(r'''
# =========================================================
# 13.1 Đoạn Results tự sinh - CHỈ phát biểu điều số liệu cho phép
# =========================================================
L = []
nS, nT = int((records.domain == 0).sum()), int((records.domain == 1).sum())
L.append(f"**Dữ liệu.** Chế độ `{DATA_MODE}`: {nS} bản ghi {cfg.source_name} "
         f"({records.duration_s[records.domain == 0].sum() / 60:.1f} phút) và {nT} bản ghi {cfg.target_name} "
         f"({TGT_LAB_COUNTS[1]} leak / {TGT_LAB_COUNTS[0]} no-leak có nhãn, {N_TGT_UNLAB} không nhãn); "
         f"frame {cfg.frame_sec:g} s @ {cfg.target_sr} Hz, dải {cfg.band_hz[0]:g}-{cfg.band_hz[1]:g} Hz, "
         f"{'VMD + ' if cfg.use_vmd else ''}{cfg.tf_repr.upper()}, CMVN theo miền = {cfg.input_norm == 'domain_cmvn'}.")
L.append(f"**Lệch miền vật lý.** Korteweg: c_{cfg.source_name} ≈ {C_PVC:.0f} m/s, c_{cfg.target_name} ≈ {C_HDPE:.0f} m/s "
         f"(r = {R_KORTEWEG:.2f}); tỉ số tần số trọng tâm đo được (không dùng nhãn) r_emp = {R_EMP:.2f}.")
if EVAL_ON and len(RES):
    for bb in RES.backbone.unique():
        g = RES[RES.backbone == bb].set_index("method")
        so = g.loc["source_only"]
        s = (f"**{bb}.** Source-only đạt AUC bản ghi {so.record_auc:.3f} [{so.record_auc_lo:.3f}, {so.record_auc_hi:.3f}] "
             f"trên {cfg.target_name} (PVC-val {so.src_val_auc:.3f})")
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
            s += (f" Trong các phương pháp không nhãn, cao nhất là {METHOD_LABEL[b]} ({uda.record_auc.max():.3f}) - "
                  f"*chọn sau khi xem kết quả nên chỉ mang tính mô tả*.")
        L.append(s)
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
if DATA_MODE == "physics_sim":
    NOT.append(f"Miền đích là **PVC biến đổi vật lý**, KHÔNG phải ống HDPE thật: không được trình bày như kết quả trên HDPE.")
if DATA_MODE == "synthetic":
    NOT.append("Dữ liệu **mô phỏng**: mọi con số chỉ kiểm chứng pipeline.")
if not EVAL_ON:
    NOT.append(f"{cfg.target_name} không có nhãn: không có chỉ số hiệu năng nào trên miền đích; bảng xếp hạng chỉ là danh sách ưu tiên khảo sát.")
if EVAL_ON and len(RES) and int(RES.n_records.iloc[0]) < 20:
    NOT.append(f"Chỉ {int(RES.n_records.iloc[0])} bản ghi {cfg.target_name} kiểm thử: CI rộng, không khái quát hoá cho mạng lưới.")
if len(cfg.seeds) < 3:
    NOT.append(f"Chỉ {len(cfg.seeds)} seed: chưa đo độ bất định do khởi tạo/phân hoạch (đặt cfg.seeds = (42, 43, 44)).")
NOT.append("Không có thông tin điểm khảo sát/tuyến ống: chưa loại trừ khả năng mô hình học 'dấu vân tay vị trí' (cần leave-one-site-out).")
NOT.append("Mô hình xử lý OFFLINE (chuẩn hoá đỉnh theo cả tệp, lọc pha 0): chưa phải bộ phát hiện thời gian thực.")

txt = "\n\n".join(L) + "\n\n**Không được khẳng định:**\n" + "\n".join(f"- {s}" for s in NOT)
display(Markdown(txt))
(OUT / "results_paragraph.md").write_text(txt, encoding="utf-8")
print("Hình:", len(FIG_INDEX), "| bảng:", sorted(p.name for p in TAB_DIR.iterdir()))
''')

md(r'''
### Kết luận, hạn chế và hướng tiếp theo / Conclusions, limitations and next steps

**Những gì notebook hiện thực (đối chiếu tài liệu):**

| Tài liệu | Hiện thực |
|---|---|
| §1 Korteweg, dịch tần, suy hao | 3.1 (bảng $c$), 4 (PSD, $r_{emp}$), 5.3 (tâm mode VMD theo miền) |
| §3.1 CMVN, lọc dải 15–350 Hz | 3.2, `DomainCMVN` (6.1) |
| §3.2 AdaBN, fine-tuning chọn lọc (1e-4 / 1e-5 / 1e-3) | `adabn_`, `run_post` (8.3) |
| §3.3A DANN + GRL, minimax | `GradReverse`, `DomainDiscriminator`, `train_da` (6.3, 8.2) |
| §3.3B Deep CORAL ($\frac{1}{4d^2}\lVert C_S-C_T\rVert_F^2$), Deep MMD | `coral_loss`, `mmd_rbf` (6.3) |
| §4 Giả lập vật lý PVC→HDPE | `phys_aug` (GPU), chế độ `physics_sim` |
| §5 Source-only / UDA / Few-shot / Oracle, chia theo bản ghi | 7–10 |
| Sơ đồ: VMD + CWT + CNN-CBAM, t-SNE trước/sau | 5.2, 6.1–6.2, 11 |

**Hạn chế và việc nên làm tiếp:**

1. **Dữ liệu HDPE thật** là điều kiện để mọi kết luận về chuyển giao PVC→HDPE có giá trị. Ở chế độ `physics_sim`,
   notebook chỉ chứng minh pipeline hoạt động và đo độ nhạy với một mô hình vật lý giả định.
2. **Chọn mô hình không nhãn** vẫn là bài toán mở trong UDA. Notebook dùng quy tắc *epoch cuối + ngưỡng 0,5*
   đăng ký trước. Có thể thử DEV (Deep Embedded Validation) hoặc reverse validation, nhưng phải giữ tách biệt với tập báo cáo.
3. **SupCon đa miền** (tài liệu §3.3C) và **One-class DA** (chỉ có HDPE no-leak) chưa được hiện thực. Có thể thêm
   như một nhánh mất mát trên cùng `G_f`.
4. **Leave-one-site-out / leave-one-pipe-out** khi có thông tin điểm khảo sát. Đây là kiểm định quan trọng nhất
   còn thiếu, giống như ở notebook v7.
5. Tăng `cfg.seeds` lên 3–5 trước khi chốt số liệu cho bài báo.
''')
