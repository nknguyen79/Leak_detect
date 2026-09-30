# %% [markdown]
# ## 1. Thiết lập môi trường / Environment setup

# %%
# =========================================================
# 1.1 Thư viện / Imports
#     Chỉ cần CPU. Mọi thư viện đều có sẵn trên Kaggle; thiếu thì cài tự động.
# =========================================================
import os, sys, re, json, math, time, hashlib, warnings, subprocess, unicodedata, itertools, platform, shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")


def _pip(pkgs):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0


try:
    import soundfile as sf
except Exception:
    _pip(["soundfile"]); import soundfile as sf
try:
    import pywt
except Exception:
    _pip(["PyWavelets"]); import pywt

import numpy as np
import pandas as pd
import scipy
import scipy.signal as sps
import scipy.stats as sst
from scipy.fft import dct, rfft, irfft
from scipy.special import expit
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display, Markdown
from joblib import Parallel, delayed
import joblib

import sklearn
from sklearn.model_selection import StratifiedKFold, KFold, ParameterGrid, train_test_split
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score, precision_recall_curve,
                             confusion_matrix, matthews_corrcoef)

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = _pip(["xgboost"])
    if HAS_XGB:
        import xgboost as xgb
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

N_CPU = os.cpu_count() or 2
print(f"Python {sys.version.split()[0]} | NumPy {np.__version__} | SciPy {scipy.__version__} | "
      f"pandas {pd.__version__} | scikit-learn {sklearn.__version__}")
print(f"librosa {librosa.__version__} | PyWavelets {pywt.__version__} | "
      f"xgboost {xgb.__version__ if HAS_XGB else 'KHÔNG CÓ -> dùng HistGradientBoosting'} | "
      f"shap {shap.__version__ if HAS_SHAP else 'không có -> dùng permutation importance'}")
print(f"CPU: {N_CPU} lõi | {platform.platform()}")

# %%
# =========================================================
# 1.2 Cấu hình trung tâm / Central configuration
#     Mọi tham số thí nghiệm nằm ở đây. Các giá trị mặc định được CỐ ĐỊNH TRƯỚC
#     (pre-registered) - không chỉnh theo kết quả trên tập kiểm tra.
# =========================================================
@dataclass
class CFG:
    # ---- Dữ liệu ----------------------------------------------------------
    data_root: str = "/kaggle/input/datasets/nguyennguyenkhac/tapdulieutt-8k-v7-2/field_rec_8k_V7.2"
    leak_keys: tuple = ("leak", "ro_ri", "rori")
    noleak_keys: tuple = ("noleak", "no_leak", "non_leak", "nonleak", "normal", "background",
                          "khong_ro_ri", "khongrori")
    audio_ext: tuple = (".wav", ".flac", ".ogg", ".mp3")
    record_regex: str = r"(R\d{1,4})"      # mã bản ghi = ĐƠN VỊ ĐỘC LẬP (nhóm của CV)

    # ---- Tín hiệu & khung phân tích ---------------------------------------
    target_sr: int = 16000                 # dữ liệu gốc 16 kHz -> giữ nguyên để xét tới 8 kHz
    frame_sec: float = 2.0                 # độ dài một khung phân loại
    hop_sec: float = 2.0                   # không chồng lấn: khung gần như độc lập về thời gian
    f_lo: float = 20.0                     # dải phân tích đặc trưng [f_lo, f_hi]
    f_hi: float = 7600.0                   # tự kẹp <= 0,95 x Nyquist

    # ---- C1: lọc điều kiện hoá (band-pass + notch lưới điện) --------------
    hp_hz: float = 20.0
    lp_hz: float = 7000.0
    filt_order: int = 4
    notch_hz: float = 50.0                 # lưới điện Việt Nam
    notch_harmonics: int = 6
    notch_q: float = 30.0

    # ---- QC mức tệp (bỏ cả tệp) ------------------------------------------
    clip_level: float = 0.999              # |x| >= mức này (full-scale = 1) coi là bão hoà
    qc_max_clip_frac: float = 0.01         # > 1% mẫu bão hoà -> loại tệp
    qc_max_zero_frac: float = 0.5          # > 50% mẫu bằng 0 (mất tín hiệu) -> loại tệp

    # ---- C2: loại khung bất thường (va chạm, xung, bão hoà, mất tín hiệu) --
    spike_k: float = 8.0                   # kẹp mẫu vượt k x sigma_MAD của tệp
    block_ms: float = 50.0                 # khối con để dò sự kiện năng lượng
    z_thr: float = 3.5                     # modified z-score (Iglewicz & Hoaglin, 1993)
    event_db: float = 6.0                  # VÀ vượt trung vị tệp >= 6 dB mới tính là sự kiện
    kurt_thr: float = 8.0                  # kurtosis (Gauss = 3) vượt ngưỡng -> khung xung
    frame_clip_frac: float = 0.001
    # ---- C3: loại âm thanh không liên quan (tiếng nói, xe, chim...) --------
    lsd_z_thr: float = 3.5                 # khoảng cách log-phổ tới phổ trung vị của tệp
    lsd_min_db: float = 3.0
    voicing_thr: float = 0.5               # đỉnh ACF chuẩn hoá (70-500 Hz) -> khối "hữu thanh"
    voicing_frac_thr: float = 0.3
    voicing_excess: float = 0.2            # phải cao hơn mức nền của chính tệp đó
    min_keep_frac: float = 0.3             # luôn giữ >= 30% số khung của mỗi tệp
    min_keep_frames: int = 3

    # ---- Khử nhiễu (C4-C6) -------------------------------------------------
    stft_nfft: int = 1024
    stft_hop: int = 256
    mbg_beta: float = 2.0                  # C4: kẹp biên độ > beta x trung vị theo thời gian
    ss_alpha: float = 2.0                  # C5: trừ phổ (Boll 1979; Berouti 1979)
    ss_floor: float = 0.1
    ss_quantile: float = 0.2
    wt_wavelet: str = "db8"                # C6: VisuShrink soft-threshold (Donoho & Johnstone 1994)
    wt_level: int = 6
    conditions: tuple = ("C0_RAW", "C1_BPF", "C2_TR", "C3_IR", "C4_MBG", "C5_SS", "C6_WT")

    # ---- Chia dữ liệu & CV ------------------------------------------------
    holdout_frac: float = 0.2              # tập KIỂM TRA KHOÁ (mức bản ghi); 0 = không dùng
    split_seed: int = 2024
    n_splits: int = 5
    seeds: tuple = (42, 43, 44)            # CV lặp 3 lần x 5 fold
    record_balanced: bool = True           # mỗi bản ghi đóng góp như nhau, mỗi lớp như nhau

    # ---- Mô hình -----------------------------------------------------------
    models: tuple = ("SVM", "RF", "XGB", "KNN")   # thêm "LR", "MLP" nếu muốn

    # ---- E1: dải tần -------------------------------------------------------
    e1_scope: str = "all"                  # "all" (mô tả, không dẫn tới quyết định nào) | "dev"
    e1_condition: str = "C3_IR"            # tín hiệu sạch nhiễu xung/không liên quan, CHƯA khử nhiễu
    e1_seeds: tuple = (42, 43)
    e1_tol: float = 0.02                   # sai khác AUC chấp nhận khi tìm "dải thiết yếu"

    # ---- E2: xử lý vs thô --------------------------------------------------
    e2_regimes: tuple = ("all", "mrmr30")  # toàn bộ đặc trưng | 30 đặc trưng mRMR chọn TRONG fold
    e2_primary: str = "mrmr30"

    # ---- E3: chọn đặc trưng ------------------------------------------------
    k_grid: tuple = (3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 0)   # 0 = tất cả
    k_main_range: tuple = (20, 50)
    k_tol: float = 0.01                    # "không ảnh hưởng nhiều" = AUC giảm <= 0,01
    rankers: tuple = ("anova", "mi", "mrmr", "et", "l1")
    method_cmp_k: tuple = (10, 20, 30)
    method_cmp_models: tuple = ("SVM", "RF")
    l1_C: float = 0.05
    rank_max_rows: int = 6000

    # ---- E4: mô hình cuối --------------------------------------------------
    inner_splits: int = 3
    run_compact: bool = True

    # ---- Thống kê & đầu ra -------------------------------------------------
    n_boot: int = 2000
    alpha: float = 0.05
    out_dir: str = "/kaggle/working/leak_ml"
    fig_dpi: int = 300
    n_jobs: int = 0                        # 0 = mọi lõi CPU
    smoke_test: bool = False


cfg = CFG()
if os.environ.get("LEAK_SMOKE") == "1":
    cfg.smoke_test = True
if os.environ.get("LEAK_DATA") is not None:
    cfg.data_root = os.environ["LEAK_DATA"]
if os.environ.get("LEAK_OUT"):
    cfg.out_dir = os.environ["LEAK_OUT"]
elif not Path("/kaggle").exists():
    cfg.out_dir = str(Path("./leak_ml_outputs").resolve())

# Lưới siêu tham số (E4, chọn bằng CV lồng theo bản ghi) và giá trị mặc định (E1-E3)
PARAM_GRID = {
    "SVM": {"C": [0.3, 1.0, 3.0, 10.0, 30.0], "gamma_mult": [0.3, 1.0, 3.0]},
    "RF":  {"max_features": ["sqrt", 0.3], "min_samples_leaf": [1, 5, 20]},
    "XGB": {"max_depth": [3, 5], "learning_rate": [0.05, 0.1], "min_child_weight": [1, 5]},
    "KNN": {"n_neighbors": [11, 21, 41, 81, 161], "weights": ["uniform", "distance"]},
    "LR":  {"C": [0.01, 0.1, 1.0, 10.0]},
    "MLP": {"alpha": [1e-4, 1e-3, 1e-2], "hidden_layer_sizes": [(64,), (64, 32)]},
}
DEFAULT_PARAMS = {
    "SVM": {"C": 1.0, "gamma_mult": 1.0},
    "RF":  {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 3},
    "XGB": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0},
    "KNN": {"n_neighbors": 41, "weights": "uniform"},
    "LR":  {"C": 0.1},
    "MLP": {"hidden_layer_sizes": (64, 32), "alpha": 1e-3},
}

if cfg.smoke_test:
    print("!! SMOKE TEST: CV 1x3, lưới k rút gọn, bootstrap 200 - CHỈ để kiểm tra pipeline.")
    cfg.seeds, cfg.n_splits, cfg.e1_seeds = (42,), 3, (42,)
    cfg.k_grid = (3, 5, 10, 20, 30, 0)
    cfg.method_cmp_k = (5, 10)
    cfg.n_boot = 200
    cfg.inner_splits = 2
    cfg.fig_dpi = 110
    PARAM_GRID.update({"SVM": {"C": [1.0, 10.0], "gamma_mult": [1.0]},
                       "RF": {"max_features": ["sqrt"], "min_samples_leaf": [3]},
                       "XGB": {"max_depth": [3], "learning_rate": [0.1], "min_child_weight": [1]},
                       "KNN": {"n_neighbors": [11, 41], "weights": ["uniform"]}})
    DEFAULT_PARAMS["RF"]["n_estimators"] = 150
    DEFAULT_PARAMS["XGB"]["n_estimators"] = 150

N_JOBS = N_CPU if cfg.n_jobs in (0, None) else int(cfg.n_jobs)
OUT = Path(cfg.out_dir)
FIG_DIR, TAB_DIR, CACHE_DIR, MODEL_DIR = (OUT / d for d in ("figures", "tables", "cache", "models"))
for d in (FIG_DIR, TAB_DIR, CACHE_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

print(json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
                 indent=1, ensure_ascii=False))
print(f"\nThư mục đầu ra: {OUT} | song song: {N_JOBS} tiến trình")

# %%
# =========================================================
# 1.3 Phong cách hình vẽ, lưu hình/bảng, tiện ích thống kê
# =========================================================
# Bảng màu phân loại (thứ tự CỐ ĐỊNH; màu đi theo thực thể, không theo thứ hạng)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PAL = {"noleak": CAT[0], "leak": CAT[1]}
MODEL_COLOR = {m: CAT[i] for i, m in enumerate(["SVM", "RF", "XGB", "KNN", "LR", "MLP"])}
COND_COLOR = {c: CAT[i] for i, c in enumerate(cfg.conditions)}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_br", ["#184f95", "#6da7ec", "#f0efec", "#ef8a8a", "#b52f2f"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": cfg.fig_dpi, "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": "#b9b8b3", "axes.linewidth": 0.6, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "figure.facecolor": "white",
})

FIG_LOG = []


def savefig(fig, name, caption=""):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=cfg.fig_dpi, bbox_inches="tight")
    FIG_LOG.append((name, caption))
    plt.show()
    plt.close(fig)


def freq_axis(ax, lo=None, hi=None, label="frequency (Hz)"):
    """Trục tần số log với nhãn dễ đọc (31.5 ... 8k)."""
    ax.set_xscale("log")
    ticks = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]
    lo = lo or ax.get_xlim()[0]; hi = hi or ax.get_xlim()[1]
    tk = [t for t in ticks if lo <= t <= hi]
    ax.set_xticks(tk); ax.set_xticklabels([f"{t / 1000:g}k" if t >= 1000 else f"{t:g}" for t in tk])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(lo, hi)
    if label:
        ax.set_xlabel(label)


def savetab(df, name, index=False, floatfmt=4):
    df.to_csv(TAB_DIR / f"{name}.csv", index=index)
    try:
        (TAB_DIR / f"{name}.tex").write_text(
            df.to_latex(index=index, float_format=lambda v: f"{v:.{floatfmt}f}", escape=True))
    except Exception:
        pass
    return df


def holm(p):
    """Hiệu chỉnh Holm-Bonferroni (Holm, 1979)."""
    p = np.asarray(p, float); m = len(p); out = np.full(m, np.nan)
    ok = np.isfinite(p); idx = np.where(ok)[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]; run = 0.0; mm = len(order)
    for i, j in enumerate(order):
        run = max(run, (mm - i) * p[j]); out[j] = min(1.0, run)
    return out


def bh_fdr(p):
    """q-value Benjamini-Hochberg (1995)."""
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m); out[order] = np.minimum(q, 1.0)
    return out


def auc_safe(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    m = np.isfinite(s)
    if m.sum() < 2 or len(np.unique(y[m])) < 2:
        return np.nan
    return float(roc_auc_score(y[m], s[m]))


def cliffs_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    u = sst.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2.0 * u / (len(a) * len(b)) - 1.0


def mwu_p(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(sst.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def boot_auc_ci(y, s, n_boot=None, seed=0, groups=None, alpha=None):
    """CI bootstrap phân tầng của AUC. groups != None -> bootstrap THEO CỤM (bản ghi)."""
    n_boot = n_boot or cfg.n_boot; alpha = alpha or cfg.alpha
    y = np.asarray(y); s = np.asarray(s, float); rng = np.random.default_rng(seed)
    vals = []
    if groups is None:
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        if len(pos) == 0 or len(neg) == 0:
            return np.nan, np.nan
        for _ in range(n_boot):
            i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            vals.append(roc_auc_score(y[i], s[i]))
    else:
        groups = np.asarray(groups)
        gids, inv = np.unique(groups, return_inverse=True)
        members = [np.where(inv == k)[0] for k in range(len(gids))]
        glab = np.array([y[m[0]] for m in members])
        gp, gn = np.where(glab == 1)[0], np.where(glab == 0)[0]
        if len(gp) == 0 or len(gn) == 0:
            return np.nan, np.nan
        for _ in range(n_boot):
            pick = np.concatenate([rng.choice(gp, len(gp)), rng.choice(gn, len(gn))])
            i = np.concatenate([members[k] for k in pick])
            vals.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_boot_delta(y, s1, s2, n_boot=None, seed=0):
    """Delta AUC ghép cặp (cùng các bản ghi): CI bootstrap phân tầng + p hai phía."""
    n_boot = n_boot or cfg.n_boot
    y = np.asarray(y); s1 = np.asarray(s1, float); s2 = np.asarray(s2, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    d = []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    d = np.asarray(d)
    lo, hi = np.percentile(d, [100 * cfg.alpha / 2, 100 * (1 - cfg.alpha / 2)])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(lo), float(hi), float(min(1.0, max(p, 1.0 / n_boot)))


def _midrank(x):
    j = np.argsort(x, kind="mergesort"); z = x[j]; n = len(x); t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n); out[j] = t
    return out


def delong_cov(y, scores):
    """AUC và hiệp phương sai theo DeLong et al. (1988), cài đặt nhanh của Sun & Xu (2014)."""
    y = np.asarray(y).astype(int); pos = y == 1; m, n = int(pos.sum()), int((~pos).sum())
    k = len(scores); v10 = np.zeros((k, m)); v01 = np.zeros((k, n)); aucs = np.zeros(k)
    for r, s in enumerate(scores):
        s = np.asarray(s, float); X, Y = s[pos], s[~pos]
        tx, ty, tz = _midrank(X), _midrank(Y), _midrank(np.concatenate([X, Y]))
        aucs[r] = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        v10[r] = (tz[:m] - tx) / n
        v01[r] = 1.0 - (tz[m:] - ty) / m
    S = np.atleast_2d(np.cov(v10)) / m + np.atleast_2d(np.cov(v01)) / n
    return aucs, S


def delong_test(y, s1, s2):
    aucs, S = delong_cov(y, [s1, s2])
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    d = aucs[0] - aucs[1]
    if not np.isfinite(var) or var <= 1e-12:
        return float(d), (1.0 if abs(d) < 1e-12 else 0.0)
    return float(d), float(2 * sst.norm.sf(abs(d) / np.sqrt(var)))


def nb_ttest(d, n_train, n_test):
    """t-test hiệu chỉnh cho CV lặp (Nadeau & Bengio, 2003)."""
    d = np.asarray(d, float); d = d[np.isfinite(d)]; J = len(d)
    if J < 2 or np.var(d, ddof=1) == 0:
        return np.nan
    t = d.mean() / np.sqrt((1.0 / J + n_test / n_train) * np.var(d, ddof=1))
    return float(2 * sst.t.sf(abs(t), J - 1))


def bin_metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int); yhat = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return dict(acc=(tp + tn) / len(y), bacc=np.nanmean([sens, spec]), sens=sens, spec=spec,
                prec=prec, f1=f1, mcc=matthews_corrcoef(y, yhat) if len(np.unique(yhat)) > 1 else 0.0,
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def fmt_ci(v, lo, hi, d=3):
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]" if np.isfinite(v) else "n/a"


print("Tiện ích thống kê sẵn sàng: Holm, BH-FDR, bootstrap cụm, DeLong, Nadeau-Bengio.")
