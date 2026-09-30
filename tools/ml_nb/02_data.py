# %% [markdown]
# ## 2. Dữ liệu: kiểm kê, định danh bản ghi, chia tập / Data inventory and splitting
#
# - Nhãn lấy từ **tên thư mục** (`leak` → 1, `noleak` → 0).
# - Mỗi mã `Rxxx` là **một lần thu độc lập**; một lần thu có thể bị cắt thành nhiều tệp
#   (`R001_leak01.wav`, `R001_leak02.wav`...). Mọi khung của cùng một bản ghi luôn nằm cùng một
#   phía train/test → **không rò rỉ dữ liệu** giữa các khung chồng/kề nhau.
# - **Tập kiểm tra khoá (hold-out)**: `cfg.holdout_frac` số bản ghi (phân tầng theo lớp) được tách ra
#   ngay từ đầu và **không** tham gia bất kỳ quyết định nào (chọn điều kiện xử lý, số đặc trưng,
#   siêu tham số). Chỉ dùng đúng một lần ở mục 9.

# %%
# =========================================================
# 2.1 Dò thư mục dữ liệu + chế độ dữ liệu mô phỏng (khi không có dữ liệu thật)
# =========================================================
def _norm_token(s):
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.lower().replace("-", "_").replace(" ", "_")


def _key_regex(keys):
    ks = sorted({_norm_token(k) for k in keys}, key=len, reverse=True)
    return re.compile(r"(?:^|[_.])(?:" + "|".join(map(re.escape, ks)) + r")(?=$|[_.\d])")


_NOLEAK_RE, _LEAK_RE = _key_regex(cfg.noleak_keys), _key_regex(cfg.leak_keys)


def label_of_token(s):
    s = _norm_token(s)
    if _NOLEAK_RE.search(s):          # xét 'noleak' TRƯỚC 'leak'
        return 0
    if _LEAK_RE.search(s):
        return 1
    return None


def label_from_path(p):
    for tok in [p.parent.name, p.stem] + list(reversed(p.parts[:-2])):
        lab = label_of_token(tok)
        if lab is not None:
            return lab
    return None


for _t, _w in [("leak", 1), ("noleak", 0), ("No-Leak", 0), ("R001_leak01", 1), ("R12_noleak_03", 0),
               ("rò_rỉ", 1), ("leakage", None), ("field_rec_8k_V7.2", None)]:
    assert label_of_token(_t) == _w, f"bộ gán nhãn sai với '{_t}'"


def audio_files_in(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in cfg.audio_ext)


def find_data_root(explicit):
    if explicit and Path(explicit).exists():
        fs = audio_files_in(explicit)
        if {0, 1}.issubset({label_from_path(p) for p in fs}):
            return Path(explicit)
        print(f"!! {explicit} tồn tại nhưng không chứa đủ hai lớp leak/noleak.")
    elif explicit:
        print(f"!! cfg.data_root = {explicit} KHÔNG tồn tại -> tự dò trong /kaggle/input.")
    cands = []
    for base in [Path("/kaggle/input"), Path("./data")]:
        if not base.exists():
            continue
        for d in [base] + [p for p in base.rglob("*") if p.is_dir()]:
            try:
                labs = {label_of_token(k.name) for k in d.iterdir() if k.is_dir()}
            except Exception:
                continue
            if {0, 1}.issubset(labs):
                cands.append(d)
    if not cands:
        return None
    cands.sort(key=lambda p: (-len(p.parts), str(p)))
    return cands[0]


def synthesize_dataset(root, n_per_class=14, sr=16000, seed=7):
    """Dữ liệu MÔ PHỎNG để kiểm tra pipeline (KHÔNG dùng để báo cáo khoa học).
    Rò rỉ = ồn băng rộng 250-2500 Hz (+ đôi khi tiếng rít); nền = ồn hồng + hài 50 Hz + tiếng bơm.
    Cả hai lớp đều chứa nhiễu xung (va chạm gậy), đoạn giống tiếng nói và đôi khi bão hoà."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    rid = 0
    for cls in ("leak", "noleak"):
        (root / cls).mkdir(parents=True, exist_ok=True)
        for _ in range(n_per_class):
            rid += 1
            gain = 10 ** (rng.uniform(-6, 6) / 20)
            snr = rng.uniform(-26, -10)
            tilt = rng.uniform(-0.6, 0.6)                   # "dấu vân tay" vị trí thu: độ nghiêng phổ riêng
            flow = cls == "noleak" and rng.random() < 0.3   # ồn dòng chảy dải thấp gây nhầm
            f1, f2 = rng.uniform(250, 700), rng.uniform(1500, 2500)
            whistle = rng.random() < 0.3
            pump = rng.random() < 0.3
            for k in range(int(rng.integers(1, 4))):
                dur = rng.uniform(12, 24) if cfg.smoke_test else rng.uniform(20, 45)
                n = int(dur * sr); t = np.arange(n) / sr
                w = rng.standard_normal(n)
                W = np.fft.rfft(w); fr = np.fft.rfftfreq(n, 1 / sr); W[1:] /= np.sqrt(fr[1:]); W[0] = 0
                W[1:] *= (fr[1:] / 1000.0) ** tilt
                bg = np.fft.irfft(W, n); bg /= bg.std()
                bg += rng.uniform(0, 0.3) * sum(np.sin(2 * np.pi * 50 * h * t + rng.uniform(0, 6)) / h
                                                for h in range(1, 5))
                if pump:
                    f0 = rng.uniform(90, 160)
                    bg += 0.3 * sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 6))
                x = bg.copy()
                if flow:
                    fl = sps.sosfilt(sps.butter(4, [100, 700], btype="bandpass", fs=sr, output="sos"), rng.standard_normal(n))
                    x += fl / fl.std() * 10 ** (rng.uniform(-20, -10) / 20)
                if cls == "leak":
                    sos = sps.butter(4, [f1, f2], btype="bandpass", fs=sr, output="sos")
                    lk = sps.sosfilt(sos, rng.standard_normal(n)); lk /= lk.std()
                    if whistle:
                        lk += 0.4 * np.sin(2 * np.pi * rng.uniform(900, 1400) * t + 0.3 * np.sin(2 * np.pi * 0.5 * t))
                    x += lk * 10 ** (snr / 20)
                for _k in range(rng.poisson(1.5)):          # va chạm gậy nghe
                    i0 = int(rng.integers(0, n - sr // 4)); L = sr // 4
                    x[i0:i0 + L] += rng.uniform(6, 20) * np.exp(-np.arange(L) / (0.03 * sr)) * \
                        np.sin(2 * np.pi * rng.uniform(100, 600) * np.arange(L) / sr)
                for _k in range(rng.poisson(0.7)):          # đoạn giống tiếng nói
                    L = int(rng.uniform(0.8, 2.0) * sr); i0 = int(rng.integers(0, max(1, n - L)))
                    tt = np.arange(L) / sr; f0 = rng.uniform(110, 220)
                    v = sum(np.sin(2 * np.pi * f0 * h * tt) * np.exp(-((f0 * h - 700) / 900) ** 2) for h in range(1, 12))
                    x[i0:i0 + L] += rng.uniform(1.5, 4) * v * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * tt)) / (np.std(v) + 1e-9)
                x = 0.05 * gain * x / x.std()
                if rng.random() < 0.1:                      # bão hoà ngắn
                    i0 = int(rng.integers(0, n - sr // 2)); x[i0:i0 + sr // 4] *= 40
                sf.write(str(root / cls / f"R{rid:03d}_{cls}{k + 1:02d}.wav"), np.clip(x, -1, 1).astype(np.float32),
                         sr, subtype="PCM_16")
    return root


DATA_ROOT = find_data_root(cfg.data_root)
SYNTHETIC = DATA_ROOT is None
if SYNTHETIC:
    print("!! KHÔNG tìm thấy dữ liệu leak/noleak -> SYNTHETIC DEMO MODE (số liệu KHÔNG có giá trị khoa học).")
    DATA_ROOT = synthesize_dataset(OUT / "synthetic_data",
                                   n_per_class=int(os.environ.get("LEAK_SYNTH_N", 10 if cfg.smoke_test else 16)))
print("Thư mục dữ liệu:", DATA_ROOT)

# %%
# =========================================================
# 2.2 Bảng kiểm kê mức TỆP và mức BẢN GHI
# =========================================================
_REC_RE = re.compile(cfg.record_regex, re.IGNORECASE)
rows = []
for p in audio_files_in(DATA_ROOT):
    lab = label_from_path(p)
    if lab is None:
        continue
    try:
        info = sf.info(str(p))
    except Exception as e:
        print("  [bỏ qua]", p.name, type(e).__name__); continue
    m = _REC_RE.search(p.stem) or _REC_RE.search(str(p.parent))
    rows.append(dict(path=str(p), file_stem=p.stem, record_id=(m.group(1).upper() if m else p.stem),
                     rid_from_regex=bool(m), label=lab, cls="leak" if lab == 1 else "noleak",
                     duration_s=info.duration, sr_native=info.samplerate, channels=info.channels))
files = pd.DataFrame(rows).sort_values(["cls", "record_id", "file_stem"]).reset_index(drop=True)
assert len(files) > 0, "Không đọc được tệp âm thanh nào"

conf = files.groupby("record_id").label.nunique()
if (conf > 1).any():
    raise ValueError(f"Mã bản ghi mang HAI nhãn: {conf[conf > 1].index.tolist()} - sửa dữ liệu hoặc cfg.record_regex")
if (~files.rid_from_regex).any():
    print(f"!! {int((~files.rid_from_regex).sum())} tệp không khớp regex {cfg.record_regex} -> dùng tên tệp làm mã bản ghi.")

records = (files.groupby("record_id")
           .agg(cls=("cls", "first"), label=("label", "first"), n_files=("file_stem", "size"),
                duration_s=("duration_s", "sum"), sr=("sr_native", "min")).reset_index())

# ---- Tách tập KIỂM TRA KHOÁ ở mức bản ghi (phân tầng theo lớp) -----------------
if cfg.holdout_frac and cfg.holdout_frac > 0:
    _dev, _test = train_test_split(records.record_id.values, test_size=cfg.holdout_frac,
                                   stratify=records.label.values, random_state=cfg.split_seed)
    records["split"] = np.where(records.record_id.isin(_test), "test", "dev")
else:
    records["split"] = "dev"
files = files.merge(records[["record_id", "split"]], on="record_id")
HAS_TEST = (records.split == "test").any()

tab1 = (records.groupby(["split", "cls"])
        .agg(records=("record_id", "nunique"), files=("n_files", "sum"),
             total_min=("duration_s", lambda s: s.sum() / 60), mean_s=("duration_s", "mean"),
             min_s=("duration_s", "min"), max_s=("duration_s", "max")).round(1).reset_index())
display(Markdown("**Bảng 1. Kiểm kê dữ liệu mức bản ghi (đơn vị thống kê độc lập).**"))
display(tab1)
savetab(tab1, "T01_inventory")
print(f"Tệp: {len(files)} | Bản ghi: {len(records)} "
      f"({(records.label == 1).sum()} leak / {(records.label == 0).sum()} noleak) | "
      f"tổng {records.duration_s.sum() / 60:.1f} phút | sr gốc: {sorted(files.sr_native.unique())} Hz")
print(f"DEV: {(records.split == 'dev').sum()} bản ghi | TEST (khoá): {(records.split == 'test').sum()} bản ghi")
if files.sr_native.max() < 2 * cfg.f_hi:
    print(f"!! Tần số lấy mẫu gốc {files.sr_native.max()} Hz < 2 x f_hi: dải phân tích sẽ bị kẹp theo Nyquist.")

# %%
# =========================================================
# 2.3 Hình 1: Tổng quan dữ liệu
# =========================================================
fig, ax = plt.subplots(1, 3, figsize=(11, 3.0))
cnt = records.groupby(["cls", "split"]).size().unstack(fill_value=0).reindex(["noleak", "leak"])
bottom = np.zeros(2)
for j, sp in enumerate([s for s in ("dev", "test") if s in cnt.columns]):
    ax[0].bar(cnt.index, cnt[sp].values, bottom=bottom, width=0.55,
              color=[PAL[c] for c in cnt.index], alpha=1.0 if sp == "dev" else 0.45,
              edgecolor="white", linewidth=2, label=sp.upper())
    for i, v in enumerate(cnt[sp].values):
        if v:
            ax[0].text(i, bottom[i] + v / 2, str(v), ha="center", va="center", color="white", fontsize=8)
    bottom += cnt[sp].values
ax[0].set_title("(a) Recordings per class (DEV / TEST)"); ax[0].set_ylabel("recordings")
ax[0].legend(loc="upper left")
bins = np.linspace(0, records.duration_s.max() * 1.05, 16)
for c in ("noleak", "leak"):
    ax[1].hist(records.loc[records.cls == c, "duration_s"], bins=bins, color=PAL[c], alpha=0.65,
               label=c, edgecolor="white", linewidth=1)
ax[1].set_title("(b) Recording duration"); ax[1].set_xlabel("s"); ax[1].legend()
fr = records.groupby(["n_files", "cls"]).size().unstack(fill_value=0).reindex(columns=["noleak", "leak"], fill_value=0)
xx = np.arange(len(fr))
for j, c in enumerate(["noleak", "leak"]):
    ax[2].bar(xx + (j - 0.5) * 0.36, fr[c].values, width=0.34, color=PAL[c], label=c)
ax[2].set_xticks(xx); ax[2].set_xticklabels(fr.index); ax[2].set_xlabel("files per recording")
ax[2].set_title("(c) Files per recording"); ax[2].legend()
fig.tight_layout()
savefig(fig, "Fig01_dataset_overview", "Dataset overview at recording level.")
