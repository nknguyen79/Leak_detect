# %% [markdown]
# ## 3. Chuỗi xử lý tín hiệu / Signal-processing chain
#
# Bảy **điều kiện tín hiệu** được so sánh trong cùng một khung đánh giá (mục 7 — câu hỏi RQ2):
#
# | Mã | Thành phần | Mục đích |
# |---|---|---|
# | `C0_RAW` | chỉ giải mã + (nếu cần) tái lấy mẫu | **tín hiệu thô**, không xử lý gì |
# | `C1_BPF` | khử DC + band-pass Butterworth pha-không 20–7000 Hz + notch 50 Hz × 6 hài | loại trôi nền/hạ âm, nhiễu điện lưới, nhiễu gần Nyquist |
# | `C2_TR` | C1 + kẹp xung mẫu (k·σ_MAD) + **loại khung bất thường**: bão hoà, mất tín hiệu, sự kiện năng lượng (va chạm gậy nghe), kurtosis cao | loại nhiễu xung / tín hiệu bất thường |
# | `C3_IR` | C2 + **loại khung không liên quan**: phổ lệch khỏi phổ nền của chính tệp (LSD), khung hữu thanh (tiếng nói, chim, động cơ gián đoạn) | loại âm thanh không liên quan tới rò rỉ |
# | `C4_MBG` | C3 + **khử nhiễu không dừng** bằng kẹp biên độ STFT về *trung vị theo thời gian* (giữ thành phần dừng) | khử nhiễu **bảo toàn** tiếng rò rỉ (đề xuất) |
# | `C5_SS` | C3 + **trừ phổ dừng** (Boll 1979; Berouti 1979) với phổ nhiễu ước lượng từ chính tệp | khử nhiễu "kinh điển" kiểu tiếng nói |
# | `C6_WT` | C3 + **khử nhiễu wavelet** soft-threshold VisuShrink (Donoho & Johnstone 1994) | khử nhiễu phổ biến trong các bài báo AE/rò rỉ |
#
# **Nguyên tắc quan trọng**
#
# 1. Mọi quy tắc loại bỏ đều **không nhìn nhãn** (label-blind) và dùng **ngưỡng tương đối so với chính
#    tệp đó** (trung vị/MAD) → áp dụng giống hệt cho leak và noleak, không tạo khác biệt giả giữa hai lớp.
# 2. Tiếng rò rỉ là **ồn băng rộng, dừng** (stationary). Các bộ khử nhiễu kiểu tiếng nói (C5, C6) coi mọi
#    thành phần dừng là "nhiễu nền" → **có nguy cơ xoá chính tín hiệu rò rỉ**. C4 làm điều ngược lại: giữ
#    phần dừng, chỉ cắt phần *nhô lên nhất thời* (tiếng nói, xe, va chạm). Thí nghiệm E2 kiểm chứng giả
#    thuyết này bằng số liệu.
# 3. Mỗi tệp luôn giữ ≥ `min_keep_frac` số khung → không bản ghi nào biến mất khỏi đánh giá.

# %%
# =========================================================
# 3.1 Các khối xử lý / Processing blocks
# =========================================================
from numpy.lib.stride_tricks import sliding_window_view


def load_raw(path, sr):
    x, sr0 = sf.read(str(path), always_2d=True, dtype="float32")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).mean(axis=1).astype(np.float64)
    if sr0 != sr:
        g = math.gcd(int(sr0), int(sr))
        x = sps.resample_poly(x, sr // g, sr0 // g)
    return x


def interp_weights(f, edges):
    """Trọng số nội suy tuyến tính để lấy tích phân luỹ tích tại các biên dải (dùng chung cho nhiều khung)."""
    e = np.clip(np.asarray(edges, float), f[0], f[-1])
    i1 = np.clip(np.searchsorted(f, e), 1, len(f) - 1); i0 = i1 - 1
    w = (e - f[i0]) / (f[i1] - f[i0])
    return i0, i1, w


def band_powers(f, P, lo, hi, _cache={}):
    """Công suất trong các dải [lo, hi] từ PSD (tích phân hình thang + nội suy biên). P: (..., nfreq)."""
    key = (len(f), float(f[1] - f[0]), tuple(np.round(lo, 3)), tuple(np.round(hi, 3)))
    if key not in _cache:
        _cache[key] = (interp_weights(f, lo), interp_weights(f, hi))
    (a0, a1, aw), (b0, b1, bw) = _cache[key]
    df = f[1] - f[0]
    cum = np.concatenate([np.zeros(P.shape[:-1] + (1,)), np.cumsum(0.5 * (P[..., 1:] + P[..., :-1]) * df, axis=-1)], axis=-1)
    ca = cum[..., a0] * (1 - aw) + cum[..., a1] * aw
    cb = cum[..., b0] * (1 - bw) + cum[..., b1] * bw
    return np.maximum(cb - ca, 0.0)


THIRD_OCT_NOMINAL = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
                     1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000]


def third_octave_bands(f_lo, f_hi):
    c = 1000.0 * 2 ** (np.arange(-16, 10) / 3)
    lo, hi = c * 2 ** (-1 / 6), c * 2 ** (1 / 6)
    m = (lo >= f_lo * 0.9) & (hi <= f_hi)
    return lo[m], hi[m], [THIRD_OCT_NOMINAL[i] for i in np.where(m)[0]]


class SignalChain:
    """Các bước xử lý của một tệp. Không dùng nhãn, không dùng thống kê chéo giữa các tệp."""

    def __init__(self, sr, P):
        self.sr, self.P = sr, P
        nyq = sr / 2
        self.f_hi = min(P["f_hi"], 0.95 * nyq)
        lp = min(P["lp_hz"], 0.95 * nyq)
        self.sos_bp = sps.butter(P["filt_order"], [P["hp_hz"], lp], btype="bandpass", fs=sr, output="sos")
        self.notches = [sps.iirnotch(h * P["notch_hz"], P["notch_q"], fs=sr)
                        for h in range(1, P["notch_harmonics"] + 1) if h * P["notch_hz"] < 0.95 * nyq]
        self.sos_voice = sps.butter(4, [60, min(1000, 0.9 * nyq)], btype="bandpass", fs=sr, output="sos")
        self.F = int(round(P["frame_sec"] * sr)); self.H = int(round(P["hop_sec"] * sr))
        self.B = int(round(P["block_ms"] * sr / 1000))
        self.lsd_lo, self.lsd_hi, _ = third_octave_bands(P["f_lo"], self.f_hi)

    # ---- C1 -------------------------------------------------------------
    def condition(self, x):
        y = sps.sosfiltfilt(self.sos_bp, x - x.mean())
        for b, a in self.notches:
            y = sps.filtfilt(b, a, y)
        return y

    # ---- C2: kẹp xung ở mức mẫu --------------------------------------------
    def spike_clip(self, x):
        s = 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-12
        k = self.P["spike_k"] * s
        return np.clip(x, -k, k), float((np.abs(x) > k).mean())

    def frame_starts(self, n):
        return np.arange(0, n - self.F + 1, self.H) if n >= self.F else np.array([], int)

    # ---- C2/C3: thống kê chất lượng từng khung + mặt nạ ----------------------
    def qc_frames(self, xr, x1, starts):
        P, F, B, sr = self.P, self.F, self.B, self.sr
        nf = len(starts)
        idx = starts[:, None] + np.arange(F)[None, :]
        fr_raw, fr1 = xr[idx], x1[idx]
        clip_frac = (np.abs(fr_raw) >= P["clip_level"]).mean(1)
        zero_frac = (fr_raw == 0).mean(1)
        rms_raw = np.sqrt((fr_raw ** 2).mean(1))
        kurt = sst.kurtosis(fr1, axis=1, fisher=False)
        # (a) sự kiện năng lượng: khối 50 ms mạnh nhất trong khung so với trung vị cả tệp
        nb = len(x1) // B
        blk = 10 * np.log10((x1[:nb * B].reshape(nb, B) ** 2).mean(1) + 1e-20)
        med_b = np.median(blk); mad_b = 1.4826 * np.median(np.abs(blk - med_b)) + 1e-9
        nbf = max(1, F // B)
        blkmax = np.array([blk[s // B: s // B + nbf].max() for s in starts])
        e_excess = blkmax - med_b; e_z = e_excess / mad_b
        # (b) khoảng cách log-phổ (1/3 octave) tới phổ trung vị của tệp
        f, Pw = sps.welch(fr1, fs=sr, nperseg=min(1024, F), axis=-1)
        Ldb = 10 * np.log10(band_powers(f, Pw, self.lsd_lo, self.lsd_hi) + 1e-20)
        lsd = np.sqrt(((Ldb - np.median(Ldb, 0)) ** 2).mean(1))
        lsd_z = (lsd - np.median(lsd)) / (1.4826 * np.median(np.abs(lsd - np.median(lsd))) + 1e-9)
        # (c) mức hữu thanh: đỉnh ACF chuẩn hoá (không chệch) ở trễ 2-14 ms, khối 40 ms bước 20 ms
        xv = sps.sosfiltfilt(self.sos_voice, x1)
        L, step = int(0.04 * sr), int(0.02 * sr)
        sub = sliding_window_view(xv, L)[::step]
        sub = sub - sub.mean(1, keepdims=True)
        nfft = 1 << int(np.ceil(np.log2(2 * L)))
        r = np.fft.irfft(np.abs(np.fft.rfft(sub, nfft, axis=1)) ** 2, nfft, axis=1)[:, :L]
        r = r / (r[:, :1] + 1e-20) * (L / (L - np.arange(L)))
        voiced = r[:, int(sr / 500): int(sr / 70)].max(1) > P["voicing_thr"]
        centers = np.arange(len(sub)) * step + L // 2
        vfrac = np.array([voiced[(centers >= s) & (centers < s + F)].mean()
                          if ((centers >= s) & (centers < s + F)).any() else 0.0 for s in starts])
        q = pd.DataFrame(dict(clip_frac=clip_frac, zero_frac=zero_frac, rms_raw=rms_raw, kurt=kurt,
                              e_excess_db=e_excess, e_z=e_z, lsd_db=lsd, lsd_z=lsd_z, voiced_frac=vfrac))
        q["f_clip"] = clip_frac > P["frame_clip_frac"]
        q["f_silence"] = (rms_raw < 1e-5) | (zero_frac > 0.2)
        q["f_event"] = (e_z > P["z_thr"]) & (e_excess > P["event_db"])
        q["f_impulsive"] = kurt > P["kurt_thr"]
        q["f_spectral"] = (lsd_z > P["lsd_z_thr"]) & (lsd > P["lsd_min_db"])
        q["f_voiced"] = (vfrac > P["voicing_frac_thr"]) & (vfrac - np.median(vfrac) > P["voicing_excess"])
        abn = (q.f_clip | q.f_silence | q.f_event | q.f_impulsive).values
        irr = (q.f_spectral | q.f_voiced).values
        score = (10.0 * (q.f_clip | q.f_silence).values + np.maximum(0, e_z) / P["z_thr"]
                 + np.maximum(0, kurt - 3) / max(P["kurt_thr"] - 3, 1e-6)
                 + np.maximum(0, lsd_z) / P["lsd_z_thr"] + vfrac)
        keep2 = self._guard(~abn, score)
        keep3 = self._guard(keep2 & ~irr, score, prefer=keep2)
        q["keep_C2"], q["keep_C3"] = keep2, keep3
        q["guard_C2"] = keep2 & abn; q["guard_C3"] = keep3 & (abn | irr)
        return q

    def _guard(self, keep, score, prefer=None):
        keep = keep.copy(); n = len(keep)
        need = min(n, max(self.P["min_keep_frames"], int(np.ceil(self.P["min_keep_frac"] * n))))
        if keep.sum() < need:
            pref = np.zeros(n) if prefer is None else (~prefer).astype(float)
            for i in np.lexsort((score, pref)):
                if keep.sum() >= need:
                    break
                keep[i] = True
        return keep

    # ---- C4: khử nhiễu KHÔNG DỪNG, bảo toàn thành phần dừng -------------------
    def denoise_mbg(self, x):
        Z = librosa.stft(x, n_fft=self.P["stft_nfft"], hop_length=self.P["stft_hop"], window="hann")
        mag = np.abs(Z); med = np.median(mag, axis=1, keepdims=True)
        g = np.minimum(1.0, self.P["mbg_beta"] * med / (mag + 1e-12))
        return librosa.istft(Z * g, hop_length=self.P["stft_hop"], window="hann", length=len(x))

    # ---- C5: trừ phổ dừng (Boll 1979; Berouti et al. 1979), nhiễu = thống kê cực tiểu ----
    def denoise_ss(self, x):
        Z = librosa.stft(x, n_fft=self.P["stft_nfft"], hop_length=self.P["stft_hop"], window="hann")
        Pz = np.abs(Z) ** 2; q = self.P["ss_quantile"]
        noise = np.quantile(Pz, q, axis=1, keepdims=True) / (-np.log(1 - q))   # bù chệch phân vị (phân bố mũ)
        G = np.sqrt(np.maximum(1.0 - self.P["ss_alpha"] * noise / (Pz + 1e-20), self.P["ss_floor"] ** 2))
        return librosa.istft(Z * G, hop_length=self.P["stft_hop"], window="hann", length=len(x))

    # ---- C6: wavelet soft-threshold, ngưỡng phổ quát (VisuShrink) -------------
    def denoise_wt(self, x):
        w = pywt.Wavelet(self.P["wt_wavelet"])
        L = max(1, min(self.P["wt_level"], pywt.dwt_max_level(len(x), w.dec_len)))
        c = pywt.wavedec(x, w, level=L, mode="symmetric")
        sigma = np.median(np.abs(c[-1])) / 0.6745
        thr = sigma * np.sqrt(2 * np.log(len(x)))
        c = [c[0]] + [pywt.threshold(ci, thr, mode="soft") for ci in c[1:]]
        return pywt.waverec(c, w, mode="symmetric")[:len(x)]

    def signals(self, xr):
        """Trả về tín hiệu của mọi điều kiện + bảng QC khung + QC tệp."""
        P = self.P
        x1 = self.condition(xr)
        x2, spike_frac = self.spike_clip(x1)
        starts = self.frame_starts(len(xr))
        sig = {"C0_RAW": xr, "C1_BPF": x1, "C2_TR": x2, "C3_IR": x2}
        conds = P["conditions"]
        if "C4_MBG" in conds: sig["C4_MBG"] = self.denoise_mbg(x2)
        if "C5_SS" in conds: sig["C5_SS"] = self.denoise_ss(x2)
        if "C6_WT" in conds: sig["C6_WT"] = self.denoise_wt(x2)
        q = self.qc_frames(xr, x1, starts) if len(starts) else pd.DataFrame()
        return sig, starts, q, spike_frac

    def mask_for(self, cond, q):
        if cond in ("C0_RAW", "C1_BPF"):
            return np.ones(len(q), bool)
        if cond == "C2_TR":
            return q.keep_C2.values
        return q.keep_C3.values


def file_qc(xr, sr, P):
    """QC mức tệp (không nhìn nhãn): bão hoà, mất tín hiệu, DC, mức, SNR ước lượng."""
    n = len(xr); B = int(round(P["block_ms"] * sr / 1000)); nb = max(1, n // B)
    blk = 10 * np.log10((xr[:nb * B].reshape(nb, B) ** 2).mean(1) + 1e-20) if n >= B else np.array([-200.0])
    rms = np.sqrt(np.mean(xr ** 2)) + 1e-20
    d = dict(duration_s=n / sr, clip_frac=float((np.abs(xr) >= P["clip_level"]).mean()),
             zero_frac=float((xr == 0).mean()), dc_ratio=float(abs(xr.mean()) / rms),
             rms_dbfs=float(20 * np.log10(rms)), snr_proxy_db=float(np.percentile(blk, 95) - np.percentile(blk, 10)))
    reasons = []
    if n < int(P["frame_sec"] * sr): reasons.append("too_short")
    if d["clip_frac"] > P["qc_max_clip_frac"]: reasons.append("clipping")
    if d["zero_frac"] > P["qc_max_zero_frac"]: reasons.append("dropout")
    d["excluded"] = bool(reasons); d["reason"] = ",".join(reasons)
    return d


SR = int(cfg.target_sr) if cfg.target_sr else int(files.sr_native.max())
PARAMS = asdict(cfg)
CHAIN = SignalChain(SR, PARAMS)
print(f"sr = {SR} Hz | khung {CHAIN.F} mẫu ({cfg.frame_sec} s), bước {CHAIN.H} | dải phân tích "
      f"{cfg.f_lo:.0f}-{CHAIN.f_hi:.0f} Hz | {len(CHAIN.notches)} notch | {len(CHAIN.lsd_lo)} dải 1/3 octave cho LSD")

# %%
# =========================================================
# 3.2 Hình 2-3: Minh hoạ chuỗi xử lý trên một bản ghi mỗi lớp
#     (các khung bị loại được tô màu theo lý do; phổ trung bình sau từng bộ khử nhiễu)
# =========================================================
def _pick_example(cls):
    cand = files[(files.cls == cls) & (files.duration_s >= 3 * cfg.frame_sec)]
    best, best_n = cand.iloc[0], -1
    for _, r in cand.head(10).iterrows():          # ưu tiên tệp có vài khung bị loại (minh hoạ rõ)
        xr = load_raw(r.path, SR); _, st_, q, _ = CHAIN.signals(xr)
        n_rej = int((~q.keep_C3).sum()) if len(q) else 0
        if 0 < n_rej <= len(q) // 2 and n_rej > best_n:
            best, best_n = r, n_rej
    return best


REASON_COLOR = {"f_event": CAT[3], "f_impulsive": CAT[7], "f_clip": CAT[6], "f_silence": "#9a9993",
                "f_spectral": CAT[4], "f_voiced": CAT[2]}
EX = {c: _pick_example(c) for c in ("leak", "noleak")}
fig = plt.figure(figsize=(12, 8.6))
gs = gridspec.GridSpec(4, 2, hspace=0.55, wspace=0.12, height_ratios=[0.8, 1, 1, 1])
psd_demo = {}
for j, c in enumerate(("noleak", "leak")):
    r = EX[c]; xr = load_raw(r.path, SR); sig, starts, q, _ = CHAIN.signals(xr)
    t = np.arange(len(xr)) / SR
    a0 = fig.add_subplot(gs[0, j]); a0.plot(t, xr, lw=0.3, color=INK2)
    for i, s in enumerate(starts):
        for rsn, col in REASON_COLOR.items():
            if q[rsn].iloc[i]:
                a0.axvspan(s / SR, (s + CHAIN.F) / SR, color=col, alpha=0.25, lw=0)
                break
    a0.set_title(f"({'ab'[j]}) {c}: {r.file_stem} - raw waveform, rejected frames shaded")
    a0.set_xlim(0, t[-1]); a0.set_xlabel("time (s)")
    for i, (cond, lab) in enumerate([("C1_BPF", "C1 band-pass + notch"), ("C4_MBG", "C4 median-background (proposed)"),
                                     ("C5_SS", "C5 spectral subtraction")]):
        if cond not in sig:
            continue
        ax = fig.add_subplot(gs[i + 1, j])
        S = librosa.amplitude_to_db(np.abs(librosa.stft(sig[cond], n_fft=1024, hop_length=256)), ref=np.max(np.abs(librosa.stft(sig["C1_BPF"], n_fft=1024, hop_length=256))))
        librosa.display.specshow(S, sr=SR, hop_length=256, x_axis="time", y_axis="log", ax=ax, cmap="magma", vmin=-80, vmax=0, rasterized=True)
        ax.set_ylim(cfg.f_lo, CHAIN.f_hi); ax.set_title(f"{lab}"); ax.set_xlabel("")
    psd_demo[c] = {k: sps.welch(v, fs=SR, nperseg=4096)[1] for k, v in sig.items()}
handles = [plt.Rectangle((0, 0), 1, 1, color=v, alpha=0.4) for v in REASON_COLOR.values()]
fig.legend(handles, [k[2:] for k in REASON_COLOR], loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.995))
savefig(fig, "Fig02_processing_examples", "Raw waveform with rejected frames and spectrograms after C1/C4/C5.")

fw = np.fft.rfftfreq(4096, 1 / SR)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
for j, c in enumerate(("noleak", "leak")):
    for cond in [k for k in cfg.conditions if k in psd_demo[c] and k not in ("C2_TR", "C3_IR")]:
        ax[j].semilogx(fw[1:], 10 * np.log10(psd_demo[c][cond][1:] + 1e-20), color=COND_COLOR[cond], lw=1.2, label=cond)
    freq_axis(ax[j], cfg.f_lo, SR / 2); ax[j].set_title(f"({'ab'[j]}) {c}: long-term PSD after each denoiser")
ax[0].set_ylabel("PSD (dB re 1/Hz)"); ax[1].legend(ncol=2)
fig.tight_layout()
savefig(fig, "Fig03_denoiser_psd", "Long-term PSD of the example files after each processing condition.")
