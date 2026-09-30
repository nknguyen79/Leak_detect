# %% [markdown]
# ## 4. Trích xuất đặc trưng thủ công / Handcrafted feature extraction
#
# Mỗi khung 2 s được mô tả bởi **~170 đặc trưng** thuộc 7 nhóm, đều đã được dùng trong các công trình về
# nhận dạng rò rỉ bằng âm thanh/rung động và phân loại âm thanh môi trường:
#
# | Nhóm | Đặc trưng | Ý nghĩa vật lý với rò rỉ | Tham chiếu |
# |---|---|---|---|
# | **Time-domain** | RMS, đỉnh, crest/impulse/shape/clearance factor, skewness, kurtosis, ZCR, Hjorth, entropy năng lượng, độ ổn định RMS khối, đường bao (CV, kurtosis), thời gian tương quan ACF, đỉnh chu kỳ ACF | ồn rò rỉ dừng, gần Gauss (kurtosis ≈ 3, crest thấp, đường bao phẳng); nhiễu xung thì ngược lại | Randall 2011; Hjorth 1970; Giannakopoulos 2015; Fares et al. 2023 |
# | **Spectral shape** | centroid, spread, skewness, kurtosis, entropy, flatness, roll-off 50/85/95 %, độ dốc dB/octave, decrease, crest, tần số đỉnh, số đỉnh âm sắc, flux, spectral contrast (7 octave), spectral kurtosis | rò rỉ làm phổ "trắng" hơn/dịch lên trong dải phát xạ tia nước; nhiễu điện/bơm tạo đỉnh âm sắc | Peeters 2004; Jiang et al. 2002; Antoni 2006 |
# | **Band energy** | năng lượng tương đối 25 dải 1/3 octave (25 Hz–6,3 kHz), 3 tỉ số dải, mức tuyệt đối 7 dải octave | **trả lời trực tiếp RQ1**: dải nào mang thông tin | Hunaidi & Chu 1999; Almeida et al. 2014 |
# | **Cepstral** | MFCC 1–13 (mean, std), MFCC0, độ biến thiên delta, LFCC 1–13 | hình bao phổ, bất biến với hệ số khuếch đại (trừ c0) | Davis & Mermelstein 1980; Zhou et al. 2011 |
# | **Wavelet** | năng lượng tương đối DWT (db4, 7 mức) và WPD (db4, mức 4, 16 dải 500 Hz), entropy wavelet | phân bố năng lượng đa phân giải — nhóm đặc trưng phổ biến nhất trong các bài báo rò rỉ ống | Mallat 1989; Coifman & Wickerhauser 1992; Xu et al. 2021 |
# | **LPC** | 10 hệ số dự báo tuyến tính + sai số dự báo | mô hình cộng hưởng của ống/gậy nghe | Makhoul 1975; Cody, Dey & Narasimhan 2020 |
# | **Complexity & modulation** | permutation entropy (trễ 1, 4), chiều fractal Higuchi, Katz, năng lượng điều biến đường bao 0,5–4 / 4–16 / 16–64 Hz | ồn rò rỉ phức tạp cao, không điều biến; tiếng nói điều biến ~4 Hz | Bandt & Pompe 2002; Higuchi 1988; Katz 1988; Atlas & Shamma 2003 |
#
# Đặc trưng **phụ thuộc mức tín hiệu** (RMS, đỉnh, mức octave tuyệt đối, MFCC0) được đánh dấu `level_dep`
# để kiểm tra riêng ở mục 10 (nếu hệ số khuếch đại của micro thay đổi giữa các lần thu, chúng có thể là confound).

# %%
# =========================================================
# 4.1 Bộ trích xuất đặc trưng / Feature extractor
# =========================================================
def linear_filterbank(freqs, n, f_lo, f_hi):
    pts = np.linspace(f_lo, f_hi, n + 2); fb = np.zeros((n, len(freqs)))
    for i in range(n):
        l, c, r = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.maximum(0, np.minimum((freqs - l) / (c - l), (r - freqs) / (r - c)))
    return fb


def perm_entropy(x, order=5, delay=1):
    emb = sliding_window_view(x, (order - 1) * delay + 1)[:, ::delay]
    codes = (np.argsort(emb, axis=1) * (order ** np.arange(order))).sum(1)
    _, cnt = np.unique(codes, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log(p)).sum() / np.log(math.factorial(order)))


def higuchi_fd(x, kmax=10):
    N = len(x); ks = np.arange(1, kmax + 1); lk = []
    for k in ks:
        Lm = [np.abs(np.diff(x[m::k])).sum() * (N - 1) / ((len(x[m::k]) - 1) * k) / k
              for m in range(k) if len(x[m::k]) > 1]
        lk.append(np.mean(Lm))
    return float(np.polyfit(np.log(1.0 / ks), np.log(np.asarray(lk) + 1e-20), 1)[0])


def katz_fd(x):
    dists = np.abs(np.diff(x)); L = dists.sum() + 1e-20
    a = np.log10(L / (dists.mean() + 1e-20)); dmax = np.max(np.abs(x - x[0])) + 1e-20
    return float(a / (a + np.log10(dmax / L)))


def _hz(v):
    return f"{v:g}" if v < 100 else f"{int(round(v))}"


class FeatureExtractor:
    def __init__(self, sr, P):
        self.sr = sr; nyq = sr / 2
        self.f_lo = P["f_lo"]; self.f_hi = min(P["f_hi"], 0.95 * nyq)
        self.nper = int(2 ** round(np.log2(0.256 * sr)))
        self.fw = np.fft.rfftfreq(self.nper, 1 / sr)
        self.m_an = (self.fw >= self.f_lo) & (self.fw <= self.f_hi)
        self.to_lo, self.to_hi, self.to_nom = third_octave_bands(self.f_lo, self.f_hi)
        oc = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000.0])
        self.oc = oc[oc * np.sqrt(2) <= self.f_hi]
        e = self.f_lo * 2 ** (np.arange(0, 400) / 6.0); e = e[e <= self.f_hi * 1.0001]
        self.fb_lo, self.fb_hi = e[:-1], e[1:]
        self.n_fft = int(2 ** round(np.log2(0.064 * sr))); self.hop = self.n_fft // 2
        self.sfreq = np.fft.rfftfreq(self.n_fft, 1 / sr)
        self.mel_fb = librosa.filters.mel(sr=sr, n_fft=self.n_fft, n_mels=40, fmin=self.f_lo, fmax=self.f_hi)
        self.lin_fb = linear_filterbank(self.sfreq, 40, self.f_lo, self.f_hi)
        ed = [0] + [62.5 * 2 ** i for i in range(7)]
        self.sc_names = [f"sc{i}_{_hz(ed[i])}-{_hz(ed[i + 1]) if i < 6 else 'nyq'}Hz" for i in range(7)]
        self.sk_nfft = int(2 ** round(np.log2(0.016 * sr))); self.fsk = np.fft.rfftfreq(self.sk_nfft, 1 / sr)
        self.dwt_L = max(2, int(np.floor(np.log2(nyq / 62.5))))
        self.dwt_names = [f"wd_A{self.dwt_L}_0-{_hz(nyq / 2 ** self.dwt_L)}Hz"] + \
            [f"wd_D{j}_{_hz(nyq / 2 ** j)}-{_hz(nyq / 2 ** (j - 1))}Hz" for j in range(self.dwt_L, 0, -1)]
        wbw = nyq / 16
        self.wp_names = [f"wp{i:02d}_{_hz(i * wbw)}-{_hz((i + 1) * wbw)}Hz" for i in range(16)]

    def __call__(self, x):
        sr = self.sr; x = np.asarray(x, np.float64); N = len(x); eps = 1e-12; d = {}
        xc = x - x.mean(); ax = np.abs(x)
        rms = np.sqrt(np.mean(x ** 2)) + eps; peak = ax.max() + eps; mav = ax.mean() + eps; sd = xc.std() + eps
        # ---------------- Time-domain ----------------
        d["td_rms_db"] = 20 * np.log10(rms)
        d["td_peak_db"] = 20 * np.log10(peak)
        d["td_crest"] = peak / rms
        d["td_impulse_factor"] = peak / mav
        d["td_shape_factor"] = rms / mav
        d["td_clearance_factor"] = peak / (np.mean(np.sqrt(ax)) ** 2 + eps)
        d["td_skewness"] = np.mean(xc ** 3) / sd ** 3
        d["td_kurtosis"] = np.mean(xc ** 4) / sd ** 4
        d["td_zcr"] = np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))
        dx = np.diff(xc); ddx = np.diff(dx)
        v0, v1, v2 = xc.var() + eps, dx.var() + eps, ddx.var() + eps
        mob = np.sqrt(v1 / v0)
        d["td_hjorth_mobility"] = mob
        d["td_hjorth_complexity"] = np.sqrt(v2 / v1) / mob
        n20 = N // 20; eb = (xc[:n20 * 20].reshape(20, n20) ** 2).sum(1) + eps; pe = eb / eb.sum()
        d["td_energy_entropy"] = -(pe * np.log(pe)).sum() / np.log(20)
        B = int(0.05 * sr); nb = N // B
        d["td_blockrms_std_db"] = (10 * np.log10((xc[:nb * B].reshape(nb, B) ** 2).mean(1) + eps)).std()
        env = np.abs(sps.hilbert(xc))
        d["td_env_cv"] = env.std() / (env.mean() + eps)
        d["td_env_kurtosis"] = sst.kurtosis(env, fisher=False)
        nfft = 1 << int(np.ceil(np.log2(2 * N)))
        r = np.fft.irfft(np.abs(np.fft.rfft(xc, nfft)) ** 2, nfft)[: int(0.05 * sr)]; r = r / (r[0] + eps)
        b_ = np.where(r < 1 / np.e)[0]; z_ = np.where(r < 0)[0]
        d["td_acf_decay_ms"] = (b_[0] if len(b_) else len(r)) / sr * 1000
        d["td_acf_zero_ms"] = (z_[0] if len(z_) else len(r)) / sr * 1000
        d["hm_acf_peak"] = r[int(sr / 500): int(sr / 50)].max()
        # ---------------- Spectral shape (Welch) ----------------
        f, Pxx = sps.welch(x, fs=sr, nperseg=self.nper, noverlap=self.nper // 2, window="hann", detrend="constant")
        fb, Pb = f[self.m_an], Pxx[self.m_an] + eps
        pn = Pb / Pb.sum(); c = (fb * pn).sum(); spread = np.sqrt(((fb - c) ** 2 * pn).sum()) + eps
        d["sp_centroid"] = c
        d["sp_spread"] = spread
        d["sp_skewness"] = ((fb - c) ** 3 * pn).sum() / spread ** 3
        d["sp_kurtosis"] = ((fb - c) ** 4 * pn).sum() / spread ** 4
        d["sp_entropy"] = -(pn * np.log(pn)).sum() / np.log(len(pn))
        d["sp_flatness"] = np.exp(np.mean(np.log(Pb))) / np.mean(Pb)
        cum = np.cumsum(pn)
        for q in (50, 85, 95):
            d[f"sp_rolloff{q}"] = fb[min(np.searchsorted(cum, q / 100), len(fb) - 1)]
        ldb = 10 * np.log10(Pb)
        d["sp_slope_db_oct"] = np.polyfit(np.log2(fb), ldb, 1)[0]
        d["sp_decrease"] = ((Pb[1:] - Pb[0]) / np.arange(1, len(Pb))).sum() / Pb[1:].sum()
        d["sp_crest_db"] = 10 * np.log10(Pb.max() / Pb.mean())
        d["sp_peak_freq"] = fb[np.argmax(Pb)]
        prom = ldb - sps.medfilt(ldb, 31)
        d["sp_n_tonal_peaks"] = len(sps.find_peaks(prom, height=10.0, distance=3)[0])
        d["sp_tonality_db"] = prom.max()
        # ---------------- Band energy ----------------
        E_tot = band_powers(f, Pxx, np.array([self.f_lo]), np.array([self.f_hi]))[0] + eps
        for nom, v in zip(self.to_nom, band_powers(f, Pxx, self.to_lo, self.to_hi)):
            d[f"be_{nom:g}Hz"] = 10 * np.log10(v / E_tot + eps)
        rr = band_powers(f, Pxx, np.array([self.f_lo, 1000.0, 100.0, 1000.0]),
                         np.array([1000.0, self.f_hi, 800.0, min(3000.0, self.f_hi)])) + eps
        d["br_below1k_vs_above_db"] = 10 * np.log10(rr[0] / rr[1])
        d["br_100-800Hz_frac"] = rr[2] / E_tot
        d["br_1k-3kHz_frac"] = rr[3] / E_tot
        for c_, v in zip(self.oc, band_powers(f, Pxx, self.oc / np.sqrt(2), self.oc * np.sqrt(2))):
            d[f"bl_oct{int(c_)}Hz_db"] = 10 * np.log10(v + eps)
        fine = band_powers(f, Pxx, self.fb_lo, self.fb_hi)
        # ---------------- Cepstral + flux + contrast (STFT 64 ms) ----------------
        S = np.abs(librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop, center=False, window="hann"))
        Pw = S ** 2 + eps
        mf = dct(10 * np.log10(self.mel_fb @ Pw + eps), type=2, axis=0, norm="ortho")[:14]
        d["mfcc0_mean"] = mf[0].mean()
        for i in range(1, 14):
            d[f"mfcc{i}_mean"] = mf[i].mean()
        for i in range(1, 14):
            d[f"mfcc{i}_std"] = mf[i].std()
        d["mfcc_delta_std"] = np.diff(mf[1:], axis=1).std(1).mean()
        lf = dct(10 * np.log10(self.lin_fb @ Pw + eps), type=2, axis=0, norm="ortho")[:14]
        for i in range(1, 14):
            d[f"lfcc{i}_mean"] = lf[i].mean()
        Sn = S / (np.linalg.norm(S, axis=0, keepdims=True) + eps)
        fl = np.sqrt((np.diff(Sn, axis=1) ** 2).sum(0))
        d["sp_flux_mean"] = fl.mean()
        d["sp_flux_std"] = fl.std()
        con = librosa.feature.spectral_contrast(S=S, sr=sr, n_fft=self.n_fft, fmin=62.5, n_bands=6).mean(1)
        for nm, v in zip(self.sc_names, con):
            d[nm] = v
        # ---------------- Spectral kurtosis (Antoni 2006) ----------------
        A2 = np.abs(librosa.stft(xc, n_fft=self.sk_nfft, hop_length=self.sk_nfft // 4, center=False, window="hann")) ** 2
        sk = (A2 ** 2).mean(1) / (A2.mean(1) ** 2 + eps) - 2.0
        m = (self.fsk >= 100) & (self.fsk <= self.f_hi)
        d["sk_mean"] = sk[m].mean()
        d["sk_max"] = sk[m].max()
        d["sk_argmax_hz"] = self.fsk[m][np.argmax(sk[m])]
        for a, b, nm in [(100, 1000, "sk_100-1000Hz"), (1000, 3000, "sk_1000-3000Hz"), (3000, self.f_hi + 1, "sk_3000Hz-up")]:
            mm = (self.fsk >= a) & (self.fsk < b)
            d[nm] = sk[mm].mean() if mm.any() else np.nan
        # ---------------- Envelope modulation ----------------
        dec = max(1, int(sr / 250)); ne = N // dec
        e = env[:ne * dec].reshape(ne, dec).mean(1); e = e - e.mean()
        fm, Pm = sps.welch(e, fs=sr / dec, nperseg=min(256, ne))
        tot = Pm[fm >= 0.5].sum() + eps
        for a, b in [(0.5, 4), (4, 16), (16, 64)]:
            d[f"mod_{a:g}-{b:g}Hz"] = Pm[(fm >= a) & (fm < b)].sum() / tot
        # ---------------- Wavelet ----------------
        cw = pywt.wavedec(xc, "db4", level=self.dwt_L, mode="symmetric")
        ew = np.array([np.sum(ci ** 2) for ci in cw]) + eps; ew /= ew.sum()
        for nm, v in zip(self.dwt_names, ew):
            d[nm] = v
        d["wd_entropy"] = -(ew * np.log(ew)).sum() / np.log(len(ew))
        wp = pywt.WaveletPacket(xc, "db4", mode="symmetric", maxlevel=4)
        ep = np.array([np.sum(nd.data ** 2) for nd in wp.get_level(4, order="freq")]) + eps; ep /= ep.sum()
        for nm, v in zip(self.wp_names, ep):
            d[nm] = v
        d["wp_entropy"] = -(ep * np.log(ep)).sum() / np.log(len(ep))
        # ---------------- Complexity ----------------
        d["nl_perm_entropy_d1"] = perm_entropy(xc, 5, 1)
        d["nl_perm_entropy_d4"] = perm_entropy(xc, 5, 4)
        d["nl_higuchi_fd"] = higuchi_fd(xc, 10)
        d["nl_katz_fd"] = katz_fd(xc)
        # ---------------- LPC ----------------
        try:
            a = librosa.lpc(xc / sd, order=10)
            for i in range(1, 11):
                d[f"lpc{i}"] = a[i]
            d["lpc_err_db"] = 10 * np.log10(sps.lfilter(a, [1.0], xc / sd).var() + eps)
        except Exception:
            for i in range(1, 11):
                d[f"lpc{i}"] = np.nan
            d["lpc_err_db"] = np.nan
        return d, fine.astype(np.float32), Pxx


FX = FeatureExtractor(SR, PARAMS)
_d, _fine, _p = FX(np.random.default_rng(0).standard_normal(CHAIN.F) * 0.01)
FEATURES = list(_d.keys())
print(f"{len(FEATURES)} đặc trưng / khung | {len(_fine)} dải 1/6 octave cho phân tích dải tần (E1) | "
      f"Welch nperseg={FX.nper} (Δf = {SR / FX.nper:.2f} Hz)")

# %%
# =========================================================
# 4.2 Danh mục đặc trưng (nhóm, mô tả, dải tần, phụ thuộc mức) - Bảng 3 của bài báo
# =========================================================
GROUPS = ["Time-domain", "Spectral shape", "Band energy", "Cepstral", "Wavelet", "LPC", "Complexity & modulation"]
GROUP_COLOR = {g: CAT[i] for i, g in enumerate(GROUPS)}
_FAM = [("td_", "Time-domain statistics", "Time-domain"), ("hm_", "Periodicity (ACF)", "Time-domain"),
        ("sp_", "Spectral shape", "Spectral shape"), ("sc", "Spectral contrast", "Spectral shape"),
        ("sk_", "Spectral kurtosis", "Spectral shape"), ("be_", "Relative 1/3-octave band energy", "Band energy"),
        ("br_", "Band-energy ratio", "Band energy"), ("bl_", "Absolute octave-band level", "Band energy"),
        ("mfcc", "MFCC", "Cepstral"), ("lfcc", "LFCC", "Cepstral"), ("wd_", "DWT relative energy", "Wavelet"),
        ("wp", "WPD relative energy", "Wavelet"), ("lpc", "Linear prediction", "LPC"),
        ("nl_", "Nonlinear complexity", "Complexity & modulation"), ("mod_", "Envelope modulation", "Complexity & modulation")]
_DESC = {
    "td_rms_db": "RMS level (dBFS)", "td_peak_db": "peak level (dBFS)", "td_crest": "crest factor peak/RMS",
    "td_impulse_factor": "impulse factor peak/mean|x|", "td_shape_factor": "shape factor RMS/mean|x|",
    "td_clearance_factor": "clearance (margin) factor", "td_skewness": "skewness", "td_kurtosis": "kurtosis (Gauss = 3)",
    "td_zcr": "zero-crossing rate", "td_hjorth_mobility": "Hjorth mobility", "td_hjorth_complexity": "Hjorth complexity",
    "td_energy_entropy": "entropy of sub-block energy (stationarity)", "td_blockrms_std_db": "std of 50-ms block level",
    "td_env_cv": "Hilbert envelope coefficient of variation", "td_env_kurtosis": "Hilbert envelope kurtosis",
    "td_acf_decay_ms": "ACF 1/e decay time", "td_acf_zero_ms": "ACF first zero crossing", "hm_acf_peak": "max ACF at 2-20 ms lag (periodicity)",
    "sp_centroid": "spectral centroid", "sp_spread": "spectral spread (bandwidth)", "sp_skewness": "spectral skewness",
    "sp_kurtosis": "spectral kurtosis (shape)", "sp_entropy": "spectral entropy", "sp_flatness": "spectral flatness (Wiener entropy)",
    "sp_rolloff50": "median frequency (50% roll-off)", "sp_rolloff85": "85% roll-off", "sp_rolloff95": "95% roll-off",
    "sp_slope_db_oct": "spectral slope (dB/octave)", "sp_decrease": "spectral decrease", "sp_crest_db": "spectral crest",
    "sp_peak_freq": "dominant frequency", "sp_n_tonal_peaks": "number of tonal peaks (>10 dB)", "sp_tonality_db": "max peak prominence",
    "sp_flux_mean": "mean spectral flux", "sp_flux_std": "std of spectral flux", "sk_mean": "mean spectral kurtosis 100 Hz-f_hi",
    "sk_max": "max spectral kurtosis", "sk_argmax_hz": "frequency of max spectral kurtosis",
    "br_below1k_vs_above_db": "energy below/above 1 kHz (dB)", "br_100-800Hz_frac": "energy fraction 100-800 Hz",
    "br_1k-3kHz_frac": "energy fraction 1-3 kHz", "mfcc0_mean": "MFCC c0 (log energy)", "mfcc_delta_std": "mean std of delta-MFCC",
    "wd_entropy": "wavelet energy entropy (DWT)", "wp_entropy": "wavelet packet energy entropy",
    "nl_perm_entropy_d1": "permutation entropy (m=5, lag 1)", "nl_perm_entropy_d4": "permutation entropy (m=5, lag 4)",
    "nl_higuchi_fd": "Higuchi fractal dimension", "nl_katz_fd": "Katz fractal dimension", "lpc_err_db": "LPC prediction error (dB)",
}
LEVEL_DEP = {n for n in FEATURES if n in ("td_rms_db", "td_peak_db", "mfcc0_mean") or n.startswith("bl_")}


def feature_band(n):
    m = re.search(r"_(\d+(?:\.\d+)?)Hz$", n)
    if n.startswith("be_") and m:
        c = float(m.group(1)); cc = 1000 * 2 ** (np.round(3 * np.log2(c / 1000)) / 3)
        return cc * 2 ** (-1 / 6), cc * 2 ** (1 / 6)
    m = re.search(r"oct(\d+)Hz", n)
    if m:
        c = float(m.group(1)); return c / np.sqrt(2), c * np.sqrt(2)
    m = re.search(r"_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?|nyq)Hz", n)
    if m and not n.startswith("mod_"):
        lo = float(m.group(1)); hi = SR / 2 if m.group(2) == "nyq" else float(m.group(2))
        return max(lo, cfg.f_lo), hi
    if n == "sk_3000Hz-up":
        return 3000.0, CHAIN.f_hi
    return np.nan, np.nan


def feature_meta(n):
    fam, grp = next(((f, g) for p, f, g in _FAM if n.startswith(p)), ("Other", "Time-domain"))
    desc = _DESC.get(n)
    if desc is None:
        m = re.match(r"(mfcc|lfcc)(\d+)_(mean|std)", n)
        if m:
            desc = f"{m.group(1).upper()} c{m.group(2)} {m.group(3)}"
        elif n.startswith("be_"):
            desc = f"relative energy, 1/3-octave band centred {n[3:]}"
        elif n.startswith("bl_"):
            desc = f"absolute level, octave band {n[6:].replace('_db', '')}"
        elif n.startswith(("wd_", "wp")):
            desc = f"relative wavelet energy {n.split('_', 1)[1]}"
        elif n.startswith("sc"):
            desc = f"spectral contrast {n.split('_', 1)[1]}"
        elif n.startswith("sk_"):
            desc = f"mean spectral kurtosis {n[3:]}"
        elif n.startswith("mod_"):
            desc = f"relative envelope modulation energy {n[4:]}"
        elif n.startswith("lpc"):
            desc = f"LPC coefficient a{n[3:]}"
        else:
            desc = n
    lo, hi = feature_band(n)
    return dict(feature=n, family=fam, group=grp, description=desc, level_dep=n in LEVEL_DEP, band_lo=lo, band_hi=hi)


FEAT_META = pd.DataFrame([feature_meta(n) for n in FEATURES]).set_index("feature")
T03 = FEAT_META.reset_index().groupby(["group", "family"]).agg(n=("feature", "size"),
                                                              examples=("feature", lambda s: ", ".join(list(s)[:4]))).reset_index()
display(Markdown(f"**Bảng 3. Danh mục {len(FEATURES)} đặc trưng thủ công.**")); display(T03)
savetab(T03, "T03_feature_catalogue_summary"); savetab(FEAT_META.reset_index(), "T03b_feature_catalogue_full")
print("Đặc trưng phụ thuộc mức tín hiệu:", sorted(LEVEL_DEP))

# %%
# =========================================================
# 4.3 Chạy chuỗi xử lý + trích đặc trưng cho MỌI tệp và MỌI điều kiện (song song, có cache)
# =========================================================
FEATURE_VERSION = "ml-v1.0"


def process_file(i, path, sr, P):
    t0 = time.time()
    chain, fx = SignalChain(sr, P), FeatureExtractor(sr, P)
    xr = load_raw(path, sr)
    qc = file_qc(xr, sr, P)
    out = dict(i=i, qc=qc, frames=None, feats={}, fine={}, psd={}, time=0.0, n_feat_frames=0)
    if qc["excluded"]:
        return out
    sig, starts, q, spike = chain.signals(xr)
    qc["spike_frac"] = spike
    q.insert(0, "frame_idx", np.arange(len(starts))); q.insert(1, "t_start", starts / sr)
    out["frames"] = q
    for cond in P["conditions"]:
        keep = np.where(chain.mask_for(cond, q))[0]; x = sig[cond]
        rows, fines, psum = [], [], 0.0
        for fi in keep:
            dd, fine, pxx = fx(x[starts[fi]: starts[fi] + chain.F])
            rows.append(dd); fines.append(fine); psum = psum + pxx
        out["feats"][cond] = (keep, rows)
        out["fine"][cond] = np.asarray(fines, np.float32)
        out["psd"][cond] = (psum, len(rows))
        out["n_feat_frames"] += len(rows)
    out["time"] = time.time() - t0
    return out


def _fingerprint():
    h = hashlib.sha1(FEATURE_VERSION.encode())
    for p in sorted(files.path):
        st = os.stat(p); h.update(f"{p}|{st.st_size}|{int(st.st_mtime)}".encode())
    keys = ["target_sr", "frame_sec", "hop_sec", "f_lo", "f_hi", "hp_hz", "lp_hz", "filt_order", "notch_hz",
            "notch_harmonics", "notch_q", "clip_level", "qc_max_clip_frac", "qc_max_zero_frac", "spike_k", "block_ms",
            "z_thr", "event_db", "kurt_thr", "frame_clip_frac", "lsd_z_thr", "lsd_min_db", "voicing_thr",
            "voicing_frac_thr", "voicing_excess", "min_keep_frac", "min_keep_frames", "stft_nfft", "stft_hop",
            "mbg_beta", "ss_alpha", "ss_floor", "ss_quantile", "wt_wavelet", "wt_level", "conditions"]
    h.update(json.dumps({k: PARAMS[k] for k in keys}, sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]


_cache = CACHE_DIR / f"features_{_fingerprint()}.pkl"
if _cache.exists():
    RESULTS = joblib.load(_cache); print(f"Nạp đặc trưng từ cache: {_cache.name}")
else:
    t0 = time.time()
    RESULTS = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(
        delayed(process_file)(i, r.path, SR, PARAMS) for i, r in files.iterrows())
    joblib.dump(RESULTS, _cache, compress=3)
    _nf = sum(r["n_feat_frames"] for r in RESULTS)
    print(f"Trích đặc trưng xong trong {time.time() - t0:.0f} s ({_nf} khung x điều kiện; "
          f"{sum(r['time'] for r in RESULTS) / max(_nf, 1) * 1000:.1f} ms CPU/khung kể cả xử lý) -> {_cache.name}")

# ---- Lắp bảng đặc trưng theo điều kiện --------------------------------------
META_COLS = ["record_id", "file_stem", "label", "cls", "split", "frame_idx", "t_start"]
FEAT, FINE, PSD_REC, QC_FRAMES, QC_FILES = {}, {}, {}, [], []
for cond in cfg.conditions:
    parts, fparts = [], []
    for res in RESULTS:
        if cond not in res["feats"] or not len(res["feats"][cond][1]):
            continue
        r = files.loc[res["i"]]; keep, rows = res["feats"][cond]
        df = pd.DataFrame(rows, columns=FEATURES)
        meta = pd.DataFrame({"record_id": r.record_id, "file_stem": r.file_stem, "label": r.label, "cls": r.cls,
                             "split": r.split, "frame_idx": keep, "t_start": keep * cfg.hop_sec})
        parts.append(pd.concat([meta, df], axis=1)); fparts.append(res["fine"][cond])
        ps, n = res["psd"][cond]
        a = PSD_REC.setdefault(cond, {}).setdefault(r.record_id, [0.0, 0])
        a[0] = a[0] + ps; a[1] += n
    FEAT[cond] = pd.concat(parts, ignore_index=True)
    FEAT[cond][FEATURES] = FEAT[cond][FEATURES].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    FINE[cond] = np.vstack(fparts)
for res in RESULTS:
    r = files.loc[res["i"]]
    QC_FILES.append({"file_stem": r.file_stem, "record_id": r.record_id, "cls": r.cls, **res["qc"]})
    if res["frames"] is not None:
        QC_FRAMES.append(res["frames"].assign(file_stem=r.file_stem, record_id=r.record_id, cls=r.cls))
QC_FILES = pd.DataFrame(QC_FILES); QC_FRAMES = pd.concat(QC_FRAMES, ignore_index=True)
REC_KEPT = sorted(FEAT["C0_RAW"].record_id.unique())
lost = sorted(set(records.record_id) - set(REC_KEPT))
if lost:
    print(f"!! {len(lost)} bản ghi mất toàn bộ tệp do QC mức tệp: {lost} -> loại khỏi phân tích.")
records = records[records.record_id.isin(REC_KEPT)].reset_index(drop=True)
for cond in cfg.conditions:
    assert set(FEAT[cond].record_id) == set(REC_KEPT), f"{cond}: thiếu bản ghi sau khi lọc khung"
print(pd.DataFrame({c: [len(FEAT[c]), FEAT[c].record_id.nunique()] for c in cfg.conditions},
                   index=["frames", "records"]).T)

# %%
# =========================================================
# 4.4 Bảng 2 + Hình 4: QC mức tệp và tỉ lệ khung bị loại theo lý do / lớp
#     (kiểm tra confound: nếu một lớp bị loại nhiều hơn hẳn thì chính bộ lọc có thể tạo ra khác biệt)
# =========================================================
excl = QC_FILES[QC_FILES.excluded]
print(f"Tệp bị loại ở QC mức tệp: {len(excl)}/{len(QC_FILES)}", excl[["file_stem", "cls", "reason"]].to_dict("records") if len(excl) else "")
REASONS = ["f_clip", "f_silence", "f_event", "f_impulsive", "f_spectral", "f_voiced"]
rej = (QC_FRAMES.groupby("cls")[REASONS + ["keep_C2", "keep_C3"]].mean()
       .assign(rejected_C2=lambda d: 1 - d.keep_C2, rejected_C3=lambda d: 1 - d.keep_C3)
       .drop(columns=["keep_C2", "keep_C3"]).T.round(4))
rec_rej = QC_FRAMES.groupby(["record_id", "cls"]).keep_C3.mean().rsub(1).reset_index(name="rej_rate")
p_rej = mwu_p(rec_rej.loc[rec_rej.cls == "leak", "rej_rate"], rec_rej.loc[rec_rej.cls == "noleak", "rej_rate"])
d_rej = cliffs_delta(rec_rej.loc[rec_rej.cls == "leak", "rej_rate"], rec_rej.loc[rec_rej.cls == "noleak", "rej_rate"])
display(Markdown("**Bảng 2. Tỉ lệ khung bị gắn cờ theo lý do và theo lớp (mức khung).**")); display(rej)
savetab(rej.reset_index().rename(columns={"index": "reason"}), "T02_rejection_by_reason")
savetab(QC_FILES, "T02b_file_qc")
print(f"Tỉ lệ loại (C3) mức bản ghi: leak trung vị {rec_rej[rec_rej.cls == 'leak'].rej_rate.median():.3f} | "
      f"noleak {rec_rej[rec_rej.cls == 'noleak'].rej_rate.median():.3f} | Mann-Whitney p = {p_rej:.3f}, Cliff δ = {d_rej:+.2f}")
REJ_CONFOUND = bool(np.isfinite(p_rej) and p_rej < 0.05)
print("=> " + ("!! Tỉ lệ loại KHÁC NHAU giữa hai lớp: phải thảo luận như một confound tiềm ẩn."
               if REJ_CONFOUND else "Tỉ lệ loại không khác biệt có ý nghĩa giữa hai lớp -> bộ lọc không tạo khác biệt giả."))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1.6, 1]})
xx = np.arange(len(REASONS) + 2); labels = [r[2:] for r in REASONS] + ["total C2", "total C3"]
for j, c in enumerate(["noleak", "leak"]):
    v = rej.loc[REASONS + ["rejected_C2", "rejected_C3"], c].values * 100
    ax[0].bar(xx + (j - 0.5) * 0.38, v, width=0.36, color=PAL[c], label=c)
ax[0].set_xticks(xx); ax[0].set_xticklabels(labels, rotation=25); ax[0].set_ylabel("% of frames flagged")
ax[0].set_title("(a) Frame rejection by reason"); ax[0].legend()
for j, c in enumerate(["noleak", "leak"]):
    v = rec_rej.loc[rec_rej.cls == c, "rej_rate"].values * 100
    ax[1].scatter(np.full(len(v), j) + np.random.default_rng(j).uniform(-0.12, 0.12, len(v)), v, s=14,
                  color=PAL[c], edgecolor="white", linewidth=0.5)
    ax[1].hlines(np.median(v), j - 0.25, j + 0.25, color=INK, lw=1.5)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["noleak", "leak"]); ax[1].set_ylabel("% frames removed (C3)")
ax[1].set_title(f"(b) Per-recording removal rate (MWU p = {p_rej:.2f})")
fig.tight_layout()
savefig(fig, "Fig04_rejection_stats", "Frame rejection statistics by reason and class.")
