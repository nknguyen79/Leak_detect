# %% [markdown]
# ## 5. E1 — Dải tần nào chứa tín hiệu rò rỉ? / Which frequency band carries the leak signal? (RQ1)
#
# Ba lớp bằng chứng độc lập, đều ở **mức bản ghi** (đơn vị độc lập), trên điều kiện `cfg.e1_condition`
# (đã loại xung và âm thanh không liên quan, **chưa** khử nhiễu để không làm méo phổ):
#
# 1. **E1a – Phổ trung bình dài hạn (LTAS)** theo lớp và **phổ hiệu** leak − noleak (dB) kèm CI bootstrap.
# 2. **E1b – Phổ khả năng phân biệt**: AUC đơn biến của năng lượng tương đối trong từng dải 1/6 octave,
#    kiểm định Mann–Whitney + hiệu chỉnh Benjamini–Hochberg (FDR) và Holm; bảng 1/3 octave (Bảng 4).
# 3. **E1d – Quét dải bằng mô hình**: huấn luyện Random Forest chỉ với thông tin trong một cửa sổ tần số
#    (octave trượt 1/3 octave; thông thấp tích luỹ `[f_lo, fc]`; thông cao tích luỹ `[fc, f_hi]`), đánh giá
#    bằng CV lặp nhóm theo bản ghi. **Dải thiết yếu** = `[fc_HP*, fc_LP*]` trong đó `fc_LP*` là tần số cắt
#    thông thấp nhỏ nhất và `fc_HP*` là tần số cắt thông cao lớn nhất mà AUC còn ≥ AUC(toàn dải) − `e1_tol`.
#
# > E1 là phân tích **mô tả**: không một tham số nào của E2–E4 được chọn từ kết quả E1 (dải lọc C1 và
# > danh mục đặc trưng được cố định trước). Vì vậy có thể dùng toàn bộ bản ghi (`cfg.e1_scope = "all"`)
# > để có độ mạnh thống kê lớn nhất mà không làm rò rỉ thông tin sang tập kiểm tra.

# %%
# =========================================================
# 5.1 Chuẩn bị: kế hoạch fold mức bản ghi dùng chung cho mọi thí nghiệm (thiết kế GHÉP CẶP)
# =========================================================
def make_record_folds(rec_ids, rec_y, n_splits, seed):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [np.asarray(rec_ids)[te] for _, te in skf.split(np.zeros(len(rec_ids)), rec_y)]


REC_LABEL = records.set_index("record_id").label
SCOPE_RECS = {"dev": records.loc[records.split == "dev", "record_id"].values,
              "all": records.record_id.values, "test": records.loc[records.split == "test", "record_id"].values}
FOLDS = {}
for scope in ("dev", "all"):
    ids = np.sort(SCOPE_RECS[scope]); yy = REC_LABEL.loc[ids].values
    k = int(min(cfg.n_splits, np.bincount(yy).min()))
    for s in sorted(set(cfg.seeds) | set(cfg.e1_seeds)):
        FOLDS[(scope, s)] = make_record_folds(ids, yy, k, s)
print({k: [len(f) for f in v] for k, v in list(FOLDS.items())[:2]}, "... (số bản ghi test mỗi fold)")


def sample_weights(y, g):
    """Mỗi bản ghi tổng trọng số như nhau; mỗi lớp tổng trọng số như nhau."""
    y = np.asarray(y); g = np.asarray(g); w = np.ones(len(y), float)
    if cfg.record_balanced:
        _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True); w = 1.0 / cnt[inv]
    for c in (0, 1):
        m = y == c
        if m.any():
            w[m] *= 0.5 / w[m].sum()
    return w * len(w) / w.sum()


def rec_scores(scores, g, rec_ids):
    s = pd.Series(np.asarray(scores, float)).groupby(np.asarray(g)).mean()
    return s.reindex(rec_ids).values


E1_COND = cfg.e1_condition if cfg.e1_condition in FEAT else cfg.conditions[0]
E1_RECS = np.sort(SCOPE_RECS[cfg.e1_scope])
_e1 = FEAT[E1_COND]; _m = _e1.record_id.isin(E1_RECS).values
E1_DF = _e1.loc[_m].reset_index(drop=True); E1_FINE = FINE[E1_COND][_m]
E1_Y = REC_LABEL.loc[E1_RECS].values
FB_FC = np.sqrt(FX.fb_lo * FX.fb_hi)
print(f"E1 trên {len(E1_RECS)} bản ghi ({E1_Y.sum()} leak / {(1 - E1_Y).sum()} noleak), điều kiện {E1_COND}, "
      f"{len(E1_DF)} khung, {E1_FINE.shape[1]} dải 1/6 octave")

# %%
# =========================================================
# 5.2 E1a + E1b: LTAS, phổ hiệu và phổ khả năng phân biệt (mức bản ghi)
# =========================================================
fw = FX.fw
_man = FX.m_an
REC_PSD = np.vstack([PSD_REC[E1_COND][r][0] / PSD_REC[E1_COND][r][1] for r in E1_RECS])
REL_PSD_DB = 10 * np.log10(REC_PSD[:, _man] / REC_PSD[:, _man].sum(1, keepdims=True) + 1e-20)
f_an = fw[_man]

# năng lượng 1/6 octave mức bản ghi (trung bình tuyến tính các khung) -> tương đối -> dB
_fb = pd.DataFrame(E1_FINE).groupby(E1_DF.record_id.values).mean().reindex(E1_RECS).values
FB_REL_DB = 10 * np.log10(_fb / _fb.sum(1, keepdims=True) + 1e-20)
FB_ABS_DB = 10 * np.log10(_fb + 1e-20)

rng = np.random.default_rng(0)
pos, neg = np.where(E1_Y == 1)[0], np.where(E1_Y == 0)[0]
nb_ = max(200, cfg.n_boot // 2)
delta = np.median(FB_REL_DB[pos], 0) - np.median(FB_REL_DB[neg], 0)
boots = np.array([np.median(FB_REL_DB[rng.choice(pos, len(pos))], 0) - np.median(FB_REL_DB[rng.choice(neg, len(neg))], 0)
                  for _ in range(nb_)])
d_lo, d_hi = np.percentile(boots, [2.5, 97.5], axis=0)

rows = []
for j in range(FB_REL_DB.shape[1]):
    a = auc_safe(E1_Y, FB_REL_DB[:, j]); lo, hi = boot_auc_ci(E1_Y, FB_REL_DB[:, j], n_boot=nb_, seed=j)
    rows.append(dict(f_lo=FX.fb_lo[j], f_hi=FX.fb_hi[j], fc=FB_FC[j], auc_rel=a, auc_lo=lo, auc_hi=hi,
                     auc_abs=auc_safe(E1_Y, FB_ABS_DB[:, j]), delta_db=delta[j], delta_lo=d_lo[j], delta_hi=d_hi[j],
                     p=mwu_p(FB_REL_DB[pos, j], FB_REL_DB[neg, j])))
E1B = pd.DataFrame(rows)
E1B["q_bh"] = bh_fdr(E1B.p.values); E1B["p_holm"] = holm(E1B.p.values)
E1B["sep"] = (E1B.auc_rel - 0.5).abs() * 2        # sức phân biệt 0..1, không phụ thuộc chiều
savetab(E1B.round(5), "T04b_fine_band_auc")

fig, ax = plt.subplots(3, 1, figsize=(9, 8.4), sharex=True)
for c, yv in (("noleak", 0), ("leak", 1)):
    A = REL_PSD_DB[E1_Y == yv]; med = np.median(A, 0); q1, q3 = np.percentile(A, [25, 75], axis=0)
    k = 9; ker = np.ones(k) / k
    sm = lambda v: np.convolve(v, ker, mode="same")
    ax[0].plot(f_an, sm(med), color=PAL[c], lw=1.4, label=f"{c} (n={len(A)})")
    ax[0].fill_between(f_an, sm(q1), sm(q3), color=PAL[c], alpha=0.18, lw=0)
_inb = f_an <= min(cfg.lp_hz, CHAIN.f_hi) * 0.97
_lvl = np.median(REL_PSD_DB[:, _inb], 0)
ax[0].set_ylim(np.percentile(_lvl, 0.5) - 8, np.percentile(_lvl, 99.5) + 5)
ax[0].set_ylabel("relative PSD (dB re total)"); ax[0].legend(loc="lower left")
ax[0].set_title(f"(a) Long-term average spectrum per class, median and IQR over recordings [{E1_COND}]")
ax[1].axhline(0, color=INK2, lw=0.8)
ax[1].fill_between(FB_FC, E1B.delta_lo, E1B.delta_hi, color=PAL["leak"], alpha=0.2, lw=0, step="mid")
ax[1].step(FB_FC, E1B.delta_db, where="mid", color=PAL["leak"], lw=1.5)
ax[1].set_ylabel("Δ level leak − noleak (dB)")
ax[1].set_title("(b) Difference spectrum (1/6 octave) with 95% bootstrap CI")
ax[2].axhline(0.5, color=INK2, lw=0.8)
ax[2].fill_between(FB_FC, E1B.auc_lo, E1B.auc_hi, color=CAT[0], alpha=0.18, lw=0, step="mid")
ax[2].step(FB_FC, E1B.auc_rel, where="mid", color=CAT[0], lw=1.5, label="relative band energy")
ax[2].step(FB_FC, E1B.auc_abs, where="mid", color=CAT[2], lw=1.1, label="absolute band level")
sig_ = E1B.q_bh < 0.05
ax[2].scatter(FB_FC[sig_], E1B.auc_rel[sig_], s=22, color=CAT[0], edgecolor="white", zorder=3, label="q(BH) < 0.05")
ax[2].set_ylabel("record-level AUC"); ax[2].set_xlabel("frequency (Hz)"); ax[2].legend(loc="best", ncol=3)
ax[2].set_title("(c) Univariate discriminability per 1/6-octave band (AUC > 0.5: higher in leak)")
freq_axis(ax[2], cfg.f_lo, CHAIN.f_hi)
fig.tight_layout()
savefig(fig, "Fig05_ltas_discriminability", "LTAS per class, difference spectrum and per-band AUC.")

# ---- Bảng 4: dải 1/3 octave (đặc trưng be_*) --------------------------------
_be = [n for n in FEATURES if n.startswith("be_")]
_rm = E1_DF.groupby("record_id")[_be].mean().reindex(E1_RECS)
T04 = []
for n in _be:
    v = _rm[n].values; lo_, hi_ = feature_band(n)
    a = auc_safe(E1_Y, v); ci = boot_auc_ci(E1_Y, v, n_boot=nb_, seed=1)
    T04.append(dict(band=n[3:], f_lo=round(lo_, 1), f_hi=round(hi_, 1), leak_median_db=np.median(v[pos]),
                    noleak_median_db=np.median(v[neg]), auc=a, auc_lo=ci[0], auc_hi=ci[1],
                    cliffs_delta=cliffs_delta(v[pos], v[neg]), p=mwu_p(v[pos], v[neg])))
T04 = pd.DataFrame(T04); T04["p_holm"] = holm(T04.p.values); T04["q_bh"] = bh_fdr(T04.p.values)
display(Markdown("**Bảng 4. Năng lượng tương đối theo dải 1/3 octave — AUC mức bản ghi (AUC > 0,5: leak cao hơn).**"))
display(T04.round(4)); savetab(T04.round(5), "T04_third_octave_auc")
_top = T04.assign(sep=(T04.auc - 0.5).abs()).sort_values("sep", ascending=False).head(5)
print("5 dải 1/3 octave phân biệt mạnh nhất:", "; ".join(
    f"{r.band} (AUC {r.auc:.2f}, p_Holm {r.p_holm:.3f})" for r in _top.itertuples()))

# %%
# =========================================================
# 5.3 E1d: Quét dải tần bằng mô hình (Random Forest, CV lặp nhóm theo bản ghi)
# =========================================================
def _win_idx(a, b):
    return np.where((FX.fb_lo >= a * 0.999) & (FX.fb_hi <= b * 1.001))[0]


WINDOWS = []
_c = cfg.f_lo
while 2 * _c <= CHAIN.f_hi * 1.001:
    WINDOWS.append(("octave", _c, 2 * _c)); _c *= 2 ** (1 / 3)
for fc in [125, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000, CHAIN.f_hi]:
    if cfg.f_lo < fc <= CHAIN.f_hi:
        WINDOWS.append(("lowpass", cfg.f_lo, fc))
for fc in [cfg.f_lo, 125, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]:
    if fc < CHAIN.f_hi / 1.5:
        WINDOWS.append(("highpass", fc, CHAIN.f_hi))
WINDOWS = [(k, a, b, _win_idx(a, b)) for k, a, b in WINDOWS]
WINDOWS = [w for w in WINDOWS if len(w[3]) >= 2]
E1_LB = 10 * np.log10(E1_FINE.astype(np.float64) + 1e-20)


def _win_features(LB, idx, shape_only=False):
    lin = 10 ** (LB[:, idx] / 10); tot = 10 * np.log10(lin.sum(1, keepdims=True) + 1e-20)
    rel = LB[:, idx] - tot
    return rel if shape_only else np.hstack([rel, tot])


def e1_fold_job(LBtr, ytr, gtr, LBte, windows, seed):
    w = sample_weights(ytr, gtr); out = {}
    for wi, (kind, a, b, idx) in enumerate(windows):
        for shape_only in ((False, True) if kind == "octave" else (False,)):
            m = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt", n_jobs=1, random_state=seed)
            m.fit(_win_features(LBtr, idx, shape_only), ytr, sample_weight=w)
            out[(wi, shape_only)] = m.predict_proba(_win_features(LBte, idx, shape_only))[:, 1].astype(np.float32)
    return out


_g = E1_DF.record_id.values; _yf = E1_DF.label.values
jobs = []
for s in cfg.e1_seeds:
    for fi, te_r in enumerate(FOLDS[(cfg.e1_scope, s)]):
        te = np.isin(_g, te_r); jobs.append((s, te))
t0 = time.time()
_res = Parallel(n_jobs=N_JOBS, backend="loky")(
    delayed(e1_fold_job)(E1_LB[~te], _yf[~te], _g[~te], E1_LB[te], WINDOWS, 1000 * s + i)
    for i, (s, te) in enumerate(jobs))
E1_SC = defaultdict(lambda: np.full((len(cfg.e1_seeds), len(_yf)), np.nan))
for (s, te), out in zip(jobs, _res):
    r = list(cfg.e1_seeds).index(s)
    for key, sc in out.items():
        E1_SC[key][r, te] = sc
rows = []
for (wi, shape_only), S in E1_SC.items():
    kind, a, b, idx = WINDOWS[wi]
    rs = rec_scores(np.nanmean(S, 0), _g, E1_RECS)
    per_rep = [auc_safe(E1_Y, rec_scores(S[r], _g, E1_RECS)) for r in range(S.shape[0])]
    auc = auc_safe(E1_Y, rs); lo, hi = boot_auc_ci(E1_Y, rs, n_boot=nb_, seed=wi)
    rows.append(dict(kind=kind, variant="shape" if shape_only else "shape+level", f_lo=a, f_hi=b,
                     fc=np.sqrt(a * b), n_bands=len(idx), auc=auc, auc_lo=lo, auc_hi=hi, auc_rep_sd=np.nanstd(per_rep)))
E1D = pd.DataFrame(rows).sort_values(["kind", "variant", "f_lo", "f_hi"]).reset_index(drop=True)
savetab(E1D.round(4), "T05_band_sweep")
print(f"Quét {len(WINDOWS)} cửa sổ x {len(jobs)} fold trong {time.time() - t0:.0f} s")

# ---- Dải thiết yếu ------------------------------------------------------------
lp = E1D[(E1D.kind == "lowpass")].sort_values("f_hi"); hp = E1D[(E1D.kind == "highpass")].sort_values("f_lo")
AUC_FULL = float(lp.auc.iloc[-1])
fc_lp = float(lp.loc[lp.auc >= AUC_FULL - cfg.e1_tol, "f_hi"].min())
fc_hp = float(hp.loc[hp.auc >= AUC_FULL - cfg.e1_tol, "f_lo"].max())
oct_ = E1D[(E1D.kind == "octave") & (E1D.variant == "shape+level")].sort_values("auc", ascending=False)
ESS = None
if fc_hp < fc_lp and len(_win_idx(fc_hp, fc_lp)) >= 2:
    idx = _win_idx(fc_hp, fc_lp)
    _r2 = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(e1_fold_job)(E1_LB[~te], _yf[~te], _g[~te], E1_LB[te], [("ess", fc_hp, fc_lp, idx)], 1000 * s + i)
        for i, (s, te) in enumerate(jobs))
    S = np.full((len(cfg.e1_seeds), len(_yf)), np.nan)
    for (s, te), out in zip(jobs, _r2):
        S[list(cfg.e1_seeds).index(s), te] = out[(0, False)]
    rs = rec_scores(np.nanmean(S, 0), _g, E1_RECS)
    ESS = dict(f_lo=fc_hp, f_hi=fc_lp, auc=auc_safe(E1_Y, rs), ci=boot_auc_ci(E1_Y, rs, n_boot=nb_, seed=7))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
for v, col in (("shape+level", CAT[0]), ("shape", CAT[2])):
    d = E1D[(E1D.kind == "octave") & (E1D.variant == v)].sort_values("fc")
    ax[0].fill_between(d.fc, d.auc_lo, d.auc_hi, color=col, alpha=0.15, lw=0)
    ax[0].plot(d.fc, d.auc, "-o", color=col, ms=3.5, label=f"octave window, {v}")
ax[0].axhline(0.5, color=INK2, lw=0.8); ax[0].axhline(AUC_FULL, color=CAT[1], lw=1.0)
ax[0].text(E1D.fc.min(), AUC_FULL + 0.01, f"full band {AUC_FULL:.3f}", color=INK2, fontsize=8)
freq_axis(ax[0], label="window centre frequency (Hz)"); ax[0].set_ylabel("record-level AUC")
ax[0].set_title("(a) Model AUC using one octave window only"); ax[0].legend(loc="lower right")
for d, col, lab, xcol in ((lp, CAT[0], "low-pass [f_lo, fc]", "f_hi"), (hp, CAT[1], "high-pass [fc, f_hi]", "f_lo")):
    ax[1].fill_between(d[xcol], d.auc_lo, d.auc_hi, color=col, alpha=0.15, lw=0)
    ax[1].plot(d[xcol], d.auc, "-o", color=col, ms=3.5, label=lab)
ax[1].axhline(AUC_FULL - cfg.e1_tol, color=INK2, lw=0.8)
ax[1].axvline(fc_lp, color=CAT[0], lw=0.8); ax[1].axvline(fc_hp, color=CAT[1], lw=0.8)
freq_axis(ax[1], label="cut-off frequency fc (Hz)"); ax[1].set_ylabel("record-level AUC")
ax[1].set_title(f"(b) Cumulative band sweep; essential band ≈ {fc_hp:.0f}-{fc_lp:.0f} Hz"); ax[1].legend(loc="lower center")
fig.tight_layout()
savefig(fig, "Fig06_band_sweep", "Band-limited model performance (octave windows, low-pass and high-pass sweeps).")

# %%
# =========================================================
# 5.4 Kết luận RQ1 (sinh tự động từ số đo)
# =========================================================
_sig = E1B[E1B.q_bh < 0.05]
_hi_leak = E1B[(E1B.auc_lo > 0.5)]; _hi_nol = E1B[(E1B.auc_hi < 0.5)]


def _ranges(df):
    if not len(df):
        return "không có"
    df = df.sort_values("f_lo"); out = []; a, b = df.f_lo.iloc[0], df.f_hi.iloc[0]
    for lo_, hi_ in zip(df.f_lo.iloc[1:], df.f_hi.iloc[1:]):
        if lo_ <= b * 1.001:
            b = hi_
        else:
            out.append((a, b)); a, b = lo_, hi_
    out.append((a, b))
    return ", ".join(f"{x:.0f}-{y:.0f} Hz" for x, y in out)


E1_SUMMARY = dict(cond=E1_COND, n_rec=int(len(E1_RECS)), auc_full=AUC_FULL, fc_lp=fc_lp, fc_hp=fc_hp,
                  best_octave=f"{oct_.f_lo.iloc[0]:.0f}-{oct_.f_hi.iloc[0]:.0f} Hz", best_octave_auc=float(oct_.auc.iloc[0]),
                  leak_higher=_ranges(_hi_leak), noleak_higher=_ranges(_hi_nol), n_sig_fdr=int(len(_sig)), ess=ESS)
txt = [f"**RQ1 — Dải tần mang thông tin rò rỉ** ({E1_COND}, {len(E1_RECS)} bản ghi):",
       f"- Năng lượng tương đối **cao hơn ở bản ghi leak** (CI 95% của AUC nằm trên 0,5): {E1_SUMMARY['leak_higher']}.",
       f"- Năng lượng tương đối **cao hơn ở bản ghi noleak**: {E1_SUMMARY['noleak_higher']}.",
       f"- Số dải 1/6 octave còn ý nghĩa sau hiệu chỉnh FDR: {len(_sig)}/{len(E1B)}.",
       f"- Cửa sổ một octave tốt nhất: {E1_SUMMARY['best_octave']} (AUC {E1_SUMMARY['best_octave_auc']:.3f}); toàn dải: AUC {AUC_FULL:.3f}.",
       f"- Thông thấp: chỉ cần [{cfg.f_lo:.0f}, {fc_lp:.0f}] Hz để đạt AUC ≥ toàn dải − {cfg.e1_tol}; "
       f"thông cao: bỏ được phần dưới {fc_hp:.0f} Hz mà vẫn đạt ngưỡng đó."]
if ESS:
    txt.append(f"- **Dải thiết yếu {ESS['f_lo']:.0f}–{ESS['f_hi']:.0f} Hz**: AUC {fmt_ci(ESS['auc'], *ESS['ci'])} "
               f"(so với toàn dải {AUC_FULL:.3f}).")
else:
    txt.append(f"- fc_HP* ({fc_hp:.0f} Hz) ≥ fc_LP* ({fc_lp:.0f} Hz): thông tin phân biệt **lặp lại** ở cả dải thấp và dải cao "
               "(mỗi phía tự nó đã đủ) → không tồn tại một dải hẹp duy nhất; báo cáo cả hai phía.")
display(Markdown("\n".join(txt)))
