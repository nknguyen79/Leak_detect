# %% [markdown]
# ## 10. E5 — Diễn giải, độ bền và tính ứng dụng / Interpretation, robustness and practicality (RQ4)
#
# 1. **Độ quan trọng đặc trưng** của mô hình tốt nhất: permutation importance trên TEST; SHAP (TreeExplainer;
#    Lundberg et al. 2020) cho mô hình cây tốt nhất; **ánh xạ độ quan trọng lên trục tần số** (đối chiếu RQ1).
# 2. **Kiểm tra confound mức tín hiệu**: mô hình chỉ dùng mức RMS; mô hình bỏ mọi đặc trưng phụ thuộc mức.
# 3. **Minh hoạ rò rỉ dữ liệu**: chia fold ngẫu nhiên theo KHUNG (sai) so với theo BẢN GHI (đúng).
# 4. **Thời gian nghe tối thiểu**: AUC mức bản ghi theo số giây nghe liên tục (giá trị thực tiễn cho người vận hành).
# 5. **Chi phí tính toán** (trích đặc trưng + suy luận) và **phân tích lỗi** theo bản ghi.

# %%
# =========================================================
# 10.1 Độ quan trọng đặc trưng: permutation (TEST) + SHAP + ánh xạ tần số
# =========================================================
_eval = DS(BEST_COND, "test" if HAS_TEST else "dev", FEAT_MAIN)
_fin = FINAL[("K_MAIN", BEST_MODEL)]
_P = _fin["prep"]
_B = np.clip((np.where(np.isfinite(_eval.X), _eval.X, _P["med"]) - _P["mu"]) / _P["sd"], -CLIP_Z, CLIP_Z).astype(np.float32)
base = auc_safe(_eval.y, score_model(BEST_MODEL, _fin["model"], _B))
rng = np.random.default_rng(0); n_rep = 5 if cfg.smoke_test else 20; imp = []
for j, n in enumerate(FEAT_MAIN):
    drops = []
    for _ in range(n_rep):
        Bp = _B.copy(); Bp[:, j] = rng.permutation(Bp[:, j])
        drops.append(base - auc_safe(_eval.y, score_model(BEST_MODEL, _fin["model"], Bp)))
    imp.append(dict(feature=n, perm_auc_drop=np.mean(drops), perm_sd=np.std(drops)))
PIMP = pd.DataFrame(imp).merge(FEAT_META.reset_index()[["feature", "group", "band_lo", "band_hi", "level_dep"]], on="feature") \
    .sort_values("perm_auc_drop", ascending=False).reset_index(drop=True)
savetab(PIMP.round(5), "T16_permutation_importance")
print(f"Permutation importance ({BEST_MODEL}, {'TEST' if HAS_TEST else 'DEV (không có TEST)'}; AUC khung gốc {base:.3f}):")
display(PIMP.head(15).round(4))

SHAP_OK = False
tree_models = [m for m in ("XGB", "RF") if m in cfg.models]
if HAS_SHAP and tree_models:
    _tm = E4_DEV[(E4_DEV.featset == "K_MAIN") & (E4_DEV.model.isin(tree_models))].sort_values("rec_auc_rep_mean").model.iloc[-1]
    try:
        _tf = FINAL[("K_MAIN", _tm)]; _Xs = _B[: min(len(_B), 1500)]
        sv = shap.TreeExplainer(_tf["model"]).shap_values(_Xs)
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if np.ndim(sv) == 3 else sv)
        fig = plt.figure(figsize=(7, 6))
        shap.summary_plot(sv, _Xs, feature_names=FEAT_MAIN, max_display=20, show=False, plot_size=None)
        plt.title(f"SHAP summary - {_tm} ({'TEST' if HAS_TEST else 'DEV'} frames)")
        savefig(plt.gcf(), "Fig13_shap_summary", f"SHAP values of the {_tm} model.")
        SHAP_IMP = pd.Series(np.abs(sv).mean(0), index=FEAT_MAIN).sort_values(ascending=False)
        SHAP_OK = True
    except Exception as e:
        print("!! SHAP thất bại:", type(e).__name__, str(e)[:200])

fig, ax = plt.subplots(figsize=(9, 3.4))
_pm = PIMP.dropna(subset=["band_lo"])
for r in _pm.itertuples():
    ax.plot([r.band_lo, r.band_hi], [r.perm_auc_drop] * 2, color=GROUP_COLOR[r.group], lw=3, solid_capstyle="butt")
for g in sorted(set(_pm.group)):
    ax.plot([], [], color=GROUP_COLOR[g], lw=3, label=g)
ax.axhline(0, color=INK2, lw=0.8); freq_axis(ax, cfg.f_lo, CHAIN.f_hi); ax.set_ylabel("AUC drop when permuted")
ax.set_title(f"Frequency map of band-specific feature importance ({BEST_MODEL}, K_MAIN)"); ax.legend(fontsize=7)
fig.tight_layout()
savefig(fig, "Fig14_importance_frequency_map", "Permutation importance of band-specific features on the frequency axis.")

# %%
# =========================================================
# 10.2 Confound mức tín hiệu, rò rỉ dữ liệu theo khung, thời gian nghe tối thiểu
# =========================================================
ROB = []
_lvl = run_cv(DS(BEST_COND, "dev", ["td_rms_db"]), [(None, 0, "RF"), (None, 0, "SVM")], tag="level-only")
for m in ("RF", "SVM"):
    s = summarize(_lvl, (None, 0, m)); ROB.append(dict(test="level only (RMS)", model=m, k=1, rec_auc=s["rec_auc_rep_mean"],
                                                       lo=s["rec_auc_lo"], hi=s["rec_auc_hi"]))
# chọn lại TRONG fold chỉ từ các đặc trưng KHÔNG phụ thuộc mức, cùng k = K_MAIN, cùng fold với E3 (ghép cặp)
_ginv = run_cv(DS(BEST_COND, "dev", [f for f in FEATURES if f not in LEVEL_DEP]),
               [("consensus", K_MAIN, m) for m in cfg.models], rankers=("consensus",), tag="gain-invariant")
for m in cfg.models:
    a, b = summarize(E3_RUN, ("consensus", K_MAIN, m)), summarize(_ginv, ("consensus", K_MAIN, m))
    c = compare(b, a, E3_RUN.ds.rec_y)
    ROB.append(dict(test="in-fold top-K_MAIN, all features", model=m, k=K_MAIN, rec_auc=a["rec_auc_rep_mean"],
                    lo=a["rec_auc_lo"], hi=a["rec_auc_hi"]))
    ROB.append(dict(test="in-fold top-K_MAIN, level-dependent features excluded", model=m, k=K_MAIN,
                    rec_auc=b["rec_auc_rep_mean"], lo=b["rec_auc_lo"], hi=b["rec_auc_hi"], d_vs_full=c["d_auc"],
                    p_delong=c["p_delong"]))
_full = run_cv(DS(BEST_COND, "dev", FEAT_MAIN), [(None, 0, BEST_MODEL)], tag="K_MAIN default (grouped)")

# rò rỉ dữ liệu: KFold ngẫu nhiên theo khung trên DEV (SAI) vs theo bản ghi (ĐÚNG, _full)
_d = _full.ds


def leak_job(tr, te, m, seed):
    prep = Prep().fit(_d.X[tr]); mdl = fit_model(m, prep.transform(_d.X[tr]), _d.y[tr], _d.g[tr], seed=seed)
    return te, score_model(m, mdl, prep.transform(_d.X[te]))


_kf = KFold(cfg.n_splits, shuffle=True, random_state=cfg.seeds[0]).split(_d.X)
_o = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(leak_job)(tr, te, BEST_MODEL, 3) for tr, te in _kf)
_sr = np.full(len(_d.y), np.nan)
for te, sc in _o:
    _sr[te] = sc
_grp = summarize(_full, (None, 0, BEST_MODEL))
LEAKDEMO = pd.DataFrame([
    dict(protocol="A. grouped by recording (correct)", frame_auc=_grp["frame_auc"], rec_auc=_grp["rec_auc"]),
    dict(protocol="B. random frame split (leaky)", frame_auc=auc_safe(_d.y, _sr), rec_auc=auc_safe(_d.rec_y, rec_scores(_sr, _d.g, _d.recs)))])
LEAKDEMO.loc[2] = ["B − A (inflation)", LEAKDEMO.frame_auc[1] - LEAKDEMO.frame_auc[0], LEAKDEMO.rec_auc[1] - LEAKDEMO.rec_auc[0]]
ROB = pd.DataFrame(ROB)
display(Markdown("**Bảng 17. Kiểm tra độ bền (DEV CV, tham số mặc định).**")); display(ROB.round(4))
display(Markdown(f"**Bảng 18. Mức thổi phồng do chia fold theo khung ({BEST_MODEL}, K_MAIN).**")); display(LEAKDEMO.round(4))
savetab(ROB.round(4), "T17_robustness"); savetab(LEAKDEMO.round(4), "T18_leakage_demo")

# thời gian nghe tối thiểu: n khung LIÊN TỤC ngẫu nhiên trong mỗi bản ghi
if HAS_TEST:
    _lt_df, _lt_s, _lt_src = DST.df, TEST_SC[("K_MAIN", BEST_MODEL)][0], "TEST"
else:
    _lt_df, _lt_s, _lt_src = NESTED["K_MAIN"].ds.df, np.nanmean(NESTED["K_MAIN"].scores[(None, K_MAIN, BEST_MODEL)], 0), "DEV nested OOF"
_lt = _lt_df.assign(s=_lt_s).sort_values(["record_id", "file_stem", "frame_idx"])
_groups = {r: g.s.values for r, g in _lt.groupby("record_id")}
_ry = REC_LABEL.loc[list(_groups)].values
rng = np.random.default_rng(1); LT = []
for n in [1, 2, 3, 5, 8, 10, 15, 20, 30, 45]:
    aucs = []
    for _ in range(50 if cfg.smoke_test else 300):
        sc = []
        for v in _groups.values():
            if len(v) <= n:
                sc.append(v.mean())
            else:
                st_ = rng.integers(0, len(v) - n + 1); sc.append(v[st_: st_ + n].mean())
        aucs.append(auc_safe(_ry, sc))
    LT.append(dict(n_frames=n, seconds=n * cfg.hop_sec + (cfg.frame_sec - cfg.hop_sec), auc=np.mean(aucs),
                   lo=np.percentile(aucs, 2.5), hi=np.percentile(aucs, 97.5)))
LT = pd.DataFrame(LT)
AUC_ALL_FRAMES = auc_safe(_ry, [v.mean() for v in _groups.values()])     # nghe toàn bộ bản ghi
LT["auc_all_frames"] = AUC_ALL_FRAMES
LT_MIN_S = float(LT.seconds[LT.auc >= AUC_ALL_FRAMES - 0.02].min()) if (LT.auc >= AUC_ALL_FRAMES - 0.02).any() else np.nan
savetab(LT.round(4), "T19_listening_time")
print(f"AUC khi nghe toàn bộ bản ghi = {AUC_ALL_FRAMES:.3f}; nghe liên tục {LT_MIN_S:.0f} s đạt AUC >= toàn bộ - 0,02")
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.fill_between(LT.seconds, LT.lo, LT.hi, color=CAT[0], alpha=0.18, lw=0)
ax.plot(LT.seconds, LT.auc, "-o", color=CAT[0], ms=4, label="n contiguous frames")
ax.axhline(AUC_ALL_FRAMES, color=INK2, lw=0.8, label=f"whole recording ({AUC_ALL_FRAMES:.3f})"); ax.legend(loc="lower right")
ax.set_xscale("log"); ax.set_xlabel("continuous listening time per recording (s)"); ax.set_ylabel("record-level AUC")
ax.set_title(f"How long must the operator listen? ({BEST_MODEL}, {_lt_src})")
fig.tight_layout()
savefig(fig, "Fig15_listening_time", "Record-level AUC as a function of continuous listening time.")

# %%
# =========================================================
# 10.3 Chi phí tính toán và phân tích lỗi theo bản ghi
# =========================================================
_r0 = files.iloc[0]; _x = load_raw(_r0.path, SR)
t0 = time.time(); _sig, _st, _q, _ = CHAIN.signals(_x); t_proc = (time.time() - t0) / (len(_x) / SR)
_n = min(20, len(_st)); t0 = time.time()
for s in _st[:_n]:
    FX(_sig[BEST_COND][s: s + CHAIN.F])
t_feat = (time.time() - t0) / max(_n, 1) * 1000
COST = [dict(item=f"signal chain C0-C6 (s CPU per s audio)", value=t_proc),
        dict(item=f"full feature extraction ({len(FEATURES)} features), ms / {cfg.frame_sec:g}-s frame", value=t_feat)]
_Xb = np.repeat(_B[:1], 1000, axis=0) if len(_B) else np.zeros((1000, len(FEAT_MAIN)), np.float32)
for m in cfg.models:
    t0 = time.time(); score_model(m, FINAL[("K_MAIN", m)]["model"], _Xb)
    COST.append(dict(item=f"{m} inference, ms / frame", value=(time.time() - t0)))
COST = pd.DataFrame(COST); display(COST.round(4)); savetab(COST.round(5), "T20_compute_cost")

if HAS_TEST:
    ERR = pd.DataFrame({m: TEST_SC[("K_MAIN", m)][1] for m in cfg.models}, index=DST.recs)
    ERR["label"] = DST.rec_y
    ERR["n_wrong"] = sum(((ERR[m] >= 0.5).astype(int) != ERR.label).astype(int) for m in cfg.models)
    ERR = ERR.join(records.set_index("record_id")[["duration_s", "n_files"]]) \
        .join(QC_FRAMES.groupby("record_id").keep_C3.mean().rsub(1).rename("rejected_frac")) \
        .join(FEAT["C0_RAW"].groupby("record_id").td_rms_db.median().rename("rms_db"))
    ERR_BAD = ERR[ERR.n_wrong >= max(1, len(cfg.models) // 2)].sort_values("n_wrong", ascending=False)
    display(Markdown(f"**Bảng 21. Bản ghi TEST bị ≥ {max(1, len(cfg.models) // 2)} mô hình phân loại sai.**"))
    display(ERR_BAD.round(3)); savetab(ERR.round(4).reset_index().rename(columns={"index": "record_id"}), "T21_test_error_analysis")
