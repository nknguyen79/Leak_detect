# %% [markdown]
# ## 9. E4 — Bốn mô hình cuối: CV lồng trên DEV + đánh giá một lần trên tập KIỂM TRA KHOÁ
#
# - **CV lồng** (nested CV; Varma & Simon 2006; Cawley & Talbot 2010): vòng ngoài = CV lặp theo bản ghi
#   (như E2–E3); trong mỗi fold ngoài: (i) xếp hạng đồng thuận **lại từ đầu** trên phần huấn luyện và lấy
#   top-`K`; (ii) chọn siêu tham số bằng lưới + CV trong `cfg.inner_splits` fold **nhóm theo bản ghi**;
#   (iii) huấn luyện lại và dự đoán fold ngoài. → ước lượng DEV không thiên lệch do lựa chọn.
# - **Mô hình cuối**: huấn luyện trên toàn bộ DEV với danh sách đặc trưng cuối của E3 và siêu tham số chọn
#   bằng CV nhóm trên DEV → đánh giá **đúng một lần** trên TEST.
# - Hai bộ đặc trưng: `K_MAIN` (bộ làm việc 20–50) và `K_MIN` (bộ tối thiểu, `cfg.run_compact`).
# - **E4c – khẳng định RQ2 trên TEST**: cùng 4 mô hình (mặc định, toàn bộ đặc trưng) huấn luyện trên DEV
#   ở `C0_RAW` và ở `BEST_COND`, so sánh ghép cặp trên cùng các bản ghi TEST (DeLong).

# %%
# =========================================================
# 9.1 CV lồng trên DEV
# =========================================================
def inner_folds(g, y, k, seed):
    recs = np.unique(g); ry = pd.Series(y).groupby(g).first().reindex(recs).values
    k = int(max(2, min(k, np.bincount(ry).min())))
    return [(~np.isin(g, recs[te]), np.isin(g, recs[te]))
            for _, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(recs, ry)]


def tune(model, A, y, g, inner, seed):
    best, best_s = None, -np.inf
    for params in ParameterGrid(PARAM_GRID[model]):
        sc = []
        for tr, va in inner:
            m = fit_model(model, A[tr], y[tr], g[tr], params, seed)
            sc.append(auc_safe(y[va], score_model(model, m, A[va])))
        if np.nanmean(sc) > best_s:
            best, best_s = params, float(np.nanmean(sc))
    return best, best_s


def nested_job(Xtr, ytr, gtr, Xte, K, models, seed):
    prep = Prep().fit(Xtr); A, B = prep.transform(Xtr), prep.transform(Xte)
    idx = consensus([RANKERS[r](A, ytr, gtr, seed) for r in cfg.rankers])[:K]
    A2, B2 = A[:, idx], B[:, idx]
    inner = inner_folds(gtr, ytr, cfg.inner_splits, seed); out = {}
    for m in models:
        best, bs = tune(m, A2, ytr, gtr, inner, seed)
        mdl = fit_model(m, A2, ytr, gtr, best, seed)
        out[m] = (score_model(m, mdl, B2).astype(np.float32), best, bs)
    return out, idx.astype(np.int16)


def run_nested(cond, K, tag):
    ds = DS(cond, "dev"); run = CVRun(ds, cfg.seeds); jobs = []; t0 = time.time()
    for r, s in enumerate(cfg.seeds):
        for fi, te_r in enumerate(FOLDS[("dev", s)]):
            te = np.isin(ds.g, te_r); run.fold_of[r, te] = fi; jobs.append((r, fi, te))
    res = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(nested_job)(ds.X[~te], ds.y[~te], ds.g[~te], ds.X[te], K, tuple(cfg.models), 104729 * cfg.seeds[r] + fi)
        for r, fi, te in jobs)
    params = defaultdict(list)
    for (r, fi, te), (out, idx) in zip(jobs, res):
        for m, (sc, best, bs) in out.items():
            run.scores.setdefault((None, K, m), np.full((len(cfg.seeds), len(ds.y)), np.nan, np.float32))[r, te] = sc
            params[m].append(json.dumps(best, default=str))
        run.orders["consensus"].append(idx)
    print(f"  [{tag}] CV lồng {cond}, K={K}: {len(jobs)} fold ngoài trong {time.time() - t0:.0f} s")
    return run, params


FEATSETS = {"K_MAIN": (K_MAIN, FEAT_MAIN)}
if cfg.run_compact and K_MIN != K_MAIN:
    FEATSETS["K_MIN"] = (K_MIN, FEAT_MIN)
NESTED, E4_ROWS = {}, []
for fs, (K, _) in FEATSETS.items():
    run, params = run_nested(BEST_COND, K, f"E4-{fs}")
    NESTED[fs] = run
    for m in cfg.models:
        s = summarize(run, (None, K, m))
        common = Counter(params[m]).most_common(1)[0]
        E4_ROWS.append(dict(featset=fs, k=K, model=m, **{a: b for a, b in s.items() if not a.startswith("_")},
                            modal_params=common[0], modal_freq=common[1] / len(params[m]), _s=s))
E4_DEV = pd.DataFrame(E4_ROWS)
_c = ["featset", "k", "model", "rec_auc_rep_mean", "rec_auc_rep_sd", "rec_auc", "rec_auc_lo", "rec_auc_hi", "frame_auc",
      "frame_auc_lo", "frame_auc_hi", "rec_ap", "rec_acc", "rec_bacc", "rec_sens", "rec_spec", "rec_f1", "rec_mcc", "modal_params"]
display(Markdown(f"**Bảng 13. Kết quả CV lồng trên DEV ({BEST_COND}).**")); display(E4_DEV[_c].round(4))
savetab(E4_DEV[_c].round(4), "T13_E4_nested_dev")

_pw = []
for fs in FEATSETS:
    sub = E4_DEV[E4_DEV.featset == fs].set_index("model")
    pairs = list(itertools.combinations(cfg.models, 2))
    rows = [dict(featset=fs, A=a, B=b, **compare(sub.at[a, "_s"], sub.at[b, "_s"], NESTED[fs].ds.rec_y)) for a, b in pairs]
    for r, p in zip(rows, holm([r["p_delong"] for r in rows])):
        r["p_delong_holm"] = p
    _pw += rows
E4_PAIR_DEV = pd.DataFrame(_pw); savetab(E4_PAIR_DEV.round(4), "T13b_E4_pairwise_dev")
BEST_MODEL = E4_DEV[E4_DEV.featset == "K_MAIN"].sort_values("rec_auc_rep_mean", ascending=False).model.iloc[0]
print(f"Mô hình tốt nhất trên DEV (K_MAIN): {BEST_MODEL}")
display(E4_PAIR_DEV[["featset", "A", "B", "d_auc", "d_lo", "d_hi", "p_delong_holm", "p_nb"]].round(4))

# %%
# =========================================================
# 9.2 Mô hình cuối (huấn luyện trên toàn bộ DEV) -> đánh giá MỘT LẦN trên TEST
# =========================================================
def final_job(model, A, y, g, B, seed):
    best, bs = tune(model, A, y, g, inner_folds(g, y, cfg.inner_splits, seed), seed)
    mdl = fit_model(model, A, y, g, best, seed)
    return model, mdl, best, (score_model(model, mdl, B) if len(B) else np.array([]))


FINAL, TEST_ROWS, TEST_SC = {}, [], {}
for fs, (K, feats) in FEATSETS.items():
    dsd = DS(BEST_COND, "dev", feats); dst = DS(BEST_COND, "test", feats) if HAS_TEST else None
    prep = Prep().fit(dsd.X); A = prep.transform(dsd.X)
    B = prep.transform(dst.X) if HAS_TEST else np.zeros((0, len(feats)), np.float32)
    outs = Parallel(n_jobs=min(N_JOBS, len(cfg.models)), backend="loky")(
        delayed(final_job)(m, A, dsd.y, dsd.g, B, 2024) for m in cfg.models)
    for m, mdl, best, p in outs:
        FINAL[(fs, m)] = dict(prep=dict(med=prep.med, mu=prep.mu, sd=prep.sd, clip_z=CLIP_Z), model=mdl, features=feats, condition=BEST_COND, params=best, sr=SR,
                              frame_sec=cfg.frame_sec, prior=mdl.prior_, name=m)
        joblib.dump(FINAL[(fs, m)], MODEL_DIR / f"final_{m}_{fs}_k{K}.joblib", compress=3)
        if HAS_TEST:
            rs = rec_scores(p, dst.g, dst.recs); TEST_SC[(fs, m)] = (p, rs)
            lo, hi = boot_auc_ci(dst.rec_y, rs, seed=5)
            flo, fhi = boot_auc_ci(dst.y, p, groups=dst.g, n_boot=max(200, cfg.n_boot // 2), seed=6)
            TEST_ROWS.append(dict(featset=fs, k=K, model=m, n_rec=len(dst.recs), rec_auc=auc_safe(dst.rec_y, rs), rec_auc_lo=lo,
                                  rec_auc_hi=hi, frame_auc=auc_safe(dst.y, p), frame_auc_lo=flo, frame_auc_hi=fhi,
                                  rec_ap=average_precision_score(dst.rec_y, rs),
                                  **{f"rec_{a}": b for a, b in bin_metrics(dst.rec_y, rs).items()}, params=json.dumps(best, default=str)))
if HAS_TEST:
    DST = DS(BEST_COND, "test", FEAT_MAIN)
    E4_TEST = pd.DataFrame(TEST_ROWS)
    display(Markdown(f"**Bảng 14. Kết quả trên TẬP KIỂM TRA KHOÁ ({len(DST.recs)} bản ghi: "
                     f"{DST.rec_y.sum()} leak / {(1 - DST.rec_y).sum()} noleak).**"))
    display(E4_TEST.round(4)); savetab(E4_TEST.round(4), "T14_E4_test")
    _pt = []
    for fs in FEATSETS:
        pairs = list(itertools.combinations(cfg.models, 2))
        rows = []
        for a, b in pairs:
            d, p = delong_test(DST.rec_y, TEST_SC[(fs, a)][1], TEST_SC[(fs, b)][1])
            rows.append(dict(featset=fs, A=a, B=b, d_auc=d, p_delong=p))
        for r, p in zip(rows, holm([r["p_delong"] for r in rows])):
            r["p_delong_holm"] = p
        _pt += rows
    E4_PAIR_TEST = pd.DataFrame(_pt); savetab(E4_PAIR_TEST.round(4), "T14b_E4_pairwise_test")
else:
    print("Không có tập kiểm tra khoá (cfg.holdout_frac = 0): chỉ báo cáo CV lồng trên DEV.")

# %%
# =========================================================
# 9.3 Hình 11-12: ROC/PR mức bản ghi (DEV CV lồng, TEST) và ma trận nhầm lẫn TEST
# =========================================================
def _roc_pr(axr, axp, y, scores_by_model, title):
    for m, s in scores_by_model.items():
        fpr, tpr, _ = roc_curve(y, s); pr, rc, _ = precision_recall_curve(y, s)
        axr.plot(fpr, tpr, color=MODEL_COLOR[m], lw=1.5, label=f"{m} ({auc_safe(y, s):.3f})")
        axp.plot(rc, pr, color=MODEL_COLOR[m], lw=1.5, label=f"{m} ({average_precision_score(y, s):.3f})")
    axr.plot([0, 1], [0, 1], color=GRID, lw=1); axp.axhline(np.mean(y), color=GRID, lw=1)
    axr.set_xlabel("false positive rate"); axr.set_ylabel("true positive rate"); axr.set_title(f"ROC - {title}")
    axp.set_xlabel("recall"); axp.set_ylabel("precision"); axp.set_title(f"PR - {title}")
    axr.legend(loc="lower right", title="AUC"); axp.legend(loc="lower left", title="AP")


ncol = 2 if HAS_TEST else 1
fig, ax = plt.subplots(2, ncol, figsize=(5.4 * ncol, 8.4), squeeze=False)
_dev_sc = {m: E4_DEV[(E4_DEV.featset == "K_MAIN") & (E4_DEV.model == m)]["_s"].iloc[0]["_rec_scores"] for m in cfg.models}
_roc_pr(ax[0, 0], ax[1, 0], NESTED["K_MAIN"].ds.rec_y, _dev_sc, f"DEV nested CV (k={K_MAIN})")
if HAS_TEST:
    _roc_pr(ax[0, 1], ax[1, 1], DST.rec_y, {m: TEST_SC[("K_MAIN", m)][1] for m in cfg.models}, f"TEST (k={K_MAIN})")
fig.tight_layout()
savefig(fig, "Fig11_roc_pr", "Record-level ROC and PR curves (DEV nested CV and hold-out TEST).")

if HAS_TEST:
    fig, ax = plt.subplots(1, len(cfg.models) + 1, figsize=(3.0 * (len(cfg.models) + 1), 3.0),
                           gridspec_kw={"width_ratios": [1] * len(cfg.models) + [1.6]})
    for j, m in enumerate(cfg.models):
        cm = confusion_matrix(DST.rec_y, (TEST_SC[("K_MAIN", m)][1] >= 0.5).astype(int), labels=[0, 1])
        ax[j].imshow(cm, cmap=SEQ, vmin=0, vmax=cm.sum()); ax[j].grid(False)
        for (i, k), v in np.ndenumerate(cm):
            ax[j].text(k, i, str(v), ha="center", va="center", color="white" if v > cm.sum() / 2 else INK, fontsize=11)
        ax[j].set_xticks([0, 1]); ax[j].set_xticklabels(["noleak", "leak"]); ax[j].set_yticks([0, 1])
        ax[j].set_yticklabels(["noleak", "leak"] if j == 0 else ["", ""]); ax[j].set_xlabel("predicted")
        ax[j].set_title(m)
    ax[0].set_ylabel("true")
    a = ax[-1]
    for j, m in enumerate(cfg.models):
        rs = TEST_SC[("K_MAIN", m)][1]
        for c, yv in (("noleak", 0), ("leak", 1)):
            v = rs[DST.rec_y == yv]
            a.scatter(np.full(len(v), j) + (yv - 0.5) * 0.3 + np.random.default_rng(j).uniform(-0.06, 0.06, len(v)), v,
                      s=16, color=PAL[c], edgecolor="white", linewidth=0.5, label=c if j == 0 else None)
    a.axhline(0.5, color=INK2, lw=0.8); a.set_xticks(range(len(cfg.models))); a.set_xticklabels(cfg.models)
    a.set_ylabel("recording score (mean frame probability)"); a.set_title("TEST recordings"); a.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "Fig12_test_confusion_scores", "Hold-out TEST confusion matrices and per-recording scores.")

# %%
# =========================================================
# 9.4 E4c: khẳng định RQ2 trên TEST - C0_RAW vs BEST_COND (toàn bộ đặc trưng, tham số mặc định)
# =========================================================
if HAS_TEST and BEST_COND != "C0_RAW":
    def raw_vs_best_job(cond, m, Xd, yd, gd, Xt, gt, recs_t):
        prep = Prep().fit(Xd)
        mdl = fit_model(m, prep.transform(Xd), yd, gd, seed=11)
        return cond, m, rec_scores(score_model(m, mdl, prep.transform(Xt)), gt, recs_t)
    _dd = {c: (DS(c, "dev"), DS(c, "test")) for c in ("C0_RAW", BEST_COND)}
    outs = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(raw_vs_best_job)(c, m, _dd[c][0].X, _dd[c][0].y, _dd[c][0].g, _dd[c][1].X, _dd[c][1].g, _dd[c][1].recs)
        for c in ("C0_RAW", BEST_COND) for m in cfg.models)
    RS = {(c, m): v for c, m, v in outs}
    rows = []
    for m in cfg.models:
        d, p = delong_test(DST.rec_y, RS[(BEST_COND, m)], RS[("C0_RAW", m)])
        lo, hi, _ = paired_boot_delta(DST.rec_y, RS[(BEST_COND, m)], RS[("C0_RAW", m)], seed=9)
        rows.append(dict(model=m, auc_raw=auc_safe(DST.rec_y, RS[("C0_RAW", m)]), auc_best=auc_safe(DST.rec_y, RS[(BEST_COND, m)]),
                         d_auc=d, d_lo=lo, d_hi=hi, p_delong=p))
    E4C = pd.DataFrame(rows); E4C["p_delong_holm"] = holm(E4C.p_delong.values)
    display(Markdown(f"**Bảng 15. TEST: {BEST_COND} so với C0_RAW (toàn bộ đặc trưng, tham số mặc định).**"))
    display(E4C.round(4)); savetab(E4C.round(4), "T15_E4c_raw_vs_best_test")
else:
    E4C = None
