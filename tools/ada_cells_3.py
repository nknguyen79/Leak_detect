# -*- coding: utf-8 -*-
"""Notebook ADA v1 — phần 3: kết quả, thống kê, động học, không gian đặc trưng, chọn mô hình,
quét λ / ablation, bán giám sát, occlusion theo dải, xuất tư liệu bài báo, kết luận."""

CELLS = []


def md(s):
    CELLS.append(("markdown", s))


def code(s):
    CELLS.append(("code", s))


# =============================================================================
# 9. KẾT QUẢ CHÍNH
# =============================================================================
md(r'''
## 9. Kết quả trên miền đích / Target-domain results (RQ2)

**Ước lượng chính** (giống v7): xác suất OOF của từng frame đích được **trung bình qua các seed**, gộp thành xác suất bản ghi
bằng `p_mean`, rồi tính AUC mức bản ghi; CI 95% bằng bootstrap **bản ghi** (`cfg.bootstrap_n` lần). Cột `AUC (seeds)` là
trung bình ± độ lệch chuẩn giữa các seed (độ bất định do phân hoạch + khởi tạo). ΔAUC so với *source-only* là bootstrap
**ghép cặp** trên cùng các bản ghi; p hiệu chỉnh **Holm** trong mỗi bài toán. *Gap closed* = (AUC − AUC<sub>SO</sub>) /
(AUC<sub>oracle</sub> − AUC<sub>SO</sub>).
''')

code(r'''
# =========================================================
# 9.1 Công cụ tổng hợp + bootstrap / aggregation and bootstrap helpers
# =========================================================
def results_where(variant="main", task=None, method=None, seed=None, shots=None):
    return [r for r in RESULTS.values()
            if r["variant"] == variant and (task is None or r["task"] == task)
            and (method is None or r["method"] == method) and (seed is None or r["seed"] == seed)
            and (shots is None or r.get("shots", 0) == shots)]

def oof_vec(rs, rule):
    v = np.full(N_FRAMES, np.nan, dtype=np.float32)
    for r in rs:
        v[r["te_idx"]] = r["pred"][rule]
    return v

def rec_table(v):
    m = ~np.isnan(v)
    return (pd.DataFrame(dict(uid=uid_all[m], y=y_all[m], p=v[m]))
            .groupby("uid").agg(y=("y", "first"), p=("p", "mean")))

def rec_metrics(g, thr=None):
    thr = cfg.threshold if thr is None else thr
    y = g.y.values.astype(int); p = g.p.values; yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    _, _, f1, _ = precision_recall_fscore_support(y, yh, average="binary", zero_division=0)
    return dict(auc=safe_auc(y, p), bacc=balanced_accuracy_score(y, yh), f1=float(f1),
                mcc=float(matthews_corrcoef(y, yh)), sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
                acc=(tp + tn) / max(len(y), 1), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), n=int(len(y)))

def boot_auc_ci(y, p, n=None, seed=0):
    n = n or cfg.bootstrap_n; y = np.asarray(y); p = np.asarray(p)
    rng = np.random.default_rng(seed); v = []
    for _ in range(int(n)):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            v.append(roc_auc_score(y[i], p[i]))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)

def paired_delta(y, pa, pb, n=None, seed=0):
    # Bootstrap GHÉP CẶP: lấy mẫu lại bản ghi, cùng chỉ số cho hai phương pháp.
    n = n or cfg.bootstrap_n; y = np.asarray(y); pa = np.asarray(pa); pb = np.asarray(pb)
    d0 = safe_auc(y, pa) - safe_auc(y, pb)
    rng = np.random.default_rng(seed); v = []
    for _ in range(int(n)):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            v.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    if not v:
        return d0, np.nan, np.nan, np.nan
    v = np.asarray(v)
    p = float(min(1.0, 2 * min(np.mean(v <= 0), np.mean(v >= 0))))
    return d0, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), p

def holm(p):
    p = np.asarray(p, dtype=float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj

def mcnemar(y, a, b):
    ca, cb = (np.asarray(a) == y), (np.asarray(b) == y)
    n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    if n01 + n10 == 0:
        return n01, n10, 1.0
    return n01, n10, float(spstats.binomtest(min(n01, n10), n01 + n10, .5).pvalue)

def method_summary(task, method, variant="main", rule=None, seeds=None, shots=None):
    rule = rule or cfg.select_rule; seeds = cfg.seeds if seeds is None else seeds
    vs, per, srcv, thr = [], [], [], []
    for s in seeds:
        rs = results_where(variant, task, method, s, shots)
        if not rs:
            continue
        v = oof_vec(rs, rule); vs.append(v)
        per.append(dict(seed=s, **rec_metrics(rec_table(v))))
        srcv += [r["src_val_rec_auc"][rule] for r in rs]
        thr += [r["thr"][rule] for r in rs]
    if not vs:
        return None
    with np.errstate(all="ignore"):
        V = np.nanmean(np.vstack(vs), axis=0) if len(vs) > 1 else vs[0]
    g = rec_table(V)
    # Ngưỡng Youden trên src_val (không dùng nhãn đích) -> phân tích độ nhạy của ngưỡng cố định 0.5
    return dict(g=g, V=V, per=pd.DataFrame(per), ens=rec_metrics(g), src_val=float(np.nanmean(srcv)),
                thr_src=float(np.nanmean(thr)), ens_srcthr=rec_metrics(g, float(np.nanmean(thr))))

METHODS_REPORT = [m for m in ("source_only", "joint_bn", "adabn", "tent", "coral", "mmd", "mcc", "dann", "cdan",
                              "cdan_e", "rcst", "oracle", "target_only") if m in cfg.methods]
REFERENCE_METHODS = ("oracle", "target_only")          # cận trên tham chiếu: không vào họ kiểm định Holm

def _pf(p):
    # p của bootstrap B lần không nhỏ hơn được 1/B -> ghi "< 1/B" thay cho 0.000
    if not np.isfinite(p):
        return "–"
    return f"< {1 / cfg.bootstrap_n:.4f}".rstrip("0") if p < 1 / cfg.bootstrap_n else f"{p:.3f}"
print("Phương pháp báo cáo:", [mlabel(m) for m in METHODS_REPORT])
''')

code(r'''
# =========================================================
# 9.2 BẢNG KẾT QUẢ CHÍNH / Main results table
# =========================================================
def build_main_table(variant="main", rule=None, seeds=None, methods=None, verbose=True):
    rule = rule or cfg.select_rule; methods = methods or METHODS_REPORT
    rows, store = [], {}
    for T in TASKS:
        tgt_uids = set(records.loc[records.domain == T["tgt"], "record_uid"])
        ref = method_summary(T["i"], "source_only", variant, rule, seeds)
        orc = method_summary(T["i"], "oracle", variant, rule, seeds)
        fam = []
        for m in methods:
            S = method_summary(T["i"], m, variant, rule, seeds)
            if S is None:
                continue
            if verbose and set(S["g"].index) != tgt_uids:
                print(f"  !! {T['name']} {m}: phủ {len(S['g'])}/{len(tgt_uids)} bản ghi đích")
            e = S["ens"]; lo, hi = boot_auc_ci(S["g"].y.values, S["g"].p.values, seed=T["i"])
            mk = ~np.isnan(S["V"])
            row = dict(task=T["name"], method_key=m, method=mlabel(m), n_rec=e["n"], AUC=e["auc"], AUC_lo=lo,
                       AUC_hi=hi, AUC_seed_mean=float(S["per"].auc.mean()),
                       AUC_seed_sd=float(S["per"].auc.std(ddof=1)) if len(S["per"]) > 1 else np.nan,
                       BAcc=e["bacc"], F1=e["f1"], MCC=e["mcc"], Sens=e["sens"], Spec=e["spec"],
                       BAcc_srcthr=S["ens_srcthr"]["bacc"], thr_src=S["thr_src"],
                       AUC_frame=safe_auc(y_all[mk], S["V"][mk]), SrcVal_AUC=S["src_val"],
                       dAUC=np.nan, dAUC_lo=np.nan, dAUC_hi=np.nan, p_boot=np.nan, p_holm=np.nan,
                       gap_closed_pct=np.nan)
            if ref is not None and m != "source_only":
                common = S["g"].index.intersection(ref["g"].index)
                d, dlo, dhi, p = paired_delta(S["g"].loc[common, "y"].values, S["g"].loc[common, "p"].values,
                                              ref["g"].loc[common, "p"].values, seed=17 + T["i"])
                row.update(dAUC=d, dAUC_lo=dlo, dAUC_hi=dhi, p_boot=p)
                if m not in REFERENCE_METHODS:
                    fam.append(len(rows))
            if ref is not None and orc is not None and m not in ("source_only",) + REFERENCE_METHODS:
                gap = orc["ens"]["auc"] - ref["ens"]["auc"]
                if gap > 0.02:
                    row["gap_closed_pct"] = 100 * (e["auc"] - ref["ens"]["auc"]) / gap
            rows.append(row); store[(T["i"], m)] = S
        if fam:
            adj = holm([rows[i]["p_boot"] for i in fam])
            for i, a in zip(fam, adj):
                rows[i]["p_holm"] = float(a)
    return pd.DataFrame(rows), store

MAIN, MAIN_STORE = build_main_table()

def _ci(a, lo, hi, nd=3):
    return "–" if not np.isfinite(a) else f"{a:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"

pretty = pd.DataFrame(dict(
    Task=MAIN.task, Method=MAIN.method,
    **{"AUC [95% CI]": [_ci(a, l, h) for a, l, h in zip(MAIN.AUC, MAIN.AUC_lo, MAIN.AUC_hi)],
       "AUC seeds (mean ± sd)": [f"{a:.3f} ± {s:.3f}" if np.isfinite(s) else f"{a:.3f}"
                                 for a, s in zip(MAIN.AUC_seed_mean, MAIN.AUC_seed_sd)],
       "BAcc": MAIN.BAcc.round(3), "BAcc@src-thr": MAIN.BAcc_srcthr.round(3),
       "F1": MAIN.F1.round(3), "MCC": MAIN.MCC.round(3),
       "ΔAUC vs SO [95% CI]": [_ci(a, l, h) for a, l, h in zip(MAIN.dAUC, MAIN.dAUC_lo, MAIN.dAUC_hi)],
       "p (Holm)": [_pf(v) for v in MAIN.p_holm], "Gap closed (%)": MAIN.gap_closed_pct.round(1),
       "Source-val AUC": MAIN.SrcVal_AUC.round(3)}))
MAIN.to_csv(TAB_DIR / "main_results_numeric.csv", index=False)
save_table(pretty, "main_results",
           f"Target-domain recording-level results (checkpoint rule '{cfg.select_rule}', {len(cfg.seeds)} seeds, "
           f"{cfg.record_agg} aggregation, threshold {cfg.threshold}). ΔAUC: paired bootstrap over target recordings vs "
           "source-only, Holm-adjusted within task over the unsupervised methods (references excluded). BAcc@src-thr: balanced accuracy at the Youden threshold of the "
           "source-validation recordings (label-free on target). Source-val AUC: retained performance on held-out "
           "source recordings.", 3)
''')

code(r'''
# =========================================================
# 9.2b [V2] PHÂN RÃ CƠ CHẾ: lợi ích đến từ chuẩn hoá (BatchNorm) hay từ căn chỉnh? / Mechanism decomposition
#      v1: bộ phân biệt ở mức may rủi từ epoch 1, PAD không giảm, quét λ phẳng -> giả thuyết "lô trộn nguồn+đích
#      qua BatchNorm" (joint-BN) mới là nguồn lợi ích chính. Các hiệu ΔAUC ghép cặp dưới đây kiểm định trực tiếp.
# =========================================================
MECH_PAIRS = [("joint_bn", "source_only", "mixed source+target batches through BatchNorm, no alignment loss"),
              ("adabn", "source_only", "post-hoc BatchNorm re-estimation only"),
              ("tent", "adabn", "test-time entropy minimisation on BN affine parameters"),
              ("coral", "joint_bn", "second-order alignment beyond joint-BN"),
              ("mcc", "joint_bn", "class-confusion minimisation beyond joint-BN"),
              ("dann", "joint_bn", "adversarial alignment beyond joint-BN"),
              ("cdan_e", "joint_bn", "conditional adversarial alignment beyond joint-BN"),
              ("rcst", "dann", "record-consistent self-training beyond DANN"),
              ("rcst", "source_only", "total gain of the proposed method")]
_mr = []
for T in TASKS:
    for a, b, meaning in MECH_PAIRS:
        A, B = MAIN_STORE.get((T["i"], a)), MAIN_STORE.get((T["i"], b))
        if A is None or B is None:
            continue
        cm = A["g"].index.intersection(B["g"].index)
        d, lo, hi, p = paired_delta(A["g"].loc[cm, "y"].values, A["g"].loc[cm, "p"].values,
                                    B["g"].loc[cm, "p"].values, seed=31)
        _mr.append(dict(task=T["name"], contrast=f"{mlabel(a)} − {mlabel(b)}", meaning=meaning,
                        dAUC=d, lo=lo, hi=hi, p_boot=p))
MECH = pd.DataFrame(_mr)
if len(MECH):
    MECH["p_holm"] = np.concatenate([holm(g.p_boot.values) for _, g in MECH.groupby("task", sort=False)])
    save_table(MECH, "mechanism_decomposition",
               "Mechanism decomposition: paired recording-level ΔAUC (bootstrap over target recordings) isolating the "
               "contribution of target-aware normalisation (joint-BN, AdaBN, TENT), of each alignment objective on top of "
               "joint-BN, and of record-consistent self-training on top of DANN. Holm correction within task.", 3)
    for T in TASKS:
        d = MECH[MECH.task == T["name"]]
        for _, r in d.iterrows():
            sig = np.isfinite(r.lo) and (r.lo > 0 or r.hi < 0) and r.p_holm < .05
            print(f"  {T['name']:>18s} | {r.contrast:<34s} ΔAUC {r.dAUC:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] "
                  f"p_Holm {_pf(r.p_holm):>8s} {'*' if sig else ''}")
    fig, axes = plt.subplots(1, len(TASKS), figsize=(6.5 * len(TASKS), 0.45 * len(MECH_PAIRS) + 1.6), squeeze=False)
    for ti, T in enumerate(TASKS):
        ax = axes[0, ti]; d = MECH[MECH.task == T["name"]].reset_index(drop=True)
        for i, r in d.iterrows():
            sig = np.isfinite(r.lo) and (r.lo > 0 or r.hi < 0) and r.p_holm < .05
            ax.errorbar(r.dAUC, i, xerr=[[r.dAUC - r.lo], [r.hi - r.dAUC]], fmt="D" if sig else "o", capsize=3,
                        color="#D62828" if "RCST" in r.contrast else "#2E6F95", mfc=None if sig else "white")
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(range(len(d))); ax.set_yticklabels(d.contrast, fontsize=8); ax.invert_yaxis()
        ax.set_xlabel("paired ΔAUC (target recordings)"); ax.set_title(T["name"])
    savefig(fig, "mechanism_decomposition",
            "Where does the adaptation gain come from? Paired ΔAUC for normalisation-only controls, for each alignment "
            "objective beyond joint-BN, and for record-consistent self-training beyond DANN (filled = Holm p < 0.05).")
''')

code(r'''
# =========================================================
# 9.3 Hình: AUC đích theo phương pháp + ΔAUC (forest) / target AUC and paired ΔAUC
# =========================================================
nT = len(TASKS)
fig, axes = plt.subplots(nT, 2, figsize=(13, 0.55 * len(METHODS_REPORT) * nT + 1.5), squeeze=False)
for ti, T in enumerate(TASKS):
    d = MAIN[MAIN.task == T["name"]].reset_index(drop=True)
    ax = axes[ti, 0]
    for i, r in d.iterrows():
        c = METHOD_COLOR.get(r.method_key, "k")
        ax.errorbar(r.AUC, i, xerr=[[r.AUC - r.AUC_lo], [r.AUC_hi - r.AUC]], fmt="o", color=c, capsize=3, ms=7)
        per = MAIN_STORE[(T["i"], r.method_key)]["per"].auc.values
        ax.scatter(per, np.full(len(per), i) + .22, s=12, color=c, alpha=.6, marker="|")
    so = d.loc[d.method_key == "source_only", "AUC"]
    if len(so):
        ax.axvline(float(so.iloc[0]), color=METHOD_COLOR["source_only"], ls="--", lw=1)
    ax.axvline(.5, color="k", lw=.6, ls=":")
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.method); ax.invert_yaxis()
    ax.set_xlabel("target record-level AUC"); ax.set_title(f"({'abcdefgh'[2 * ti]}) {T['name']}: AUC [95% CI], | = seeds")
    ax = axes[ti, 1]
    dd = d[d.method_key != "source_only"].reset_index(drop=True)
    for i, r in dd.iterrows():
        sig = np.isfinite(r.p_holm) and r.p_holm < .05
        ax.errorbar(r.dAUC, i, xerr=[[r.dAUC - r.dAUC_lo], [r.dAUC_hi - r.dAUC]], fmt="D" if sig else "o",
                    color=METHOD_COLOR.get(r.method_key, "k"), capsize=3, ms=6, mfc=None if sig else "white")
        ax.text(1.01, i, f"p={r.p_holm:.3f}" if np.isfinite(r.p_holm) else "", va="center", fontsize=8,
                transform=ax.get_yaxis_transform(), clip_on=False)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(range(len(dd))); ax.set_yticklabels(dd.method); ax.invert_yaxis()
    ax.set_xlabel("ΔAUC vs source-only (paired bootstrap)")
    ax.set_title(f"({'abcdefgh'[2 * ti + 1]}) improvement over source-only (filled = Holm p < 0.05)")
savefig(fig, "target_auc_forest",
        "Target-domain recording-level AUC per method with 95% bootstrap CIs and per-seed values (left), and paired "
        "ΔAUC relative to source-only with Holm-adjusted p-values (right). Dashed line: source-only AUC.")

if nT >= 2:
    H_ = MAIN.pivot(index="method", columns="task", values="AUC").reindex([mlabel(m) for m in METHODS_REPORT])
    fig, ax = plt.subplots(figsize=(1.6 + 1.8 * nT, 0.45 * len(H_) + 1.2))
    im = ax.imshow(H_.values, cmap="viridis", vmin=max(.5, np.nanmin(H_.values) - .02), vmax=1)
    for i in range(H_.shape[0]):
        for j in range(H_.shape[1]):
            ax.text(j, i, f"{H_.values[i, j]:.3f}", ha="center", va="center",
                    color="w" if H_.values[i, j] < np.nanmean(H_.values) else "k", fontsize=9)
    ax.set_xticks(range(H_.shape[1])); ax.set_xticklabels(H_.columns, rotation=15)
    ax.set_yticks(range(H_.shape[0])); ax.set_yticklabels(H_.index); ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=.046, label="target AUC")
    savefig(fig, "auc_heatmap_tasks", "Target record-level AUC of every method on every transfer task.")
''')

code(r'''
# =========================================================
# 9.4 ROC mức bản ghi + ma trận nhầm lẫn / record-level ROC curves and confusion matrices
# =========================================================
fig, axes = plt.subplots(1, nT, figsize=(5.2 * nT, 4.6), squeeze=False)
for ti, T in enumerate(TASKS):
    ax = axes[0, ti]
    for m in METHODS_REPORT:
        S = MAIN_STORE.get((T["i"], m))
        if S is None:
            continue
        fpr, tpr, _ = roc_curve(S["g"].y, S["g"].p)
        ax.plot(fpr, tpr, color=METHOD_COLOR.get(m, "k"), lw=2.2 if m in ADV_METHODS else 1.3,
                ls="--" if m in ("source_only", "oracle") else "-", label=f"{mlabel(m)} ({S['ens']['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="k", lw=.6, ls=":")
    ax.set_xlabel("false-positive rate (recordings)"); ax.set_ylabel("true-positive rate")
    ax.set_title(T["name"]); ax.legend(fontsize=8, loc="lower right")
savefig(fig, "roc_target", "Record-level ROC curves on the target domain (seed-averaged out-of-fold probabilities).")

fig, axes = plt.subplots(nT, len(METHODS_REPORT), figsize=(2.3 * len(METHODS_REPORT), 2.4 * nT), squeeze=False)
for ti, T in enumerate(TASKS):
    for mi, m in enumerate(METHODS_REPORT):
        ax = axes[ti, mi]; S = MAIN_STORE.get((T["i"], m))
        if S is None:
            ax.axis("off"); continue
        e = S["ens"]; C = np.array([[e["tn"], e["fp"]], [e["fn"], e["tp"]]])
        ax.imshow(C, cmap="Blues", vmin=0, vmax=max(1, C.max()))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(C[i, j]), ha="center", va="center",
                        color="w" if C[i, j] > C.max() / 2 else "k", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["noleak", "leak"], fontsize=7)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["noleak", "leak"], fontsize=7); ax.grid(False)
        ax.set_title(f"{mlabel(m)}\nBAcc {e['bacc']:.2f}", fontsize=8)
        if mi == 0:
            ax.set_ylabel(f"{T['name']}\ntrue", fontsize=8)
savefig(fig, "confusion_target",
        f"Record-level confusion matrices on the target domain at the pre-registered threshold {cfg.threshold} "
        "(rows: true class; columns: predicted).")

# McNemar mức bản ghi vs source-only (bổ sung cho ΔAUC; dùng quyết định ở ngưỡng cố định)
mc = []
for T in TASKS:
    ref = MAIN_STORE.get((T["i"], "source_only"))
    if ref is None:
        continue
    for m in METHODS_REPORT:
        S = MAIN_STORE.get((T["i"], m))
        if S is None or m == "source_only":
            continue
        common = S["g"].index.intersection(ref["g"].index)
        y = S["g"].loc[common, "y"].values.astype(int)
        n01, n10, p = mcnemar(y, (S["g"].loc[common, "p"].values >= cfg.threshold).astype(int),
                              (ref["g"].loc[common, "p"].values >= cfg.threshold).astype(int))
        mc.append(dict(task=T["name"], method=mlabel(m), only_method_correct=n01, only_SO_correct=n10, p_exact=p))
if mc:
    MC = pd.DataFrame(mc)
    MC["p_holm"] = np.concatenate([holm(g.p_exact.values) for _, g in MC.groupby("task", sort=False)])
    save_table(MC, "mcnemar_vs_source_only",
               "Exact McNemar test on record-level decisions (threshold fixed in advance) of each method versus "
               "source-only; Holm correction within task.", 4)
''')

# =============================================================================
# 10. ĐỘNG HỌC
# =============================================================================
md(r'''
## 10. Động học huấn luyện đối nghịch / Adversarial training dynamics (RQ3)

Dấu hiệu một lượt DANN "lành mạnh": độ chính xác của $G_d$ **giảm về ≈ 0.5** khi $\lambda_p$ tăng (đặc trưng hai miền
không còn phân biệt được), trong khi mất mát phân lớp nguồn vẫn giảm. PAD của đặc trưng (đo bằng bộ phân lớp tuyến tính
độc lập, mục 7.3) phải **thấp hơn** source-only. Đường AUC đích theo epoch dùng nhãn đích nên **chỉ để quan sát**.
''')

code(r'''
# =========================================================
# 10.1 Mất mát, độ chính xác bộ phân biệt (lô huấn luyện và CHẾ ĐỘ EVAL), λ_p, chất lượng nhãn giả RCST
# =========================================================
s0 = cfg.seeds[0]
def _first_run(m):
    rs = [r for r in results_where("main", 0, m, s0) if r.get("ep_hist") is not None and r["fold"] in (0, None)]
    return rs[0] if rs else None
fig, axes = plt.subplots(2, 3, figsize=(17, 7.2)); axes = axes.ravel()
for m in [x for x in METHODS_REPORT if x not in POSTHOC_METHODS]:
    r = _first_run(m)
    if r is None:
        continue
    E, H = r["ep_hist"], r["hist"]; c = METHOD_COLOR.get(m, "k")
    axes[0].plot(E.epoch, E.loss_cls, color=c, label=mlabel(m))
    if m in ADV_METHODS:
        axes[1].plot(E.epoch, E.loss_da, color=c, label=mlabel(m))
        axes[2].plot(E.epoch, E.disc_acc, color=c, ls=":", lw=1)
        if "disc_acc_eval" in H:
            axes[2].plot(H.epoch, H.disc_acc_eval, color=c, marker="o", ms=3, label=f"{mlabel(m)} (eval)")
    elif m in ("coral", "mmd", "mcc"):
        axes[1].plot(E.epoch, E.loss_da, color=c, ls="--", label=f"{mlabel(m)} (weighted)")
    if m in ADV_METHODS or m in ("coral", "mcc"):
        axes[3].plot(E.epoch, E.lam, color=c, label=mlabel(m))
    if m == "rcst":
        if "pl_acc_rec" in E and E.pl_acc_rec.notna().any():
            axes[4].plot(E.epoch, E.pl_acc_rec, color=c, marker="o", ms=3, label="record-level pseudo-labels")
            axes[4].plot(E.epoch, E.pl_acc_frame, color="#555555", marker="s", ms=3, ls="--",
                         label="frame-level, same coverage")
            ax2 = axes[4].twinx(); ax2.plot(E.epoch, E.pl_cov, color="#BBBBBB", lw=1)
            ax2.set_ylim(0, 1); ax2.set_ylabel("coverage (fraction of records)", color="#888888"); ax2.grid(False)
        axes[5].plot(E.epoch, E.loss_st, color=c, label="pseudo-label + consistency loss")
axes[0].set_title("(a) label loss $\\mathcal{L}_y$ (train)"); axes[0].legend(fontsize=7)
axes[1].set_title("(b) domain loss $\\mathcal{L}_d$ / alignment loss"); axes[1].axhline(math.log(2), color="k", ls=":", lw=.8)
axes[1].legend(fontsize=7)
axes[2].set_title("(c) discriminator accuracy: train batches (dotted) vs eval mode"); axes[2].axhline(.5, color="k", ls=":", lw=.8)
axes[2].set_ylim(.3, 1.02); axes[2].legend(fontsize=7)
axes[3].set_title("(d) adaptation weight λ_p"); axes[3].legend(fontsize=7)
axes[4].set_title("(e) RCST pseudo-label accuracy [monitor, uses target labels]"); axes[4].set_ylim(0, 1.02)
axes[4].legend(fontsize=7, loc="lower left")
axes[5].set_title("(f) RCST self-training loss"); axes[5].legend(fontsize=7)
for a_ in axes:
    a_.set_xlabel("epoch")
savefig(fig, "adversarial_dynamics",
        f"Training dynamics on {TASKS[0]['name']} (seed {s0}, fold 0): (a) source label loss; (b) domain-discriminator loss "
        "(dotted: ln 2) or weighted alignment loss; (c) discriminator accuracy on noisy training batches (dotted) and in "
        "eval mode on held-out features (markers); (d) λ_p; (e) accuracy of record-level versus frame-level pseudo-labels "
        "at equal coverage (grey: fraction of target recordings pseudo-labelled); (f) RCST self-training loss.")
''')

code(r'''
# =========================================================
# 10.2 AUC đích [giám sát] và PAD theo epoch, trung bình qua fold × seed / monitored target AUC and PAD
# =========================================================
fig, axes = plt.subplots(nT, 3, figsize=(16, 3.4 * nT), squeeze=False)
for ti, T in enumerate(TASKS):
    for m in [x for x in METHODS_REPORT if x != "adabn"]:
        rs = [r for r in results_where("main", T["i"], m) if r.get("hist") is not None]
        if not rs:
            continue
        H = pd.concat([r["hist"].assign(job=r["key"]) for r in rs])
        agg = H.groupby("epoch").agg(a=("tgt_auc_rec", "mean"), sa=("tgt_auc_rec", "std"),
                                     pad_m=("pad", "mean"), sv=("src_val_rec_auc", "mean"))
        # [lưu ý] không đặt tên cột "pad": ở pandas 2.x `agg.pad` là phương thức DataFrame.pad, không phải cột
        c = METHOD_COLOR.get(m, "k")
        axes[ti, 0].plot(agg.index, agg.a, color=c, label=mlabel(m))
        axes[ti, 0].fill_between(agg.index, agg.a - agg.sa.fillna(0), agg.a + agg.sa.fillna(0), color=c, alpha=.12)
        axes[ti, 1].plot(agg.index, agg["pad_m"], color=c, label=mlabel(m))
        axes[ti, 2].plot(agg.index, agg.sv, color=c, label=mlabel(m))
    for a in axes[ti]:
        a.axvline(SELECT_FROM_EPOCH, color="k", ls=":", lw=.8); a.set_xlabel("epoch")
    axes[ti, 0].set_title(f"{T['name']}: target rec-AUC [monitor only]"); axes[ti, 0].legend(fontsize=7)
    axes[ti, 0].set_ylim(-.02, 1.02); axes[ti, 2].set_ylim(-.02, 1.02)
    axes[ti, 1].set_title("proxy A-distance of f (source vs target)"); axes[ti, 1].set_ylim(-.05, 2.05)
    axes[ti, 2].set_title("source-validation rec-AUC")
savefig(fig, "epoch_curves",
        "Per-epoch behaviour averaged over folds and seeds (band: ±1 sd): target recording-level AUC (uses target labels, "
        "shown for monitoring only), proxy A-distance between source and target bottleneck features, and "
        "source-validation AUC. Dotted line: earliest selectable epoch.")
''')

# =============================================================================
# 11. KHÔNG GIAN ĐẶC TRƯNG
# =============================================================================
md(r'''
## 11. Không gian đặc trưng sau thích ứng / Feature-space analysis (RQ3)

Dùng các mô hình đã giữ (bài toán đầu, seed đầu, fold 0). Với mỗi phương pháp: **PAD** tổng (nguồn vs đích),
**PAD theo lớp** (căn chỉnh *có điều kiện*: leak-nguồn vs leak-đích, noleak-nguồn vs noleak-đích — dùng nhãn đích, chỉ để
phân tích), **MMD** Gauss, và **silhouette** theo lớp của đặc trưng đích (đặc trưng đích còn tách lớp không?).
Tham chiếu "input" = thống kê log-Mel (mục 3.7). Lý tưởng: PAD tổng giảm **và** silhouette lớp đích không giảm.
''')

code(r'''
# =========================================================
# 11.1 Mô hình đã giữ + đặc trưng + PAD/MMD/silhouette / kept models, features and discrepancy table
# =========================================================
FS_T, FS_S = 0, cfg.seeds[0]
P_fs = PARTS[(FS_T, FS_S)]
tgt_tr0, tgt_te0 = P_fs["fold_idx"][0]
KEPT = {}
_so = RESULTS.get(job_key(dict(variant="main", task=FS_T, seed=FS_S, fold=None, method="source_only")))
if _so is not None and _so["state"] is not None:
    KEPT["source_only"] = load_state(LeakNet().to(DEVICE), _so["state"]).eval()
    if "adabn" in cfg.methods:
        KEPT["adabn"] = adabn_model(_so["state"], tgt_tr0, DEVICE)
    if "tent" in cfg.methods:
        KEPT["tent"] = tent_model(_so["state"], tgt_tr0, DEVICE)
for m in TRAINED:
    r = RESULTS.get(job_key(dict(variant="main", task=FS_T, seed=FS_S, fold=0, method=m)))
    if r is not None and r["state"] is not None:
        KEPT[m] = load_state(LeakNet().to(DEVICE), r["state"]).eval()
KEPT = {m: KEPT[m] for m in METHODS_REPORT if m in KEPT}

_half = max(50, cfg.tsne_max_points // 2)
idx_S = subsample(np.concatenate([P_fs["src_tr_idx"], P_fs["src_va_idx"]]), _half, 11)
idx_T = subsample(P_fs["tgt_all_idx"], _half, 12)
yS, yT = y_all[idx_S], y_all[idx_T]

def gauss_mmd(A, B, n=600, seed=0):
    rng = np.random.default_rng(seed)
    A = A[rng.choice(len(A), min(n, len(A)), replace=False)]; B = B[rng.choice(len(B), min(n, len(B)), replace=False)]
    X = np.vstack([A, B]); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    K = np.exp(-D / (np.median(D[D > 0]) + 1e-9)); na = len(A)
    return float(K[:na, :na].mean() + K[na:, na:].mean() - 2 * K[:na, na:].mean())

FEATS, fs_rows = {}, []
FEATS["input (log-Mel stats)"] = (FEAT_IN[idx_S], FEAT_IN[idx_T])
for m, net in KEPT.items():
    FEATS[mlabel(m)] = (infer(net, idx_S, DEVICE, want_feat=True)[1], infer(net, idx_T, DEVICE, want_feat=True)[1])
for name, (A, B) in FEATS.items():
    sil = (silhouette_score(StandardScaler().fit_transform(B), yT)
           if len(np.unique(yT)) == 2 and len(B) > 10 else np.nan)
    fs_rows.append(dict(representation=name, PAD=pad_score(A, B, k=5),
                        PAD_leak=pad_score(A[yS == 1], B[yT == 1], k=5),
                        PAD_noleak=pad_score(A[yS == 0], B[yT == 0], k=5),
                        MMD=gauss_mmd(A, B), target_class_silhouette=sil))
FS_TAB = pd.DataFrame(fs_rows)
save_table(FS_TAB, "feature_discrepancy",
           f"Domain discrepancy of learned features on {TASKS[FS_T]['name']} (seed {FS_S}, fold-0 models; "
           f"{len(idx_S)} source and {len(idx_T)} target frames): proxy A-distance overall and per class, Gaussian MMD, "
           "and class silhouette of target features (higher = target classes better separated).", 3)
''')

code(r'''
# =========================================================
# 11.2 t-SNE trước/sau thích ứng + biểu đồ PAD / t-SNE before vs after adaptation, PAD bars
# =========================================================
names = list(FEATS.keys())
fig, axes = plt.subplots(2, len(names), figsize=(3.3 * len(names), 6.6), squeeze=False)
for j, name in enumerate(names):
    A, B = FEATS[name]
    Z = TSNE(2, init="pca", random_state=0,
             perplexity=min(30, max(5, (len(A) + len(B)) // 5))).fit_transform(
        StandardScaler().fit_transform(np.vstack([A, B])))
    za, zb = Z[:len(A)], Z[len(A):]
    ax = axes[0, j]
    ax.scatter(za[:, 0], za[:, 1], s=4, color=PAL["src"], alpha=.45, label="source")
    ax.scatter(zb[:, 0], zb[:, 1], s=4, color=PAL["tgt"], alpha=.45, label="target")
    pad = FS_TAB.loc[FS_TAB.representation == name, "PAD"].values[0]
    ax.set_title(f"{name}\nPAD = {pad:.2f}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    if j == 0:
        ax.legend(markerscale=3, fontsize=7); ax.set_ylabel("colour = domain")
    ax = axes[1, j]
    for lab, mk, c in [(0, "o", PAL["noleak"]), (1, "^", PAL["leak"])]:
        ax.scatter(za[yS == lab, 0], za[yS == lab, 1], s=4, marker=mk, color=c, alpha=.15)
        ax.scatter(zb[yT == lab, 0], zb[yT == lab, 1], s=7, marker=mk, color=c, alpha=.7,
                   label=f"target {'leak' if lab else 'noleak'}")
    sil = FS_TAB.loc[FS_TAB.representation == name, "target_class_silhouette"].values[0]
    ax.set_title(f"target silhouette = {sil:.2f}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    if j == 0:
        ax.legend(markerscale=2, fontsize=7); ax.set_ylabel("colour = class (faint = source)")
savefig(fig, "tsne_adaptation",
        f"t-SNE of bottleneck features on {TASKS[FS_T]['name']}: top row coloured by domain (source vs target), bottom row "
        "by class (target highlighted). Successful adaptation mixes the domains (lower PAD) while keeping target classes "
        "apart.")

fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
xx = np.arange(len(FS_TAB)); cols_ = ["#999999"] + [METHOD_COLOR.get(m, "k") for m in KEPT]
ax[0].bar(xx, FS_TAB.PAD, color=cols_[:len(FS_TAB)])
ax[0].set_xticks(xx); ax[0].set_xticklabels(FS_TAB.representation, rotation=25, ha="right", fontsize=8)
ax[0].set_ylabel("proxy A-distance"); ax[0].set_ylim(0, 2.05); ax[0].set_title("(a) Overall domain discrepancy")
w = .38
ax[1].bar(xx - w / 2, FS_TAB.PAD_noleak, w, color=PAL["noleak"], label="no-leak")
ax[1].bar(xx + w / 2, FS_TAB.PAD_leak, w, color=PAL["leak"], label="leak")
ax[1].set_xticks(xx); ax[1].set_xticklabels(FS_TAB.representation, rotation=25, ha="right", fontsize=8)
ax[1].set_ylim(0, 2.05); ax[1].set_title("(b) Class-conditional discrepancy"); ax[1].legend()
savefig(fig, "pad_bars",
        "Proxy A-distance between source and target features: (a) overall, (b) within each class. "
        "Class-conditional alignment (b) is what transfers the decision boundary.")
''')

# =============================================================================
# 12. CHỌN MÔ HÌNH KHÔNG NHÃN ĐÍCH
# =============================================================================
md(r'''
## 12. Chọn mô hình khi KHÔNG có nhãn đích / Model selection without target labels (RQ4)

| Quy tắc | Dùng gì | Hợp lệ cho UDA? |
|---|---|---|
| `last` | epoch cuối của lịch cố định | **có** (quy tắc chính, đăng ký trước) |
| `src_val` | BCE nhỏ nhất trên `src_val` | có |
| `dev` | rủi ro DEV nhỏ nhất (You et al., 2019) | có |
| `oracle` | AUC bản ghi **đích** lớn nhất | **không** — chỉ để đo mức lạc quan |

"Optimism" = AUC(`oracle`) − AUC(quy tắc chính): phần kết quả mà một bài báo sẽ thổi phồng nếu chọn epoch bằng nhãn đích.
''')

code(r'''
# =========================================================
# 12.1 AUC theo quy tắc chọn + tương quan tiêu chí với AUC đích / rule comparison and criterion validity
# =========================================================
sel_rows = []
for T in TASKS:
    for m in METHODS_REPORT:
        r_ = {}
        for rule in RULES:
            S = method_summary(T["i"], m, "main", rule)
            if S is not None:
                r_[rule] = S["ens"]["auc"]
        if r_:
            sel_rows.append(dict(task=T["name"], method=mlabel(m), **{f"AUC_{k}": v for k, v in r_.items()},
                                 optimism=r_.get("oracle", np.nan) - r_.get(cfg.select_rule, np.nan)))
SEL = pd.DataFrame(sel_rows)
save_table(SEL, "selection_rules",
           f"Target record-level AUC under different checkpoint-selection rules. 'oracle' uses target labels and is "
           f"reported only to quantify optimism (= AUC_oracle − AUC_{cfg.select_rule}).", 3)

corr = []
for m in [x for x in METHODS_REPORT if x != "adabn"]:
    for crit, sign in [("src_val_bce", -1), ("dev", -1), ("pad", -1), ("tgt_entropy", -1)]:
        rhos = []
        for r in results_where("main", None, m):
            H = r.get("hist")
            if H is None or len(H) < 4:
                continue
            a, b = sign * H[crit].values, H.tgt_auc_rec.values
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= 4 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
                rhos.append(spstats.spearmanr(a[ok], b[ok]).correlation)
        if rhos:
            corr.append(dict(method=mlabel(m), criterion=f"−{crit}", n_runs=len(rhos),
                             spearman_mean=float(np.mean(rhos)), spearman_sd=float(np.std(rhos))))
if corr:
    CORR = pd.DataFrame(corr)
    save_table(CORR, "selection_criterion_validity",
               "Within-run Spearman correlation across evaluation epochs between each label-free criterion (sign-flipped "
               "so that higher is better) and the monitored target recording-level AUC. Values near zero mean the "
               "criterion cannot guide selection.", 3)

fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
d0 = SEL[SEL.task == TASKS[0]["name"]].reset_index(drop=True)
mk = {"last": "o", "src_val": "s", "dev": "D", "oracle": "*"}
dy_ = {"last": -.24, "src_val": -.08, "dev": .08, "oracle": .24}
for i, r in d0.iterrows():
    for rule in RULES:
        v = r.get(f"AUC_{rule}", np.nan)
        if np.isfinite(v):
            ax[0].scatter(v, i + dy_[rule], marker=mk[rule], s=90 if rule == "oracle" else 45,
                          color="#C8553D" if rule == "oracle" else "#2E6F95", alpha=.85,
                          label=rule if i == 0 else None)
ax[0].set_yticks(range(len(d0))); ax[0].set_yticklabels(d0.method); ax[0].invert_yaxis()
ax[0].set_xlabel("target record-level AUC"); ax[0].legend(fontsize=8)
ax[0].set_title(f"(a) {TASKS[0]['name']}: selection rule (★ uses target labels)")
_m = "dann" if "dann" in cfg.methods else (adv_present[0] if adv_present else None)
if _m:
    for r in results_where("main", 0, _m):
        H = r["hist"]
        ax[1].scatter(H.dev, H.tgt_auc_rec, s=12, alpha=.6)
    ax[1].set_xlabel("DEV risk (label-free, lower = better)"); ax[1].set_ylabel("target rec-AUC [monitor]")
    ax[1].set_title(f"(b) {mlabel(_m)}: DEV vs target AUC across epochs (colour = run)")
savefig(fig, "model_selection",
        "Model selection without target labels: (a) target AUC obtained with each checkpoint rule; (b) relationship "
        "between the label-free DEV criterion and the monitored target AUC across epochs and runs.")
''')

# =============================================================================
# 13. ĐỘ NHẠY & ABLATION
# =============================================================================
md(r'''
## 13. Độ nhạy theo λ và ablation / λ sensitivity and ablation (RQ4)

Chạy trên bài toán `cfg.sweep_task`, các seed `cfg.sweep_seeds`, **mọi fold** đích. Mọi so sánh dùng **cùng seed** với lượt
chính. Biến thể ablation (DANN): bỏ FIR ngẫu nhiên (`no_fir`), bỏ mọi augmentation (`no_aug`), bỏ lấy mẫu nguồn cân bằng
lớp (`no_balance`); CDAN **không** trọng số entropy (`cdan_noent`) so với CDAN+E.
''')

code(r'''
# =========================================================
# 13.1 Quét λ_max cho DANN / lambda sweep
# =========================================================
# [V2.1] so sánh GHÉP CẶP với lượt chính -> chỉ dùng seed có trong cfg.seeds (an toàn khi chạy theo giai đoạn)
SW_T, SW_S = cfg.sweep_task, tuple(s_ for s_ in cfg.sweep_seeds if s_ in cfg.seeds) or (cfg.seeds[0],)
LAMBDAS = sorted(set((cfg.lambda_max,) + tuple(cfg.lambda_grid)))
if cfg.run_lambda_sweep and "dann" in cfg.methods:
    jobs = []
    for lm in LAMBDAS:
        if lm != cfg.lambda_max:
            jobs += main_jobs(variant=f"lam{lm:g}", methods=["dann"], seeds=SW_S, tasks=[SW_T], hp={"lambda_max": lm})
    run_jobs(jobs, "lambda sweep")
    ref = method_summary(SW_T, "source_only", "main", seeds=SW_S)
    orc = method_summary(SW_T, "oracle", "main", seeds=SW_S)
    lam_rows = []
    for lm in LAMBDAS:
        S = method_summary(SW_T, "dann", "main" if lm == cfg.lambda_max else f"lam{lm:g}", seeds=SW_S)
        if S is None:
            continue
        lo, hi = boot_auc_ci(S["g"].y, S["g"].p, seed=3)
        lam_rows.append(dict(lambda_max=lm, AUC=S["ens"]["auc"], lo=lo, hi=hi, BAcc=S["ens"]["bacc"],
                             src_val_AUC=S["src_val"]))
    LAM = pd.DataFrame(lam_rows)
    save_table(LAM, "lambda_sweep",
               f"Sensitivity of DANN to the final adversarial weight λ_max on {TASKS[SW_T]['name']} "
               f"(seeds {list(SW_S)}, all target folds).", 3)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.errorbar(LAM.lambda_max, LAM.AUC, yerr=[LAM.AUC - LAM.lo, LAM.hi - LAM.AUC], fmt="o-",
                color=METHOD_COLOR["dann"], capsize=4, label="DANN (target)")
    ax.plot(LAM.lambda_max, LAM.src_val_AUC, "s:", color="#555555", label="DANN (source-val)")
    if ref is not None:
        ax.axhline(ref["ens"]["auc"], color=METHOD_COLOR["source_only"], ls="--", label="source-only (target)")
    if orc is not None:
        ax.axhline(orc["ens"]["auc"], color=METHOD_COLOR["oracle"], ls="-.", label="oracle (target)")
    ax.set_xscale("log"); ax.set_xlabel("λ_max"); ax.set_ylabel("record-level AUC"); ax.legend(fontsize=8)
    ax.set_title(f"λ sensitivity — {TASKS[SW_T]['name']}")
    savefig(fig, "lambda_sensitivity",
            "Target AUC (95% CI) and source-validation AUC of DANN as a function of the final adversarial weight λ_max.")
else:
    print("Bỏ qua quét λ (cfg.run_lambda_sweep=False hoặc không có DANN).")
''')

code(r'''
# =========================================================
# 13.2 Ablation / ablation study
# =========================================================
ABLATIONS = {
    "weak_disc": ("dann", {"disc_hidden": 256, "disc_lr_mult": 1.0}, "dann"),     # bộ phân biệt của v1
    "no_chain_aug": ("rcst", {"chain_p": 0.0}, "rcst"),
    "rcst_no_pl": ("rcst", {"rcst_pl": False}, "rcst"),
    "rcst_no_cons": ("rcst", {"rcst_cons": False}, "rcst"),
    "so_no_chain_aug": ("source_only", {"chain_p": 0.0}, "source_only"),
    "no_fir": ("dann", {"fir_p": 0.0}, "dann"), "no_aug": ("dann", {"augment": False}, "dann"),
    "no_balance": ("dann", {"balanced": False}, "dann"), "cdan_noent": ("cdan", {}, "cdan_e")}
AB_S = tuple(s_ for s_ in cfg.ablation_seeds if s_ in cfg.seeds) or (cfg.seeds[0],)   # [V2.1] như SW_S
if cfg.run_ablation:
    todo = {k: v for k, v in ABLATIONS.items() if k in cfg.ablation_variants and v[2] in cfg.methods}
    jobs = []
    for name, (m, hp, _) in todo.items():
        if m == "source_only":                       # source-only không dùng đích -> một job mỗi seed
            jobs += [dict(variant=f"abl_{name}", task=SW_T, seed=s_, fold=None, method=m, hp=hp) for s_ in AB_S]
        else:
            jobs += main_jobs(variant=f"abl_{name}", methods=[m], seeds=AB_S, tasks=[SW_T], hp=hp)
    run_jobs(jobs, "ablation")
    ab_rows = []
    for name, (m, hp, refm) in todo.items():
        S = method_summary(SW_T, m, f"abl_{name}", seeds=AB_S)
        R = method_summary(SW_T, refm, "main", seeds=AB_S)
        if S is None or R is None:
            continue
        common = S["g"].index.intersection(R["g"].index)
        d, lo, hi, p = paired_delta(S["g"].loc[common, "y"], S["g"].loc[common, "p"], R["g"].loc[common, "p"], seed=5)
        ab_rows.append(dict(variant=name, method=mlabel(m), reference=mlabel(refm), AUC=S["ens"]["auc"],
                            AUC_reference=R["ens"]["auc"], dAUC=d, dAUC_lo=lo, dAUC_hi=hi, p_boot=p, key=m))
    if ab_rows:
        ABL = pd.DataFrame(ab_rows); ABL["p_holm"] = holm(ABL.p_boot.values)
        save_table(ABL.drop(columns="key"), "ablation",
                   f"Ablation on {TASKS[SW_T]['name']} (seeds {list(AB_S)}): target AUC of each variant, its reference "
                   "configuration, and paired ΔAUC (variant − reference) with Holm correction.", 3)
        fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(ABL) + 1.4))
        for i, r in ABL.iterrows():
            ax.errorbar(r.dAUC, i, xerr=[[r.dAUC - r.dAUC_lo], [r.dAUC_hi - r.dAUC]], fmt="o", capsize=3,
                        color=METHOD_COLOR.get(r.key, "k"))
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(range(len(ABL))); ax.set_yticklabels([f"{v} (vs {rf})" for v, rf in zip(ABL.variant, ABL.reference)])
        ax.invert_yaxis(); ax.set_xlabel("ΔAUC (variant − reference), paired bootstrap")
        savefig(fig, "ablation", "Effect of removing individual components (negative = the component helps).")
else:
    print("Bỏ qua ablation (cfg.run_ablation=False).")
''')

# =============================================================================
# 14. BÁN GIÁM SÁT
# =============================================================================
md(r'''
## 14. (Tuỳ chọn) Cần bao nhiêu bản ghi đích có nhãn? / Semi-supervised curve (RQ5)

Bật `cfg.run_ssda=True`. Với *n* bản ghi đích **có nhãn mỗi lớp** (chọn ngẫu nhiên trong `tgt_train` của fold): so sánh
"nguồn + *n* nhãn đích" (không căn chỉnh) với "DANN + *n* nhãn đích". Điểm *n* = 0 lấy từ lượt chính (source-only, DANN),
"all" = oracle. Đường cong này cho biết chi phí gán nhãn hiện trường cần thiết để thay thế UDA.
''')

code(r'''
# =========================================================
# 14.1 Đường cong bán giám sát / semi-supervised learning curve
# =========================================================
if cfg.run_ssda and "dann" in cfg.methods:
    jobs = []
    for n in cfg.ssda_shots:
        jobs += main_jobs(variant="ssda", methods=["oracle", "dann"], seeds=SW_S, tasks=[SW_T], shots=int(n))
    run_jobs(jobs, "semi-supervised")
    xs = [0] + [int(n) for n in cfg.ssda_shots]
    curve = []
    for fam, m0 in [("Source + target labels (no adaptation)", "source_only"), ("DANN + target labels", "dann")]:
        for n in xs:
            S = (method_summary(SW_T, m0, "main", seeds=SW_S) if n == 0 else
                 method_summary(SW_T, "oracle" if m0 == "source_only" else "dann", "ssda", seeds=SW_S, shots=n))
            if S is not None:
                lo, hi = boot_auc_ci(S["g"].y, S["g"].p, seed=n)
                curve.append(dict(family=fam, shots_per_class=n, AUC=S["ens"]["auc"], lo=lo, hi=hi))
    CUR = pd.DataFrame(curve)
    save_table(CUR, "ssda_curve", f"Target AUC versus number of labelled target recordings per class "
                                  f"({TASKS[SW_T]['name']}, seeds {list(SW_S)}).", 3)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for (fam, d), c in zip(CUR.groupby("family", sort=False), ["#6E7B8B", METHOD_COLOR["dann"]]):
        ax.errorbar(d.shots_per_class, d.AUC, yerr=[d.AUC - d.lo, d.hi - d.AUC], fmt="o-", capsize=3, color=c, label=fam)
    orc = method_summary(SW_T, "oracle", "main", seeds=SW_S)
    if orc is not None:
        ax.axhline(orc["ens"]["auc"], color=METHOD_COLOR["oracle"], ls="-.", label="all target labels (oracle)")
    ax.set_xlabel("labelled target recordings per class"); ax.set_ylabel("target record-level AUC"); ax.legend(fontsize=8)
    savefig(fig, "ssda_curve", "Semi-supervised adaptation: target AUC as labelled target recordings are added.")
else:
    print("Bỏ qua đường cong bán giám sát (đặt cfg.run_ssda=True để chạy).")
''')

# =============================================================================
# 15. OCCLUSION THEO DẢI
# =============================================================================
md(r'''
## 15. Mô hình dựa vào dải tần nào trên miền đích? / Band-occlusion sensitivity

Che lần lượt từng nhóm dải Mel (đặt về mức trung bình sau chuẩn hoá) trên **frame kiểm thử đích** (fold 0) và đo mức giảm AUC
bản ghi. So sánh source-only với các mô hình đã thích ứng cho thấy thích ứng **dịch chuyển sự chú ý** sang dải nào — đối chiếu
với bảng AUC theo dải trong từng miền (mục 3.5).
''')

code(r'''
# =========================================================
# 15.1 Occlusion theo nhóm dải Mel / mel-band occlusion
# =========================================================
if cfg.run_band_occlusion and cfg.backbone == "cnn2d" and KEPT:
    G = 8
    edges = np.linspace(0, SPEC_SHAPE[1], G + 1).astype(int)
    mel_hz = librosa.mel_frequencies(n_mels=SPEC_SHAPE[1] + 2, fmin=F_MIN_EFF, fmax=F_MAX_EFF, htk=True)[1:-1]
    labels_ = [f"{mel_hz[a]:.0f}–{mel_hz[b - 1]:.0f}" for a, b in zip(edges[:-1], edges[1:])]
    occ_rows = []
    for m, net in KEPT.items():
        base, occ = [], [[] for _ in range(G)]
        with torch.no_grad():
            for s in range(0, len(tgt_te0), 256):
                S_ = net.spec(fetch(tgt_te0[s:s + 256], DEVICE))
                base.append(torch.sigmoid(net.head(net.trunk_from_spec(S_))[0].float()).cpu().numpy())
                for gi, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
                    S2 = S_.clone(); S2[:, :, a:b, :] = 0.0
                    occ[gi].append(torch.sigmoid(net.head(net.trunk_from_spec(S2))[0].float()).cpu().numpy())
        gb = rec_probs_from(tgt_te0, np.concatenate(base)); a0 = safe_auc(gb.y, gb.p)
        for gi in range(G):
            go = rec_probs_from(tgt_te0, np.concatenate(occ[gi]))
            occ_rows.append(dict(method=mlabel(m), band_hz=labels_[gi], auc_base=a0,
                                 auc_drop=a0 - safe_auc(go.y, go.p)))
    OCC = pd.DataFrame(occ_rows)
    save_table(OCC.pivot(index="band_hz", columns="method", values="auc_drop").reindex(labels_).reset_index(),
               "band_occlusion", f"Drop in target recording-level AUC when each mel-band group is occluded "
                                 f"({TASKS[FS_T]['name']}, fold-0 test recordings).", 3)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ms_ = OCC.method.unique(); w = .8 / len(ms_)
    for k, m in enumerate(ms_):
        d = OCC[OCC.method == m].set_index("band_hz").reindex(labels_)
        key = [kk for kk, vv in METHOD_LABEL.items() if vv == m]
        ax.bar(np.arange(G) + (k - (len(ms_) - 1) / 2) * w, d.auc_drop, w,
               color=METHOD_COLOR.get(key[0] if key else "", "k"), label=m)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(range(G)); ax.set_xticklabels(labels_, rotation=20); ax.set_xlabel("occluded band (Hz)")
    ax.set_ylabel("AUC drop"); ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    savefig(fig, "band_occlusion",
            "Band-occlusion sensitivity on target test recordings: larger drops mean the model relies more on that band.")
else:
    print("Bỏ qua occlusion (backbone không phải cnn2d hoặc không có mô hình đã giữ).")
''')

# =============================================================================
# 16. XUẤT TƯ LIỆU BÀI BÁO
# =============================================================================
md(r'''
## 16. Xuất tư liệu cho bài báo / Export artefacts for the manuscript

- `tables/`: mỗi bảng ở **CSV + LaTeX (booktabs) + Markdown**, đánh số theo thứ tự xuất hiện; `main_results_numeric.csv` giữ số thô.
- `figures/`: mỗi hình **PNG 300 dpi + PDF vector**; `figure_index.csv` chứa chú thích.
- `manuscript_draft.md`: đoạn **Methods/Results** tiếng Anh sinh tự động từ số liệu đo được (chỉ khẳng định "cải thiện có ý nghĩa"
  khi CI của ΔAUC không chứa 0 **và** p Holm < 0.05) + tóm tắt tiếng Việt.
- `ada_summary.json`, `checkpoints/`, `ada_paper_artifacts.zip`.
''')

code(r'''
# =========================================================
# 16.1 Bảng siêu tham số / hyper-parameter table
# =========================================================
hp_rows = [
    ("Data", "Domains", ", ".join(DOMAINS_ALL)),
    ("Data", "Independent recordings", f"{len(records)} ({int(records.label.sum())} leak)"),
    ("Data", "Frame / hop", f"{cfg.frame_sec:g} s / {cfg.hop_sec:g} s at {cfg.target_sr} Hz"),
    ("Pre-processing", "Filters", f"DC removal, {cfg.highpass_hz:g} Hz high-pass, "
                                  f"{'50 Hz notch ×' + str(cfg.mains_harmonics) if cfg.notch_mains else 'no notch'}, per-file peak norm"),
    ("Data", "Recording identity", "Rxxx / Sxx qualified by domain" + (" and class" if cfg.record_id_by_class else "")),
    ("Front-end", "log-Mel", f"{cfg.n_mels} bands, n_fft {cfg.n_fft}, hop {cfg.hop_length}, {F_MIN_EFF:.0f}–{F_MAX_EFF:.0f} Hz"
                             + (" (bandwidth-harmonised)" if HARMONISED else "")),
    ("Front-end", "SpecAugment", f"2 freq masks ≤{cfg.specaug_freq_mask}, 2 time masks ≤{cfg.specaug_time_mask}"),
    ("Network", "G_f widths / bottleneck", f"{list(cfg.widths)} / {cfg.bottleneck_dim}"),
    ("Network", "Discriminator", f"MLP {cfg.disc_hidden}-{cfg.disc_hidden}-1, dropout {cfg.disc_dropout}, "
                                 f"lr ×{cfg.disc_lr_mult:g}"),
    ("Adaptation", "RCST", f"pseudo-labels from epoch {int(math.ceil(cfg.rcst_start_frac * cfg.epochs))}, record selection "
                           f"{cfg.rcst_sel[0]:.0%}→{cfg.rcst_sel[1]:.0%} (rank-balanced), bag {cfg.rcst_bag[0]}×{cfg.rcst_bag[1]}, "
                           f"weights PL {cfg.rcst_pl_weight:g} / consistency {cfg.rcst_cons_weight:g}"),
    ("Adaptation", "MCC / TENT", f"MCC weight {cfg.mcc_weight:g}, T = {cfg.mcc_temp:g}; TENT {cfg.tent_steps} steps, "
                                 f"Adam lr {cfg.tent_lr:g} on BN affine parameters"),
    ("Adaptation", "λ schedule", f"λ_max(2/(1+exp(−{cfg.lambda_gamma:g}p))−1), λ_max = {cfg.lambda_max:g}"),
    ("Adaptation", "CORAL / MMD weight", f"{cfg.coral_weight:g} / {cfg.mmd_weight:g} (same ramp)"),
    ("Optimisation", "Optimiser", f"AdamW, lr {cfg.lr:g} (disc ×{cfg.disc_lr_mult:g}), wd {cfg.weight_decay:g}"),
    ("Optimisation", "Schedule", f"{cfg.epochs} epochs × {cfg.iters_per_epoch} iterations, warm-up {cfg.warmup_frac:.0%} + cosine"),
    ("Optimisation", "Batch", f"{cfg.batch_size} source (class-balanced={cfg.class_balanced_source}) + {cfg.batch_size} target"),
    ("Optimisation", "Regularisation", f"label smoothing {cfg.label_smoothing}, grad-clip {cfg.grad_clip}, AMP={cfg.amp}"),
    ("Augmentation", "Waveform", f"random FIR p={cfg.aug_fir_p}, noise SNR {cfg.aug_noise_snr_db} dB p={cfg.aug_noise_p}, "
                                 f"circular shift p={cfg.aug_shift_p}"),
    ("Augmentation", "Acquisition-chain randomisation", f"p={cfg.aug_chain_p}, high-pass {cfg.aug_hp_hz} Hz, tilt "
                                                        f"{cfg.aug_tilt_db_oct} dB/oct, low-pass {cfg.aug_lp_hz} Hz"),
    ("Protocol", "Target", f"{cfg.target_protocol}, {cfg.tgt_folds}-fold over target recordings"),
    ("Protocol", "Source validation", f"{cfg.src_val_frac:.0%} of source recordings"),
    ("Protocol", "Repetitions", f"seeds {list(cfg.seeds)}"),
    ("Protocol", "Checkpoint rule", f"{cfg.select_rule} (from epoch {SELECT_FROM_EPOCH})"),
    ("Statistics", "Record aggregation / threshold", f"{cfg.record_agg} / {cfg.threshold}"),
    ("Statistics", "CIs and tests", f"bootstrap over recordings (B = {cfg.bootstrap_n}), paired ΔAUC, Holm"),
]
save_table(pd.DataFrame(hp_rows, columns=["group", "parameter", "value"]), "hyperparameters",
           "Experimental settings and hyper-parameters (all fixed before looking at target-domain results).", 3)
''')

code(r'''
# =========================================================
# 16.2 Văn bản nháp Methods/Results + tóm tắt + đóng gói / draft text, summary JSON, archive
# =========================================================
def _row(task, m):
    d = MAIN[(MAIN.task == task) & (MAIN.method_key == m)]
    return None if d.empty else d.iloc[0]

def _sig(r):
    return (r is not None and np.isfinite(r.dAUC_lo) and (r.dAUC_lo > 0 or r.dAUC_hi < 0)
            and np.isfinite(r.p_holm) and r.p_holm < .05)

L = ["# Draft text (auto-generated from measured results — verify before use)\n", "## Methods (draft)\n"]
L.append(
    f"Recordings from {len(DOMAINS_ALL)} acquisition domains ({', '.join(DOMAINS_ALL)}) of the tapdulieuMix_16k corpus were "
    f"used ({len(records)} independent recordings, {records.duration_s.sum() / 60:.1f} min). Files belonging to the same "
    f"acquisition were grouped into one recording (identifier qualified by domain"
    + (" and class" if cfg.record_id_by_class else "") + "), which is the unit of every split and statistic. "
    f"Signals were resampled to {cfg.target_sr} Hz, high-pass filtered at {cfg.highpass_hz:g} Hz, notch-filtered at 50 Hz "
    f"and harmonics, peak-normalised per file and split into {cfg.frame_sec:g}-s frames with {cfg.hop_sec:g}-s hop. "
    + (f"Because the domain-average spectra (computed without labels) differed by more than {cfg.harmonise_tol_db:g} dB "
       f"outside {F_MIN_EFF:.0f}–{F_MAX_EFF:.0f} Hz, all spectral front-ends were restricted to this common band. "
       if HARMONISED else "")
    + f"A log-Mel CNN feature extractor with a {cfg.bottleneck_dim}-d bottleneck was shared by all methods, and training "
      f"used acquisition-chain randomisation (random high-pass, spectral tilt and low-pass) in addition to random-FIR, "
      f"noise and shift augmentation. We compared source-only training; normalisation-based adaptation (joint "
      f"source+target BatchNorm without any alignment loss, AdaBN, TENT); Deep CORAL; MCC; DANN and CDAN+E with a "
      f"{cfg.disc_hidden}-unit discriminator trained at {cfg.disc_lr_mult:g}× the learning rate (λ_p ramped to "
      f"λ_max = {cfg.lambda_max:g}); and the proposed record-consistent self-training (RCST), which adds rank-balanced "
      f"recording-level pseudo-labels and a within-recording consistency penalty to DANN. Supervised source+target and "
      f"target-only models served as upper references. Each transfer task was evaluated with an inductive "
      f"{cfg.tgt_folds}-fold split of the target recordings: unlabelled target data used for adaptation never included "
      f"the recordings being tested. Training used AdamW ({cfg.epochs}×{cfg.iters_per_epoch} iterations, batch "
      f"{cfg.batch_size}+{cfg.batch_size}); the checkpoint was chosen by the pre-registered '{cfg.select_rule}' rule "
      f"without target labels. Experiments were repeated with {len(cfg.seeds)} seeds (new partitions and initialisations) "
      f"and out-of-fold probabilities were averaged across seeds. The primary metric is the recording-level AUC (mean "
      f"frame probability); 95% CIs were obtained by bootstrap over target recordings ({cfg.bootstrap_n} resamples) and "
      f"methods were compared with source-only by paired bootstrap ΔAUC with Holm correction.\n")
L.append("\n## Results (draft)\n")
VN = []
for T in TASKS:
    so, dn, ce, rc, orc = (_row(T["name"], m) for m in ("source_only", "dann", "cdan_e", "rcst", "oracle"))
    s = f"**{T['name']}.** "
    if so is not None:
        s += f"Source-only reached a target AUC of {_ci(so.AUC, so.AUC_lo, so.AUC_hi)}. "
    for r in (dn, ce, rc):
        if r is None:
            continue
        verdict = ("significantly improved" if _sig(r) and r.dAUC > 0 else
                   "significantly reduced" if _sig(r) else "did not significantly change")
        s += (f"{r.method} {verdict} the target AUC ({r.AUC:.3f}; ΔAUC = {_ci(r.dAUC, r.dAUC_lo, r.dAUC_hi)}, "
              f"Holm p {_pf(r.p_holm)}" + (f", {r.gap_closed_pct:.0f}% of the source–oracle gap closed"
                                             if np.isfinite(r.gap_closed_pct) else "") + "). ")
        VN.append(f"- {T['name']}: {r.method} AUC {r.AUC:.3f} (ΔAUC {r.dAUC:+.3f} [{r.dAUC_lo:+.3f}, {r.dAUC_hi:+.3f}], "
                  f"p Holm {_pf(r.p_holm)}) → " + ("CẢI THIỆN có ý nghĩa" if _sig(r) and r.dAUC > 0 else
                                                  "GIẢM có ý nghĩa" if _sig(r) else "KHÔNG khác biệt có ý nghĩa"))
    if orc is not None:
        s += f"The supervised oracle reached {_ci(orc.AUC, orc.AUC_lo, orc.AUC_hi)}."
    L.append(s + "\n")
if "MECH" in globals() and len(MECH):
    for T in TASKS:
        d = MECH[MECH.task == T["name"]].set_index("contrast")
        parts = [f"{c}: ΔAUC {r.dAUC:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}], Holm p {_pf(r.p_holm)}" for c, r in d.iterrows()
                 if c.startswith(("Joint-BN", "DANN − Joint", "RCST − DANN"))]
        if parts:
            L.append(f"\nMechanism ({T['name']}): " + "; ".join(parts) + ".\n")
if "FS_TAB" in globals() and len(FS_TAB):
    pads = {r.representation: r.PAD for r in FS_TAB.itertuples()}
    L.append(f"\nOn {TASKS[FS_T]['name']}, the proxy A-distance between source and target features was "
             + ", ".join(f"{k}: {v:.2f}" for k, v in pads.items()) + ".\n")
if "SEL" in globals() and len(SEL):
    o = SEL.optimism.dropna()
    if len(o):
        L.append(f"\nSelecting checkpoints with target labels would have inflated target AUC by {o.mean():+.3f} on average "
                 f"(max {o.max():+.3f}) relative to the label-free '{cfg.select_rule}' rule.\n")
L.append("\n## Tóm tắt tiếng Việt\n" + "\n".join(VN) + "\n")
(OUT / "manuscript_draft.md").write_text("\n".join(L), encoding="utf-8")
display(Markdown("\n".join(L)))

pd.DataFrame([dict(figure=k, caption=v) for k, v in FIG_INDEX.items()]).to_csv(OUT / "figure_index.csv", index=False)
pd.DataFrame([dict(table=k, caption=v) for k, v in TAB_INDEX.items()]).to_csv(OUT / "table_index.csv", index=False)
summary = dict(version="ada-v1", run_fingerprint=RUN_FP, synthetic=bool(SYNTHETIC), domains=DOMAINS_ALL,
               tasks=[T["name"] for T in TASKS], f_max_eff=F_MAX_EFF, harmonised=bool(HARMONISED),
               n_records=int(len(records)), n_frames=int(N_FRAMES), config=_cfg_print,
               main_results=MAIN.drop(columns=[], errors="ignore").to_dict(orient="records"),
               runtime_min=(time.time() - T_START) / 60,
               software=dict(python=sys.version.split()[0], torch=torch.__version__,
                             torchaudio=torchaudio.__version__, numpy=np.__version__),
               gpus=[torch.cuda.get_device_properties(i).name for i in range(N_GPU)])
json.dump(summary, open(OUT / "ada_summary.json", "w"), indent=1, ensure_ascii=False, default=float)

with zipfile.ZipFile(OUT / "ada_paper_artifacts.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in list(FIG_DIR.glob("*")) + list(TAB_DIR.glob("*")):
        z.write(p, p.relative_to(OUT))
    for name in ("manuscript_draft.md", "figure_index.csv", "table_index.csv", "ada_summary.json",
                 "manifest_files.csv", "manifest_records.csv"):
        if (OUT / name).exists():
            z.write(OUT / name, name)
print(f"Đã xuất {len(FIG_INDEX)} hình, {len(TAB_INDEX)} bảng -> {OUT / 'ada_paper_artifacts.zip'}")
print(f"Tổng thời gian chạy: {(time.time() - T_START) / 60:.1f} phút")
if SYNTHETIC:
    print("!! ĐANG Ở SYNTHETIC DEMO MODE — mọi số liệu trên là của dữ liệu mô phỏng, KHÔNG dùng cho bài báo.")
''')

# =============================================================================
# 17. KẾT LUẬN
# =============================================================================
md(r'''
## 17. Diễn giải, hạn chế và danh mục kiểm tra trước khi viết bài / Interpretation, limitations, checklist

**Đọc kết quả theo thứ tự:**

1. **Mục 3 có cho thấy dịch chuyển miền thật không?** PAD đầu vào cao và AUC chéo miền (Bảng ma trận A→B) thấp hơn
   rõ rệt so với đường chéo → có động cơ cho thích ứng. Nếu AUC chéo miền ≈ trong miền, UDA không có gì để cải thiện.
2. **Giả thiết UDA có đứng không?** Bảng AUC theo dải trong từng miền: dải "FLIP" có ý nghĩa ⇒ dịch chuyển có điều kiện;
   tỉ lệ lớp khác nhau nhiều ⇒ dịch chuyển nhãn. Cả hai làm DANN (căn chỉnh biên) kém tin cậy; CDAN+E và PAD theo lớp
   (mục 11) cho biết căn chỉnh có điều kiện có xảy ra không.
3. **Bảng kết quả chính**: chỉ viết "DANN cải thiện" khi CI của ΔAUC không chứa 0 **và** p Holm < 0.05; báo cáo cả
   *gap closed* và AUC nguồn-val (DANN không được phá hiệu năng nguồn).
4. **Mục 10–11**: độ chính xác bộ phân biệt → ~0.5, PAD giảm, silhouette lớp đích không giảm = thích ứng đúng nghĩa.
   PAD giảm nhưng silhouette giảm = **căn chỉnh sai lớp (negative transfer)**.
5. **Mục 12**: báo cáo kết quả theo quy tắc **không dùng nhãn đích**; nêu mức lạc quan của `oracle` như một phát hiện về phương pháp luận.

**Hạn chế phải nêu trong bài báo:**

- Số **bản ghi** đích nhỏ ⇒ CI rộng; mọi kết luận ở mức bản ghi, không phải frame.
- "Miền" ở đây là **nhãn thao tác** (thư mục) — có thể gộp nhiều yếu tố (cảm biến, điểm khảo sát, áp suất, chuỗi thu).
  Nếu có metadata chi tiết, nên tách miền theo từng yếu tố (`cfg.domain_source="csv"`).
- Quy nạp K-fold dùng ít dữ liệu đích không nhãn hơn kịch bản *transductive* thường gặp trong tài liệu UDA → kết quả bảo thủ hơn.
- Siêu tham số (λ_max, lịch, kiến trúc) cố định trước; quét λ (mục 13) chỉ là phân tích độ nhạy, không dùng để chọn.

| Kiểm tra trước khi nộp | Mục |
|---|---|
| Không còn cặp trùng nội dung giữa nguồn và đích | 3.6 |
| Băng thông các miền đã hoà hợp (hoặc nêu lý do không cần) | 3.2 |
| 5 assertion phân hoạch đạt cho mọi seed | 6.2 |
| Mọi bản ghi đích có đúng một dự đoán OOF mỗi seed | 9.2 (dòng "phủ") |
| Kết quả chính dùng quy tắc chọn không nhãn đích | 9.2, 12 |
| SYNTHETIC = False | 16.2 |
''')
