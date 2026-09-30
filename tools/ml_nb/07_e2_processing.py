# %% [markdown]
# ## 7. E2 — Xử lý tín hiệu có giúp nhận dạng tốt hơn tín hiệu thô? / Processed vs raw (RQ2)
#
# **Thiết kế có kiểm soát**: chỉ thay đổi điều kiện tín hiệu (C0…C6); mọi thứ khác giữ nguyên — cùng
# danh mục đặc trưng, cùng các fold mức bản ghi (ghép cặp), cùng 4 mô hình với **siêu tham số mặc định
# cố định**, hai chế độ đặc trưng: `all` (toàn bộ) và `mrmr30` (30 đặc trưng mRMR chọn lại **trong từng
# fold**). Chỉ dùng tập **DEV**.
#
# Thống kê: AUC mức bản ghi (trung bình qua các lần lặp CV và AUC của điểm OOF trung bình kèm CI
# bootstrap), so sánh **ghép cặp** với `C0_RAW` bằng DeLong (p hiệu chỉnh Holm theo từng mô hình),
# CI bootstrap của ΔAUC và t-test hiệu chỉnh Nadeau–Bengio trên AUC từng fold.
#
# **Quy tắc chọn điều kiện tốt nhất (đăng ký trước)**: AUC mức bản ghi trung bình qua 4 mô hình ở chế
# độ `cfg.e2_primary`. Điều kiện này được dùng cho E3–E4.

# %%
# =========================================================
# 7.1 Chạy E2: 7 điều kiện x 2 chế độ đặc trưng x 4 mô hình x (lặp x fold)
# =========================================================
def parse_regime(rg):
    if rg == "all":
        return None, 0
    m = re.fullmatch(r"([a-z]+)(\d+)", rg)
    return m.group(1), int(m.group(2))


E2_RUNS, E2_ROWS = {}, []
for cond in cfg.conditions:
    ds = DS(cond, "dev")
    combos, rks = [], set()
    for rg in cfg.e2_regimes:
        rk, k = parse_regime(rg)
        if rk:
            rks.add(rk)
        combos += [(rk, k, m) for m in cfg.models]
    run = run_cv(ds, combos, rankers=tuple(sorted(rks)), tag="E2")
    E2_RUNS[cond] = run
    for rg in cfg.e2_regimes:
        rk, k = parse_regime(rg)
        for m in cfg.models:
            s = summarize(run, (rk, k, m))
            E2_ROWS.append(dict(condition=cond, regime=rg, model=m, n_frames=len(ds.y),
                                **{a: b for a, b in s.items() if not a.startswith("_")}, _s=s))
E2_TAB = pd.DataFrame(E2_ROWS)
_show = ["condition", "regime", "model", "n_frames", "rec_auc_rep_mean", "rec_auc_rep_sd", "rec_auc", "rec_auc_lo",
         "rec_auc_hi", "frame_auc", "rec_bacc", "rec_sens", "rec_spec", "rec_f1", "rec_mcc"]
savetab(E2_TAB[_show].round(4), "T06_E2_processing_results")
PIV = {rg: E2_TAB[E2_TAB.regime == rg].pivot(index="model", columns="condition", values="rec_auc_rep_mean")
       .reindex(index=list(cfg.models), columns=list(cfg.conditions)) for rg in cfg.e2_regimes}
for rg in cfg.e2_regimes:
    display(Markdown(f"**Bảng 6{'ab'[list(cfg.e2_regimes).index(rg)]}. AUC mức bản ghi (trung bình {len(cfg.seeds)} lần lặp CV) — chế độ `{rg}`.**"))
    _t = PIV[rg].copy(); _t.loc["mean over models"] = _t.mean(0)
    display(_t.round(4))

MEAN_BY_COND = PIV[cfg.e2_primary].mean(0).sort_values(ascending=False)
BEST_COND = MEAN_BY_COND.index[0]
print("\nAUC trung bình 4 mô hình theo điều kiện (chế độ chính):")
print(MEAN_BY_COND.round(4).to_string())
print(f"=> Điều kiện tốt nhất (quy tắc đăng ký trước): {BEST_COND}")

# %%
# =========================================================
# 7.2 So sánh ghép cặp: mỗi điều kiện vs C0_RAW, và từng bước so với bước liền trước
# =========================================================
STEP_PREV = {"C1_BPF": "C0_RAW", "C2_TR": "C1_BPF", "C3_IR": "C2_TR", "C4_MBG": "C3_IR", "C5_SS": "C3_IR", "C6_WT": "C3_IR"}
_get = {(c, rg, m): s for c, rg, m, s in zip(E2_TAB.condition, E2_TAB.regime, E2_TAB.model, E2_TAB["_s"])}
rec_y_dev = DS(cfg.conditions[0], "dev").rec_y
CMP = []
for rg in cfg.e2_regimes:
    for m in cfg.models:
        for kind, pairs in (("vs_raw", [(c, "C0_RAW") for c in cfg.conditions if c != "C0_RAW"]),
                            ("stepwise", [(c, STEP_PREV[c]) for c in cfg.conditions if c in STEP_PREV and STEP_PREV[c] in cfg.conditions])):
            rows = [dict(regime=rg, model=m, comparison=kind, A=a, B=b,
                         **compare(_get[(a, rg, m)], _get[(b, rg, m)], rec_y_dev)) for a, b in pairs]
            ph = holm([r["p_delong"] for r in rows])
            for r, p in zip(rows, ph):
                r["p_delong_holm"] = p
            CMP += rows
E2_CMP = pd.DataFrame(CMP)
savetab(E2_CMP.round(4), "T07_E2_paired_tests")
_v = E2_CMP[(E2_CMP.comparison == "vs_raw") & (E2_CMP.regime == cfg.e2_primary)]
display(Markdown(f"**Bảng 7. ΔAUC mức bản ghi so với C0_RAW (chế độ `{cfg.e2_primary}`), DeLong + Holm.**"))
display(_v.pivot(index="A", columns="model", values="d_auc").round(4))
display(_v.pivot(index="A", columns="model", values="p_delong_holm").round(4))

# %%
# =========================================================
# 7.3 Hình 7-8: bản đồ nhiệt AUC và biểu đồ rừng ΔAUC so với tín hiệu thô
# =========================================================
fig, ax = plt.subplots(1, len(cfg.e2_regimes), figsize=(6.2 * len(cfg.e2_regimes), 2.9), squeeze=False)
vmin = np.nanmin([PIV[r].values.min() for r in cfg.e2_regimes]); vmax = np.nanmax([PIV[r].values.max() for r in cfg.e2_regimes])
for j, rg in enumerate(cfg.e2_regimes):
    M = PIV[rg].values; a = ax[0, j]
    im = a.imshow(M, cmap=SEQ, vmin=vmin, vmax=vmax, aspect="auto")
    for (i, k), v in np.ndenumerate(M):
        a.text(k, i, f"{v:.3f}", ha="center", va="center", fontsize=7.5,
               color="white" if (v - vmin) / max(vmax - vmin, 1e-9) > 0.6 else INK)
    a.set_xticks(range(M.shape[1])); a.set_xticklabels([c.replace("_", "\n") for c in PIV[rg].columns], fontsize=7.5)
    a.set_yticks(range(M.shape[0])); a.set_yticklabels(PIV[rg].index); a.grid(False)
    a.set_title(f"({'ab'[j]}) record-level AUC, features: {rg}")
fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.9, label="AUC")
savefig(fig, "Fig07_E2_heatmap", "Record-level AUC per processing condition and model.")

fig, ax = plt.subplots(figsize=(8.5, 3.6))
conds = [c for c in cfg.conditions if c != "C0_RAW"]
for i, m in enumerate(cfg.models):
    d = _v[_v.model == m].set_index("A").reindex(conds)
    y = np.arange(len(conds)) + (i - (len(cfg.models) - 1) / 2) * 0.17
    ax.errorbar(d.d_auc, y, xerr=[d.d_auc - d.d_lo, d.d_hi - d.d_auc], fmt="o", ms=4, color=MODEL_COLOR[m],
                ecolor=MODEL_COLOR[m], elinewidth=1.2, capsize=0, label=m)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(range(len(conds))); ax.set_yticklabels(conds); ax.invert_yaxis()
ax.set_xlabel("ΔAUC vs C0_RAW (record level, 95% paired bootstrap CI)")
ax.set_title(f"Effect of each processing condition relative to raw audio ({cfg.e2_primary})")
ax.legend(ncol=len(cfg.models), loc="lower right")
fig.tight_layout()
savefig(fig, "Fig08_E2_delta_vs_raw", "Paired ΔAUC of each processing condition relative to raw audio.")

# %%
# =========================================================
# 7.4 Kết luận RQ2 (sinh tự động)
# =========================================================
_raw = PIV[cfg.e2_primary]["C0_RAW"]; _best = PIV[cfg.e2_primary][BEST_COND]
_sig_better = _v[(_v.A == BEST_COND) & (_v.p_delong_holm < 0.05) & (_v.d_auc > 0)].model.tolist()
_sig_worse = _v[(_v.p_delong_holm < 0.05) & (_v.d_auc < 0)][["A", "model"]].values.tolist()
E2_SUMMARY = dict(best=BEST_COND, raw_mean=float(_raw.mean()), best_mean=float(_best.mean()),
                  delta_mean=float(_best.mean() - _raw.mean()), sig_better=_sig_better, sig_worse=_sig_worse,
                  ranking=MEAN_BY_COND.round(4).to_dict())
lines = [f"**RQ2 — Xử lý vs thô** (DEV, {len(rec_y_dev)} bản ghi, chế độ `{cfg.e2_primary}`):",
         f"- AUC trung bình 4 mô hình: C0_RAW = {_raw.mean():.3f}; tốt nhất {BEST_COND} = {_best.mean():.3f} "
         f"(Δ = {_best.mean() - _raw.mean():+.3f}).",
         "- Xếp hạng điều kiện: " + " > ".join(f"{c} ({v:.3f})" for c, v in MEAN_BY_COND.items()) + ".",
         f"- {BEST_COND} tốt hơn C0_RAW **có ý nghĩa** (DeLong, Holm < 0,05) ở: {', '.join(_sig_better) if _sig_better else 'không mô hình nào'}.",
         f"- Các cặp (điều kiện, mô hình) **kém hơn có ý nghĩa** so với thô: {_sig_worse if _sig_worse else 'không có'}."]
for c in ("C5_SS", "C6_WT"):
    if c in PIV[cfg.e2_primary]:
        lines.append(f"- {c}: AUC trung bình {PIV[cfg.e2_primary][c].mean():.3f} "
                     f"({PIV[cfg.e2_primary][c].mean() - _raw.mean():+.3f} so với thô) — kiểm chứng giả thuyết "
                     "\"khử nhiễu dừng xoá mất tiếng rò rỉ\".")
display(Markdown("\n".join(lines)))
