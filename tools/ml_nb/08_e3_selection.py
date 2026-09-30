# %% [markdown]
# ## 8. E3 — Tuyển chọn đặc trưng: ít nhất mà không giảm hiệu năng / Feature selection (RQ3)
#
# Trên điều kiện tốt nhất của E2 (`BEST_COND`), tập DEV:
#
# 1. **E3a – Liên quan đơn biến** (mức bản ghi): AUC, Cliff's δ, Mann–Whitney + FDR cho từng đặc trưng.
# 2. **E3b – Dư thừa**: phân cụm phân cấp theo |ρ Spearman|; đếm số cụm ở |ρ| ≥ 0,9.
# 3. **E3c – Xếp hạng đồng thuận TRONG TỪNG FOLD** từ 5 phương pháp thuộc 3 họ — lọc (ANOVA-F,
#    thông tin tương hỗ), lọc đa biến (mRMR-FCQ), nhúng (Extra-Trees, hồi quy logistic L1) — gộp Borda.
#    Việc chọn **lặp lại trong mỗi fold** là bắt buộc: chọn trên toàn bộ dữ liệu rồi mới CV sẽ cho kết quả
#    lạc quan giả tạo (Ambroise & McLachlan 2002).
# 4. **Đường cong AUC theo số đặc trưng k** cho 4 mô hình →
#    `K_MIN` = k nhỏ nhất có AUC trung bình ≥ AUC tốt nhất − `cfg.k_tol` (**câu trả lời RQ3**);
#    `K_MAIN` = k tốt nhất trong khoảng `cfg.k_main_range` = [20, 50] (bộ làm việc chính).
# 5. **E3d – So sánh phương pháp chọn** ở k ∈ `cfg.method_cmp_k`; **độ ổn định** Nogueira et al. (2018).
# 6. **Danh sách cuối**: xếp hạng đồng thuận trung bình qua mọi fold → top-`K_MAIN` và top-`K_MIN`.

# %%
# =========================================================
# 8.1 E3a + E3b: liên quan đơn biến và cấu trúc dư thừa
# =========================================================
DS3 = DS(BEST_COND, "dev")
_rm = pd.DataFrame(DS3.X, columns=FEATURES).groupby(DS3.g).median().reindex(DS3.recs)
_p, _n = DS3.rec_y == 1, DS3.rec_y == 0
UNI = []
for n in FEATURES:
    v = _rm[n].values
    UNI.append(dict(feature=n, group=FEAT_META.loc[n, "group"], auc_rec=auc_safe(DS3.rec_y, v),
                    auc_frame=auc_safe(DS3.y, DS3.X[:, FEATURES.index(n)]), cliffs_delta=cliffs_delta(v[_p], v[_n]),
                    p=mwu_p(v[_p], v[_n]), leak_median=np.nanmedian(v[_p]), noleak_median=np.nanmedian(v[_n])))
UNI = pd.DataFrame(UNI); UNI["q_bh"] = bh_fdr(UNI.p.fillna(1).values)
UNI["sep"] = (UNI.auc_rec - 0.5).abs() * 2
UNI = UNI.sort_values("sep", ascending=False).reset_index(drop=True)
savetab(UNI.round(5), "T08_univariate_relevance")
display(Markdown(f"**Bảng 8. 20 đặc trưng liên quan đơn biến mạnh nhất (mức bản ghi, {BEST_COND}).** "
                 f"Có {int((UNI.q_bh < 0.05).sum())}/{len(UNI)} đặc trưng còn ý nghĩa sau FDR."))
display(UNI.head(20)[["feature", "group", "auc_rec", "auc_frame", "cliffs_delta", "p", "q_bh"]].round(4))

_Xs = pd.DataFrame(DS3.X, columns=FEATURES)
RHO = _Xs.rank().corr().abs().fillna(0).to_numpy(copy=True)
np.fill_diagonal(RHO, 1.0)
_Z = sch.linkage(squareform(1 - RHO, checks=False), method="average")
N_CLUST90 = len(np.unique(sch.fcluster(_Z, t=0.1, criterion="distance")))
print(f"Dư thừa: {len(FEATURES)} đặc trưng tạo thành {N_CLUST90} cụm ở |ρ| ≥ 0,9 "
      f"-> số chiều 'hiệu dụng' nhỏ hơn nhiều so với số đặc trưng.")

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
top = UNI.head(25).iloc[::-1]
ax[0].barh(range(len(top)), top.auc_rec - 0.5, left=0.5, color=[GROUP_COLOR[g] for g in top.group], height=0.7)
ax[0].axvline(0.5, color=INK2, lw=0.8)
ax[0].set_yticks(range(len(top))); ax[0].set_yticklabels(top.feature, fontsize=7)
ax[0].set_xlabel("record-level AUC (>0.5: higher in leak)"); ax[0].set_title("(a) Top-25 univariate features")
for g in GROUPS:
    if g in set(top.group):
        ax[0].barh([np.nan], [0], color=GROUP_COLOR[g], label=g)
ax[0].legend(fontsize=7, loc="lower right")
t40 = UNI.head(40).feature.tolist(); ii = [FEATURES.index(n) for n in t40]
sub = RHO[np.ix_(ii, ii)]; o = sch.leaves_list(sch.linkage(squareform(1 - sub, checks=False), "average"))
im = ax[1].imshow(sub[np.ix_(o, o)], cmap=SEQ, vmin=0, vmax=1); ax[1].grid(False)
ax[1].set_xticks(range(len(o))); ax[1].set_xticklabels([t40[k] for k in o], rotation=90, fontsize=5.5)
ax[1].set_yticks(range(len(o))); ax[1].set_yticklabels([t40[k] for k in o], fontsize=5.5)
ax[1].set_title("(b) |Spearman ρ| among the top-40 features (clustered)")
fig.colorbar(im, ax=ax[1], shrink=0.8)
fig.tight_layout()
savefig(fig, "Fig09_univariate_redundancy", "Univariate relevance and redundancy structure.")

# %%
# =========================================================
# 8.2 E3c + E3d: xếp hạng TRONG fold, đường cong AUC(k), so sánh phương pháp
# =========================================================
_k = [k for k in cfg.k_grid if k == 0 or k < len(FEATURES)]
combos = [("consensus", k, m) for k in _k for m in cfg.models]
combos += [(r, k, m) for r in list(cfg.rankers) for k in cfg.method_cmp_k for m in cfg.method_cmp_models]
combos = list(dict.fromkeys(combos))
E3_RUN = run_cv(DS3, combos, rankers=tuple(cfg.rankers) + ("consensus",), tag="E3")

rows = []
for (rk, k, m) in combos:
    s = summarize(E3_RUN, (rk, k, m), boot=False)
    rows.append(dict(ranker=rk, k=(k if k else len(FEATURES)), model=m,
                     **{a: b for a, b in s.items() if not a.startswith("_")}))
E3_TAB = pd.DataFrame(rows)
CURVE = E3_TAB[E3_TAB.ranker == "consensus"].pivot(index="k", columns="model", values="rec_auc_rep_mean")[list(cfg.models)]
CURVE_F = E3_TAB[E3_TAB.ranker == "consensus"].pivot(index="k", columns="model", values="frame_auc")[list(cfg.models)]
CURVE["mean"] = CURVE.mean(1)
best_auc = CURVE["mean"].max()
K_MIN = int(CURVE.index[CURVE["mean"] >= best_auc - cfg.k_tol].min())
_rng = CURVE.loc[(CURVE.index >= cfg.k_main_range[0]) & (CURVE.index <= cfg.k_main_range[1])]
K_MAIN = int(_rng["mean"].idxmax()) if len(_rng) else int(CURVE["mean"].idxmax())
K_MIN_MODEL = {m: int(CURVE.index[CURVE[m] >= CURVE[m].max() - cfg.k_tol].min()) for m in cfg.models}
display(Markdown(f"**Bảng 9. AUC mức bản ghi theo số đặc trưng k (xếp hạng đồng thuận chọn trong fold, {BEST_COND}).**"))
display(CURVE.round(4))
savetab(E3_TAB.round(4), "T09_E3_k_curve_and_methods")
print(f"AUC trung bình tốt nhất = {best_auc:.4f} tại k = {int(CURVE['mean'].idxmax())}")
print(f"K_MIN  (ít nhất, giảm <= {cfg.k_tol}) = {K_MIN}  | theo từng mô hình: {K_MIN_MODEL}")
print(f"K_MAIN (tốt nhất trong {cfg.k_main_range}) = {K_MAIN}")

MCMP = E3_TAB[(E3_TAB.k.isin(cfg.method_cmp_k)) & (E3_TAB.model.isin(cfg.method_cmp_models))] \
    .pivot_table(index=["ranker"], columns=["model", "k"], values="rec_auc_rep_mean")
display(Markdown("**Bảng 10. So sánh phương pháp chọn đặc trưng (AUC mức bản ghi).**")); display(MCMP.round(4))
savetab(MCMP.round(4).reset_index(), "T10_E3_selection_methods")


def nogueira_stability(orders, k, p):
    Z = np.zeros((len(orders), p))
    for i, o in enumerate(orders):
        Z[i, o[:k]] = 1
    M = len(orders); pf = Z.mean(0); s2 = M / (M - 1) * pf * (1 - pf)
    kb = Z.sum(1).mean()
    return float(1 - s2.mean() / ((kb / p) * (1 - kb / p)))


STAB = pd.DataFrame([dict(ranker=r, k=k, stability=nogueira_stability(E3_RUN.orders[r], k, len(FEATURES)))
                     for r in list(cfg.rankers) + ["consensus"] for k in sorted(set(cfg.method_cmp_k) | {K_MIN, K_MAIN})])
display(Markdown("**Bảng 11. Độ ổn định lựa chọn qua các fold (Nogueira; 1 = luôn chọn cùng một tập).**"))
display(STAB.pivot(index="ranker", columns="k", values="stability").round(3))
savetab(STAB.round(4), "T11_selection_stability")

# %%
# =========================================================
# 8.3 Danh sách đặc trưng cuối cùng (xếp hạng đồng thuận trung bình qua mọi fold DEV)
# =========================================================
_pos = np.zeros((len(E3_RUN.orders["consensus"]), len(FEATURES)))
for i, o in enumerate(E3_RUN.orders["consensus"]):
    _pos[i, o] = np.arange(len(FEATURES))
FINAL_RANK = np.argsort(_pos.mean(0), kind="stable")
FEAT_MAIN = [FEATURES[i] for i in FINAL_RANK[:K_MAIN]]
FEAT_MIN = [FEATURES[i] for i in FINAL_RANK[:K_MIN]]
sel_freq = lambda k: (_pos < k).mean(0)
SEL = pd.DataFrame(dict(rank=np.arange(1, K_MAIN + 1), feature=FEAT_MAIN))
SEL = SEL.merge(FEAT_META.reset_index()[["feature", "group", "family", "description", "level_dep", "band_lo", "band_hi"]], on="feature")
SEL["sel_freq_topKmain"] = [sel_freq(K_MAIN)[FEATURES.index(n)] for n in SEL.feature]
SEL["mean_rank"] = [_pos.mean(0)[FEATURES.index(n)] + 1 for n in SEL.feature]
SEL = SEL.merge(UNI[["feature", "auc_rec", "cliffs_delta", "q_bh"]], on="feature")
SEL["direction"] = np.where(SEL.auc_rec >= 0.5, "higher in leak", "higher in noleak")
SEL["in_K_MIN"] = SEL["rank"] <= K_MIN
display(Markdown(f"**Bảng 12. {K_MAIN} đặc trưng được chọn (K_MAIN); {K_MIN} dòng đầu = bộ tối thiểu K_MIN.**"))
display(SEL.round(3))
savetab(SEL.round(4), "T12_selected_features")
(OUT / "selected_features.json").write_text(json.dumps(
    {"condition": BEST_COND, "K_MAIN": FEAT_MAIN, "K_MIN": FEAT_MIN}, indent=1, ensure_ascii=False))

fig = plt.figure(figsize=(12.5, 4.2)); gs = gridspec.GridSpec(1, 3, width_ratios=[1.35, 1, 0.8], wspace=0.35)
a = fig.add_subplot(gs[0])
for m in cfg.models:
    a.plot(CURVE.index, CURVE[m], "-o", ms=3, color=MODEL_COLOR[m], lw=1.3, label=m)
a.plot(CURVE.index, CURVE["mean"], "-", color=INK, lw=2.0, label="mean of 4 models")
a.axhline(best_auc - cfg.k_tol, color=INK2, lw=0.8)
a.axvline(K_MIN, color=INK2, lw=0.8); a.axvline(K_MAIN, color=CAT[1], lw=0.8)
a.text(K_MIN, a.get_ylim()[0], f" K_MIN={K_MIN}", va="bottom", fontsize=8, color=INK2)
a.text(K_MAIN, a.get_ylim()[0] + 0.02, f" K_MAIN={K_MAIN}", va="bottom", fontsize=8, color=INK2)
a.set_xscale("log"); a.set_xlabel("number of selected features k (in-fold consensus ranking)")
a.set_ylabel("record-level AUC (DEV CV)"); a.set_title("(a) Performance vs number of features"); a.legend(fontsize=7)
b = fig.add_subplot(gs[1])
_sfr = SEL.iloc[::-1]
b.barh(range(len(_sfr)), _sfr.sel_freq_topKmain, color=[GROUP_COLOR[g] for g in _sfr.group], height=0.7)
b.set_yticks(range(len(_sfr))); b.set_yticklabels(_sfr.feature, fontsize=5.5 if K_MAIN > 30 else 7)
b.set_xlabel(f"selection frequency in top-{K_MAIN} across folds"); b.set_title("(b) Selected features")
c = fig.add_subplot(gs[2])
comp = pd.DataFrame({"K_MIN": SEL[SEL.in_K_MIN].group.value_counts(), "K_MAIN": SEL.group.value_counts()}).reindex(GROUPS).fillna(0)
left = np.zeros(2)
for g in GROUPS:
    v = comp.loc[g, ["K_MIN", "K_MAIN"]].values
    if v.sum():
        c.barh([0, 1], v, left=left, color=GROUP_COLOR[g], edgecolor="white", linewidth=1.5, label=g, height=0.6)
    left += v
c.set_yticks([0, 1]); c.set_yticklabels(["K_MIN", "K_MAIN"]); c.set_xlabel("features"); c.grid(False, axis="y")
c.set_title("(c) Composition by group"); c.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1)
savefig(fig, "Fig10_feature_selection", "AUC vs number of selected features, selected features and group composition.")

lines = [f"**RQ3 — Tuyển chọn đặc trưng** ({BEST_COND}, DEV):",
         f"- {len(FEATURES)} đặc trưng ban đầu chỉ tạo ~{N_CLUST90} cụm độc lập (|ρ| < 0,9) → dư thừa lớn.",
         f"- AUC trung bình 4 mô hình: toàn bộ = {CURVE.loc[len(FEATURES), 'mean']:.3f}; tốt nhất = {best_auc:.3f} "
         f"(k = {int(CURVE['mean'].idxmax())}).",
         f"- **Bộ tối thiểu K_MIN = {K_MIN} đặc trưng** giữ AUC trong phạm vi {cfg.k_tol} so với tốt nhất "
         f"(AUC {CURVE.loc[K_MIN, 'mean']:.3f}); theo từng mô hình: {K_MIN_MODEL}.",
         f"- **Bộ làm việc K_MAIN = {K_MAIN} đặc trưng** (AUC {CURVE.loc[K_MAIN, 'mean']:.3f}).",
         f"- Thành phần K_MIN theo nhóm: {SEL[SEL.in_K_MIN].group.value_counts().to_dict()}.",
         f"- Độ ổn định (đồng thuận, k = K_MAIN): {STAB[(STAB.ranker == 'consensus') & (STAB.k == K_MAIN)].stability.iloc[0]:.2f}."]
display(Markdown("\n".join(lines)))
