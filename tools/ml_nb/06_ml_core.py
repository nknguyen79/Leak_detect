# %% [markdown]
# ## 6. Lõi học máy dùng chung / Shared machine-learning core
#
# **Bốn mô hình** (được dùng nhiều nhất trong các công trình nhận dạng rò rỉ bằng âm thanh/rung động —
# xem tổng quan của Fan, Tariq & Zayed 2022):
#
# | Mô hình | Lý do chọn | Cài đặt |
# |---|---|---|
# | **SVM (RBF)** | chuẩn mực phổ biến nhất cho đặc trưng thủ công (Kang et al. 2018; Xu et al. 2021; Fares et al. 2023) | `SVC`, γ = hệ số × 1/k, trọng số mẫu cân bằng |
# | **Random Forest** | bền với đặc trưng dư thừa, cho độ quan trọng (Breiman 2001; Ning et al. 2021) | 300 cây, trọng số mẫu cân bằng |
# | **XGBoost** | gradient boosting mạnh nhất trên dữ liệu bảng (Chen & Guestrin 2016; Tijani et al. 2022) | 300 cây `hist` (thay bằng HistGradientBoosting nếu thiếu) |
# | **k-NN** | mốc phi tham số kinh điển (Cover & Hart 1967; El-Zahab et al. 2018) | khoảng cách Euclid trên dữ liệu chuẩn hoá; hiệu chỉnh tiên nghiệm lớp |
#
# **Giao thức chống rò rỉ dữ liệu** (Kaufman et al. 2012; Roberts et al. 2017; Varma & Simon 2006):
# điền khuyết → chuẩn hoá → **chọn đặc trưng** → huấn luyện đều được `fit` **chỉ trên phần huấn luyện của
# từng fold**; fold được chia trên **bảng bản ghi** (phân tầng theo lớp) nên không bản ghi nào xuất hiện ở
# cả hai phía. Đầu ra mức khung được gộp về **mức bản ghi bằng trung bình xác suất** (quy tắc đăng ký
# trước); ngưỡng quyết định 0,5 trên mô hình đã cân bằng lớp.

# %%
# =========================================================
# 6.1 Tiền xử lý, mô hình, bộ xếp hạng đặc trưng
# =========================================================
CLIP_Z = 8.0
SUPPORTS_SW = {"SVM": True, "RF": True, "XGB": True, "KNN": False, "LR": True, "MLP": False}


class Prep:
    """Điền khuyết bằng trung vị + chuẩn hoá z + kẹp |z| <= 8 (fit trên TRAIN)."""

    def fit(self, X):
        X = np.asarray(X, np.float64)
        self.med = np.nanmedian(X, 0); self.med[~np.isfinite(self.med)] = 0.0
        Xf = np.where(np.isfinite(X), X, self.med)
        self.mu = Xf.mean(0); self.sd = Xf.std(0); self.sd[self.sd < 1e-12] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, np.float64); X = np.where(np.isfinite(X), X, self.med)
        return np.clip((X - self.mu) / self.sd, -CLIP_Z, CLIP_Z).astype(np.float32)


def build_model(name, params=None, seed=0, n_feat=10, n_train=1000):
    p = dict(DEFAULT_PARAMS[name]); p.update(params or {})
    if name == "SVM":
        return SVC(kernel="rbf", C=p["C"], gamma=p["gamma_mult"] / max(n_feat, 1), cache_size=500)
    if name == "RF":
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_features=p["max_features"],
                                      min_samples_leaf=p["min_samples_leaf"], n_jobs=1, random_state=seed)
    if name == "XGB":
        if HAS_XGB:
            return xgb.XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                                     subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
                                     min_child_weight=p["min_child_weight"], reg_lambda=p["reg_lambda"],
                                     tree_method="hist", n_jobs=1, random_state=seed, verbosity=0, eval_metric="logloss")
        return HistGradientBoostingClassifier(max_iter=p["n_estimators"], max_depth=p["max_depth"],
                                              learning_rate=p["learning_rate"], random_state=seed)
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=int(min(p["n_neighbors"], max(1, n_train - 1))), weights=p["weights"], n_jobs=1)
    if name == "LR":
        return LogisticRegression(C=p["C"], max_iter=3000)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=p["hidden_layer_sizes"], alpha=p["alpha"], early_stopping=True,
                             max_iter=500, random_state=seed)
    raise ValueError(name)


def prior_correct(p, pi):
    """Đưa xác suất của mô hình KHÔNG cân bằng lớp về tiên nghiệm 50/50 (ngưỡng 0,5 = cân bằng)."""
    p = np.clip(p, 1e-6, 1 - 1e-6); a = p * 0.5 / pi; b = (1 - p) * 0.5 / (1 - pi)
    return a / (a + b)


def fit_model(name, A, y, g, params=None, seed=0):
    m = build_model(name, params, seed, n_feat=A.shape[1], n_train=len(y))
    if SUPPORTS_SW[name]:
        m.fit(A, y, sample_weight=sample_weights(y, g))
    else:
        m.fit(A, y)
    m.prior_ = float(np.mean(y))
    return m


def score_model(name, m, B):
    p = expit(m.decision_function(B)) if name == "SVM" else m.predict_proba(B)[:, 1]
    return prior_correct(p, m.prior_) if not SUPPORTS_SW[name] else p


# ---- Bộ xếp hạng đặc trưng (trả về thứ tự chỉ số, tốt nhất trước) ---------------
def _sub(A, y, g, n, seed):
    if len(y) <= n:
        return A, y, g
    i = np.random.default_rng(seed).choice(len(y), n, replace=False)
    return A[i], y[i], g[i]


def rank_anova(A, y, g, seed):
    F = np.nan_to_num(f_classif(A, y)[0]); return np.argsort(-F, kind="stable")


def rank_mi(A, y, g, seed):
    A2, y2, _ = _sub(A, y, g, min(cfg.rank_max_rows, 4000), seed)
    mi = mutual_info_classif(A2, y2, n_neighbors=3, random_state=seed)
    return np.lexsort((-np.nan_to_num(f_classif(A, y)[0]), -mi))


def rank_mrmr(A, y, g, seed, k_max=None):
    """mRMR dạng FCQ: độ liên quan = F-ANOVA, độ dư thừa = |tương quan Pearson| (Ding & Peng 2005; Peng et al. 2005)."""
    F = np.nan_to_num(f_classif(A, y)[0]); p = A.shape[1]; k_max = k_max or p
    C = np.nan_to_num(np.abs(np.corrcoef(A, rowvar=False)))
    sel = [int(np.argmax(F))]; red = C[:, sel[0]].copy(); used = np.zeros(p, bool); used[sel[0]] = True
    while len(sel) < min(k_max, p):
        sc = F / (red / len(sel) + 1e-3); sc[used] = -np.inf
        j = int(np.argmax(sc)); sel.append(j); used[j] = True; red += C[:, j]
    rest = [j for j in np.argsort(-F) if not used[j]]
    return np.array(sel + rest)


def rank_et(A, y, g, seed):
    A2, y2, g2 = _sub(A, y, g, cfg.rank_max_rows, seed)
    et = ExtraTreesClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=3, n_jobs=1, random_state=seed)
    et.fit(A2, y2, sample_weight=sample_weights(y2, g2))
    return np.argsort(-et.feature_importances_, kind="stable")


def rank_l1(A, y, g, seed):
    lr = LogisticRegression(penalty="l1", solver="liblinear", C=cfg.l1_C, max_iter=2000)
    lr.fit(A, y, sample_weight=sample_weights(y, g))
    return np.lexsort((-np.nan_to_num(f_classif(A, y)[0]), -np.abs(lr.coef_[0])))


RANKERS = {"anova": rank_anova, "mi": rank_mi, "mrmr": rank_mrmr, "et": rank_et, "l1": rank_l1}


def consensus(orders):
    """Xếp hạng đồng thuận Borda: trung bình vị trí qua các phương pháp (Saeys et al. 2008)."""
    p = len(orders[0]); pos = np.zeros((len(orders), p))
    for i, o in enumerate(orders):
        pos[i, o] = np.arange(p)
    return np.argsort(pos.mean(0), kind="stable")

# %%
# =========================================================
# 6.2 Bộ máy CV lặp (song song theo fold) + tổng hợp chỉ số + so sánh ghép cặp
# =========================================================
class DS:
    """Khung nhìn dữ liệu của một điều kiện trên một phạm vi bản ghi."""

    def __init__(self, cond, scope, feats=None):
        df = FEAT[cond]; recs = np.sort(SCOPE_RECS[scope])
        m = df.record_id.isin(recs).values
        self.df = df.loc[m, META_COLS].reset_index(drop=True)
        self.feats = list(feats or FEATURES)
        self.X = np.array(df.loc[m, self.feats].to_numpy(np.float32), copy=True)
        self.y = self.df.label.to_numpy(int); self.g = self.df.record_id.to_numpy()
        self.recs = recs; self.rec_y = REC_LABEL.loc[recs].values
        self.cond, self.scope = cond, scope


def fold_job(Xtr, ytr, gtr, Xte, combos, rankers, seed):
    prep = Prep().fit(Xtr); A, B = prep.transform(Xtr), prep.transform(Xte)
    orders = {}
    need = [r for r in rankers if r != "consensus"]
    if "consensus" in rankers:
        need += [r for r in cfg.rankers if r not in need]
    for r in need:
        orders[r] = RANKERS[r](A, ytr, gtr, seed)
    if "consensus" in rankers:
        orders["consensus"] = consensus([orders[r] for r in cfg.rankers])
    out = {}
    for rk, k, model in combos:
        idx = np.arange(A.shape[1]) if (not k or rk is None) else orders[rk][:k]
        m = fit_model(model, A[:, idx], ytr, gtr, seed=seed)
        out[(rk, k, model)] = score_model(model, m, B[:, idx]).astype(np.float32)
    return out, {r: o.astype(np.int16) for r, o in orders.items()}


class CVRun:
    def __init__(self, ds, seeds):
        self.ds, self.seeds = ds, list(seeds)
        self.scores = {}; self.orders = defaultdict(list); self.fold_of = np.full((len(seeds), len(ds.y)), -1)


def run_cv(ds, combos, rankers=(), seeds=None, tag=""):
    seeds = list(seeds or cfg.seeds); t0 = time.time()
    run = CVRun(ds, seeds); jobs = []
    for r, s in enumerate(seeds):
        for fi, te_r in enumerate(FOLDS[(ds.scope, s)]):
            te = np.isin(ds.g, te_r); run.fold_of[r, te] = fi; jobs.append((r, fi, te))
    res = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(fold_job)(ds.X[~te], ds.y[~te], ds.g[~te], ds.X[te], combos, tuple(rankers), 7919 * seeds[r] + fi)
        for r, fi, te in jobs)
    for (r, fi, te), (out, orders) in zip(jobs, res):
        for key, sc in out.items():
            run.scores.setdefault(key, np.full((len(seeds), len(ds.y)), np.nan, np.float32))[r, te] = sc
        for rk, o in orders.items():
            run.orders[rk].append(o)
    print(f"  [{tag}] {ds.cond}/{ds.scope}: {len(jobs)} fold x {len(combos)} cấu hình trong {time.time() - t0:.0f} s")
    return run


def summarize(run, key, boot=True):
    ds = run.ds; S = run.scores[key]; R = S.shape[0]
    rep_rec = [rec_scores(S[r], ds.g, ds.recs) for r in range(R)]
    rec_auc_rep = [auc_safe(ds.rec_y, v) for v in rep_rec]
    fold_auc = []
    for r in range(R):
        for fi, te_r in enumerate(FOLDS[(ds.scope, run.seeds[r])]):
            ii = np.isin(ds.recs, te_r); fold_auc.append(auc_safe(ds.rec_y[ii], rep_rec[r][ii]))
    s_mean = np.nanmean(S, 0); rs = rec_scores(s_mean, ds.g, ds.recs)
    out = dict(rec_auc=auc_safe(ds.rec_y, rs), rec_auc_rep_mean=np.nanmean(rec_auc_rep), rec_auc_rep_sd=np.nanstd(rec_auc_rep),
               frame_auc=auc_safe(ds.y, s_mean), rec_ap=average_precision_score(ds.rec_y, rs))
    if boot:
        out["rec_auc_lo"], out["rec_auc_hi"] = boot_auc_ci(ds.rec_y, rs, seed=1)
        out["frame_auc_lo"], out["frame_auc_hi"] = boot_auc_ci(ds.y, s_mean, n_boot=max(200, cfg.n_boot // 2), seed=2, groups=ds.g)
    out.update({f"rec_{k}": v for k, v in bin_metrics(ds.rec_y, rs).items()
                if k in ("acc", "bacc", "sens", "spec", "f1", "mcc")})
    out["_rec_scores"] = rs; out["_fold_auc"] = np.asarray(fold_auc)
    return out


def compare(sumA, sumB, rec_y, n_splits=None):
    """So sánh ghép cặp A - B trên cùng các bản ghi: bootstrap, DeLong, Nadeau-Bengio."""
    d, p_dl = delong_test(rec_y, sumA["_rec_scores"], sumB["_rec_scores"])
    lo, hi, p_bt = paired_boot_delta(rec_y, sumA["_rec_scores"], sumB["_rec_scores"], seed=3)
    K = n_splits or cfg.n_splits
    p_nb = nb_ttest(sumA["_fold_auc"] - sumB["_fold_auc"], K - 1, 1)
    return dict(d_auc=d, d_lo=lo, d_hi=hi, p_delong=p_dl, p_boot=p_bt, p_nb=p_nb)


print("Lõi ML sẵn sàng | mô hình:", list(cfg.models), "| bộ xếp hạng:", list(cfg.rankers) + ["consensus"])
