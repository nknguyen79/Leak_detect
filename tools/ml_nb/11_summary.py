# %% [markdown]
# ## 11. Tổng hợp kết quả cho bài báo / Results summary for the manuscript
#
# Đoạn dưới đây được **sinh tự động từ số đo** của lần chạy này (không có số viết cứng). Sao chép vào mục
# *Results* rồi biên tập văn phong. Toàn bộ bảng (`tables/*.csv`, `*.tex`) và hình (`figures/*.png`, `*.pdf`,
# 300 dpi) nằm trong thư mục đầu ra và được nén thành `leak_ml_outputs.zip`.

# %%
# =========================================================
# 11.1 Đoạn kết quả tự sinh + tệp JSON các con số chính
# =========================================================
_m = E4_DEV[E4_DEV.featset == "K_MAIN"].set_index("model")
_bm = _m.loc[BEST_MODEL]
res_lines = [
    f"### Kết quả chính {'(DỮ LIỆU MÔ PHỎNG — KHÔNG DÙNG ĐỂ BÁO CÁO)' if SYNTHETIC else ''}",
    f"**Dữ liệu.** {len(records)} bản ghi độc lập ({int(records.label.sum())} leak / {int((1 - records.label).sum())} noleak), "
    f"{len(files)} tệp, {records.duration_s.sum() / 60:.1f} phút, {SR} Hz; khung {cfg.frame_sec:g} s. "
    f"DEV = {int((records.split == 'dev').sum())} bản ghi; TEST khoá = {int((records.split == 'test').sum())} bản ghi.",
    f"**Tiền xử lý.** Bộ lọc bất thường + không liên quan loại {100 * (1 - QC_FRAMES.keep_C3.mean()):.1f}% số khung "
    f"(leak {100 * rej.loc['rejected_C3', 'leak']:.1f}%, noleak {100 * rej.loc['rejected_C3', 'noleak']:.1f}%; "
    f"khác biệt giữa hai lớp: Mann–Whitney p = {p_rej:.3f}).",
    f"**RQ1 (dải tần).** Năng lượng tương đối cao hơn ở leak trong: {E1_SUMMARY['leak_higher']}; cao hơn ở noleak trong: "
    f"{E1_SUMMARY['noleak_higher']}. Cửa sổ một octave tốt nhất {E1_SUMMARY['best_octave']} (AUC {E1_SUMMARY['best_octave_auc']:.3f}; "
    f"toàn dải {E1_SUMMARY['auc_full']:.3f}). Quét dải: thông thấp đến {E1_SUMMARY['fc_lp']:.0f} Hz và thông cao từ "
    f"{E1_SUMMARY['fc_hp']:.0f} Hz giữ AUC trong phạm vi {cfg.e1_tol} so với toàn dải"
    + (f"; dải thiết yếu {E1_SUMMARY['ess']['f_lo']:.0f}–{E1_SUMMARY['ess']['f_hi']:.0f} Hz đạt AUC {E1_SUMMARY['ess']['auc']:.3f}."
       if E1_SUMMARY["ess"] else " (thông tin lặp lại ở cả hai phía phổ)."),
    f"**RQ2 (xử lý vs thô).** AUC mức bản ghi trung bình 4 mô hình: thô {E2_SUMMARY['raw_mean']:.3f} → "
    f"{E2_SUMMARY['best']} {E2_SUMMARY['best_mean']:.3f} (Δ {E2_SUMMARY['delta_mean']:+.3f}). Khác biệt có ý nghĩa sau Holm ở: "
    f"{', '.join(E2_SUMMARY['sig_better']) if E2_SUMMARY['sig_better'] else 'không mô hình nào'}."
    + (" Trên TEST: " + "; ".join(f"{r.model} {r.auc_raw:.3f}→{r.auc_best:.3f} (p_Holm {r.p_delong_holm:.3f})"
                                    for r in E4C.itertuples()) + "." if E4C is not None else ""),
    f"**RQ3 (chọn đặc trưng).** {len(FEATURES)} đặc trưng (~{N_CLUST90} cụm độc lập). Bộ tối thiểu **{K_MIN}** đặc trưng giữ "
    f"AUC trong phạm vi {cfg.k_tol} so với tốt nhất; bộ làm việc **{K_MAIN}** đặc trưng. Nhóm chiếm ưu thế trong K_MIN: "
    f"{', '.join(f'{g} ({n})' for g, n in SEL[SEL.in_K_MIN].group.value_counts().items())}.",
    f"**Mô hình (CV lồng, DEV, K_MAIN).** " + "; ".join(
        f"{m}: AUC {_m.loc[m, 'rec_auc_rep_mean']:.3f} ± {_m.loc[m, 'rec_auc_rep_sd']:.3f}, BAcc {_m.loc[m, 'rec_bacc']:.3f}, "
        f"F1 {_m.loc[m, 'rec_f1']:.3f}" for m in cfg.models) + f". Tốt nhất: **{BEST_MODEL}**.",
]
if HAS_TEST:
    _t = E4_TEST[E4_TEST.featset == "K_MAIN"].set_index("model")
    res_lines.append(f"**TEST khoá ({len(DST.recs)} bản ghi, K_MAIN).** " + "; ".join(
        f"{m}: AUC {fmt_ci(_t.loc[m, 'rec_auc'], _t.loc[m, 'rec_auc_lo'], _t.loc[m, 'rec_auc_hi'])}, "
        f"Sens {_t.loc[m, 'rec_sens']:.2f}, Spec {_t.loc[m, 'rec_spec']:.2f}" for m in cfg.models) + ".")
res_lines.append(
    f"**Độ bền.** Chỉ mức RMS: AUC {ROB[ROB.test.str.startswith('level only')].rec_auc.max():.3f}; chọn lại không dùng đặc trưng "
    f"phụ thuộc mức ({BEST_MODEL}): ΔAUC {ROB[(ROB.test.str.contains('excluded')) & (ROB.model == BEST_MODEL)].d_vs_full.iloc[0]:+.3f}. "
    f"Chia fold theo khung làm AUC bản ghi tăng giả {LEAKDEMO.rec_auc[2]:+.3f}. "
    f"Nghe liên tục {LT_MIN_S:.0f} s đạt AUC trong phạm vi 0,02 so với nghe toàn bộ bản ghi (AUC {AUC_ALL_FRAMES:.3f}).")
RESULTS_MD = "\n\n".join(res_lines)
display(Markdown(RESULTS_MD))
(OUT / "RESULTS_summary.md").write_text(RESULTS_MD, encoding="utf-8")

KEY = dict(synthetic=bool(SYNTHETIC), n_records=int(len(records)), n_files=int(len(files)), sr=SR, frame_sec=cfg.frame_sec,
           conditions=list(cfg.conditions), best_condition=BEST_COND, e1=E1_SUMMARY, e2=E2_SUMMARY, K_MIN=K_MIN, K_MAIN=K_MAIN,
           features_K_MIN=FEAT_MIN, features_K_MAIN=FEAT_MAIN, best_model=BEST_MODEL,
           dev_nested={m: dict(auc=float(_m.loc[m, "rec_auc_rep_mean"]), sd=float(_m.loc[m, "rec_auc_rep_sd"])) for m in cfg.models},
           test=({m: float(E4_TEST[(E4_TEST.featset == 'K_MAIN') & (E4_TEST.model == m)].rec_auc.iloc[0]) for m in cfg.models}
                 if HAS_TEST else None),
           config=json.loads(json.dumps(asdict(cfg), default=str)))
(OUT / "key_results.json").write_text(json.dumps(KEY, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

# %%
# =========================================================
# 11.2 Danh mục hình/bảng + nén đầu ra
# =========================================================
display(Markdown("**Danh mục hình**\n\n" + "\n".join(f"- `{n}` — {c}" for n, c in FIG_LOG)))
display(Markdown("**Danh mục bảng**\n\n" + "\n".join(f"- `{p.name}`" for p in sorted(TAB_DIR.glob("*.csv")))))
_zip = shutil.make_archive(str(OUT.parent / "leak_ml_outputs"), "zip", root_dir=OUT,
                           base_dir=".") if not cfg.smoke_test else None
print("Đã nén:", _zip if _zip else "(bỏ qua ở smoke test)")
print("Mô hình đã lưu:", sorted(p.name for p in MODEL_DIR.glob("*.joblib")))

# %% [markdown]
# ### Dùng mô hình đã lưu cho bản ghi mới / Inference on a new recording
#
# ```python
# import joblib, numpy as np
# b = joblib.load("/kaggle/working/leak_ml/models/final_SVM_K_MAIN_k30.joblib")   # tên tệp thực tế ở mục 11.2
# # 1) chạy SignalChain(b["sr"], PARAMS).signals(x) và lấy tín hiệu + mặt nạ khung của b["condition"]
# # 2) trích đặc trưng từng khung bằng FeatureExtractor, giữ đúng thứ tự cột b["features"]
# # 3) chuẩn hoá: z = clip((X - b["prep"]["mu"]) / b["prep"]["sd"], -8, 8) sau khi điền NaN bằng b["prep"]["med"]
# # 4) điểm bản ghi = trung bình xác suất các khung; > 0,5 -> nghi rò rỉ
# ```
#
# **Lưu ý khi viết bài**: nếu thư mục đầu ra có dòng *SYNTHETIC DEMO MODE*, mọi con số chỉ để kiểm tra
# chương trình. Hãy gắn dataset thật và chạy lại toàn bộ trước khi trích dẫn.
