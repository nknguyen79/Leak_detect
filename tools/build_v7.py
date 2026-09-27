#!/usr/bin/env python3
"""Dựng notebook v7 từ v6 một cách TÁI LẬP ĐƯỢC.

Mỗi chỉnh sửa là một phép thay thế có kiểm tra (chuỗi gốc phải xuất hiện đúng số lần
mong đợi), nên nếu v6 thay đổi thì script dừng ngay thay vì âm thầm bỏ sót.

    python tools/build_v7.py notebooks/leak-cnn-pvc-kaggle-v6.ipynb notebooks/leak-cnn-pvc-kaggle-v7.ipynb
"""
import copy
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v7_cells as C  # noqa: E402  (nội dung các cell viết lại toàn bộ)

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "notebooks/leak-cnn-pvc-kaggle-v6.ipynb")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "notebooks/leak-cnn-pvc-kaggle-v7.ipynb")

nb = json.loads(SRC.read_text(encoding="utf-8"))
cells = nb["cells"]
assert len(cells) == 104, f"v6 phải có 104 cell, thấy {len(cells)}"


def src(i):
    return "".join(cells[i]["source"])


def put(i, text):
    cells[i]["source"] = text.splitlines(keepends=True)


def rep(i, old, new, count=1):
    s = src(i)
    n = s.count(old)
    assert n == count, f"cell {i}: tìm thấy {n} lần (mong đợi {count}) chuỗi:\n{old[:200]}"
    put(i, s.replace(old, new))


def rep_re(i, pattern, new, min_count=1):
    s = src(i)
    s2, n = re.subn(pattern, new, s)
    assert n >= min_count, f"cell {i}: regex {pattern!r} khớp {n} lần"
    put(i, s2)


def splice(i, start, end, new):
    """Thay đoạn từ `start` (bao gồm) đến hết `end` (bao gồm)."""
    s = src(i)
    a = s.index(start)
    b = s.index(end, a) + len(end)
    assert s.count(start) == 1, f"cell {i}: điểm đầu không duy nhất"
    put(i, s[:a] + new + s[b:])


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def cell_type(i, t):
    assert cells[i]["cell_type"] == t, f"cell {i} là {cells[i]['cell_type']}, mong đợi {t}"


# ---------------------------------------------------------------------------
# 0. Xoá toàn bộ output của v6 (v7 phải được chạy lại từ đầu)
# ---------------------------------------------------------------------------
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

# ---------------------------------------------------------------------------
# Các cell markdown viết lại toàn bộ
# ---------------------------------------------------------------------------
for i, text in C.MARKDOWN.items():
    cell_type(i, "markdown")
    put(i, text.strip("\n") + "\n")

# ---------------------------------------------------------------------------
# Các cell code viết lại toàn bộ
# ---------------------------------------------------------------------------
for i, text in C.CODE.items():
    cell_type(i, "code")
    put(i, text.strip("\n") + "\n")

# ---------------------------------------------------------------------------
# 1.2 Cấu hình trung tâm (cell 4)
# ---------------------------------------------------------------------------
rep(4, '''#     [V6] Các thay đổi so với v5 được đánh dấu [V6-...]; xem tổng hợp ở mục 16.''',
       '''#     [V6] Các thay đổi so với v5 được đánh dấu [V6-...]; [V7-...] là thay đổi của v7.
#     Tổng hợp đầy đủ ở mục 16.''')
rep(4, '''    leak_keys: tuple = ("leak", "ro_ri", "rori", "rr")''',
       '''    # [V7-D1] Khớp theo TỪ (ranh giới "_", ".", chữ số), không còn khớp chuỗi con.
    #          Bỏ khoá "rr": nó khớp nhầm mọi tên chứa hai chữ r liền nhau.
    leak_keys: tuple = ("leak", "ro_ri", "rori")''')
rep(4, '''# sau vài epoch (validation-trong chỉ ~23 bản ghi) nên việc chọn epoch gần''',
       '''# sau vài epoch (validation-trong chỉ vài chục bản ghi) nên việc chọn epoch gần''')
rep(4, '''    select_smooth_w: int = 3               # làm mịn tiêu chí theo cửa sổ trượt (chống nhiễu chọn epoch)''',
       '''    select_smooth_w: int = 3               # làm mịn tiêu chí theo cửa sổ trượt (chống nhiễu chọn epoch)
    # [V7-B2] v6 KHAI BÁO "không chọn checkpoint trong warm-up" nhưng mã vẫn chọn từ epoch 0
    #         (log v6: "chọn ep1", LeakCNN1D chọn trung bình epoch 0.8). v7 THỰC THI ràng buộc.
    select_after_lr_peak: bool = True''')
rep(4, '''    notch_q: float = 30.0''',
       '''    notch_q: float = 30.0
    # [V7] Chuẩn hoá đỉnh theo TỪNG TỆP. 100 = max|x| (giống v6, tương thích notebook suy luận);
    #      ví dụ 99.9 = phân vị bền với xung nhiễu. Mô hình phổ đều chuẩn hoá theo mẫu nên
    #      tham số này hầu như chỉ ảnh hưởng đặc trưng RMS.
    peak_norm_pct: float = 100.0''')
rep(4, '''    # [V6-M2] v5 đặt 1000 Hz theo giả thuyết "năng lượng rò rỉ < 800 Hz". Mục 3.5
    # của chính notebook cho thấy dải phân biệt mạnh nhất nằm ở ~1.3-2.7 kHz.
    # 2800 Hz bao trọn dải mang thông tin đã đo; cell 3.5b có thể tự chỉnh lại.
    low_band_hz: float = 2800.0
    low_band_auto: bool = True     # cell 3.5b tự đặt lại low_band_hz theo số liệu đo được''',
       '''    # [V7-B3] v6 tự hạ low_band_hz 2800 -> 1000 Hz dựa trên MỘT dải "mạnh nhất" (283-413 Hz,
    #         không còn ý nghĩa sau hiệu chỉnh đa so sánh) và cắt mất dải 1.3-2.7 kHz mà chính
    #         Grad-CAM của LeakCNN2D dùng (~2.5 kHz). v7: CỐ ĐỊNH trước (pre-registered).
    #         Nếu bật auto, dải được mở để bao TOP-3 dải mang thông tin, không bao giờ hạ thấp.
    low_band_hz: float = 2800.0
    low_band_auto: bool = False''')
rep(4, '''    lr: float = 3e-4''',
       '''    lr: float = 3e-4
    record_balanced_loss: bool = False  # [V7] trọng số 1/n_frame(bản ghi): mỗi BẢN GHI đóng góp như nhau''')
rep(4, '''    seed: int = 42
    seeds: tuple = (42,)           # [V6-M11] >1 seed -> báo cáo độ ổn định giữa các seed''',
       '''    seed: int = 42
    # [V7-S1] CV LẶP: mỗi seed = một phân hoạch fold MỚI + khởi tạo trọng số mới.
    #         Seed đầu là lần chạy chính (checkpoint, diễn giải, triển khai); trung bình OOF qua
    #         các lần lặp là ước lượng ỔN ĐỊNH NHẤT và vẫn trung thực (mỗi dự đoán đến từ mô
    #         hình chưa từng thấy bản ghi đó).
    seeds: tuple = (42, 43, 44)''')
rep(4, '''    transfer_backbone: str = "resnet18"   # hoặc 'mobilenet_v3_small\'''',
       '''    transfer_backbone: str = "resnet18"   # hoặc 'mobilenet_v3_small'
    transfer_min_size: int = 128          # [V7] phóng log-Mel lên >=128 -> bản đồ 4x4 thay vì 2x2''')
rep(4, '''    # 8.801 frame chỉ đến từ 116 BẢN GHI độc lập: mọi khoảng tin cậy phải bootstrap
    # theo CỤM bản ghi, nếu không CI sẽ hẹp hơn thực tế ~7-8 lần.''',
       '''    # Hàng nghìn frame chỉ đến từ vài chục BẢN GHI độc lập: mọi khoảng tin cậy phải
    # bootstrap theo CỤM bản ghi, nếu không CI sẽ hẹp hơn thực tế gần một bậc độ lớn.''')
rep(4, '''    calibrate_aggregation: bool = True        # [V6-T9] Platt cho xác suất trước khi gộp mức bản ghi''',
       '''    # [V7-S2] Giảm "ngã rẽ phân tích": quy tắc gộp mức bản ghi được ĐĂNG KÝ TRƯỚC (p_mean trên
    #         xác suất THÔ). Bản có hiệu chỉnh Platt và bản chọn quy tắc trên validation-trong
    #         vẫn được in ra, nhưng chỉ là phân tích độ nhạy. (v6: Platt làm AUC bản ghi của
    #         MultiBranchCNN giảm 0.856 -> 0.801.)
    calibrate_aggregation: bool = False
    record_agg: str = "p_mean"                # p_mean | p_med | p_p90 | frac_pos
    agg_selection: str = "fixed"              # "fixed" (đăng ký trước) | "inner" (chọn trên validation-trong)''')
rep(4, '''    run_baseline_models: bool = True    # LogReg/SVM/GBM trên đặc trưng dải tần - câu hỏi "CNN có thêm giá trị?"
    baseline_models: tuple = ("logreg", "svm", "gbm")''',
       '''    run_baseline_models: bool = True    # LogReg/SVM/GBM trên đặc trưng dải tần - câu hỏi "CNN có thêm giá trị?"
    baseline_models: tuple = ("logreg", "svm", "gbm")
    # [V7-S3] thêm bộ đặc trưng MẠNH hơn: thống kê log-Mel (mean+std theo thời gian của 64 dải,
    #         đã trừ mức trung bình -> bất biến với gain, giống front-end của CNN).
    baseline_feature_sets: tuple = ("spectral13", "logmel_stats")''')
rep(4, '''    run_leakage_ablation: bool = True
    run_frame_ablation: bool = False    # ablation độ dài frame (tốn thêm ~1 chu kỳ huấn luyện)
    frame_sec_grid: tuple = (1.0, 2.0, 4.0)
    run_preproc_ablation: bool = False  # ablation tiền xử lý (notch / high-pass / chuẩn hoá)''',
       '''    run_leakage_ablation: bool = True
    # [V7-A1] Ablation cấu hình tín hiệu/huấn luyện, BẬT mặc định (mỗi biến thể = 1 chu kỳ CV của
    #         MỘT mô hình). Biến thể được "chọn" bằng AUC trên validation-TRONG -> không thiên lệch.
    run_signal_ablation: bool = True
    ablation_model: str = "LeakCNN2D"         # cố định TRƯỚC, không chọn theo kết quả
    ablation_variants: tuple = ("no_fir", "rec_balanced", "sr16k", "frame4s", "frame1s",
                                "no_notch", "no_highpass")''')
rep(4, '''cfg = CFG()
if os.environ.get("LEAK_SMOKE") == "1":
    cfg.smoke_test = True''',
       '''cfg = CFG()
if os.environ.get("LEAK_SMOKE") == "1":
    cfg.smoke_test = True
# [V7] Cho phép chạy ngoài Kaggle mà không sửa mã
if os.environ.get("LEAK_DATA") is not None:
    cfg.data_root = os.environ["LEAK_DATA"]
if os.environ.get("LEAK_OUT"):
    cfg.out_dir = os.environ["LEAK_OUT"]
elif not Path("/kaggle").exists():
    cfg.out_dir = str(Path("./leak_outputs").resolve())''')
rep(4, '''    cfg.run_frame_ablation = cfg.run_preproc_ablation = False
    cfg.seeds = (cfg.seed,)''',
       '''    cfg.seeds = (cfg.seed, cfg.seed + 1)              # vẫn chạy thử nhánh CV lặp
    cfg.ablation_variants = ("no_fir", "sr16k")       # vẫn chạy thử nhánh ablation''')
rep(4, '''MIN_EPOCHS_EFF = (max(cfg.min_epochs, LR_PEAK_EPOCH + 2)
                  if cfg.min_epochs_after_lr_peak else cfg.min_epochs)''',
       '''MIN_EPOCHS_EFF = (max(cfg.min_epochs, LR_PEAK_EPOCH + 2)
                  if cfg.min_epochs_after_lr_peak else cfg.min_epochs)
# [V7-B2] Epoch sớm nhất được phép CHỌN làm checkpoint (thực thi trong train_one_fold)
SELECT_FROM_EPOCH = int(LR_PEAK_EPOCH) if cfg.select_after_lr_peak else 0
assert cfg.epochs > SELECT_FROM_EPOCH, "cfg.epochs phải lớn hơn epoch đỉnh LR"''')
rep(4, '''print(f"Lịch LR     : {cfg.lr_schedule} trên {SCHED_TOTAL} epoch | đỉnh ở epoch {LR_PEAK_EPOCH} "
      f"| min_epochs hiệu dụng {MIN_EPOCHS_EFF}")''',
       '''print(f"Lịch LR     : {cfg.lr_schedule} trên {SCHED_TOTAL} epoch | đỉnh ở epoch {LR_PEAK_EPOCH} "
      f"| min_epochs hiệu dụng {MIN_EPOCHS_EFF} | chỉ chọn checkpoint từ epoch {SELECT_FROM_EPOCH}")
print(f"CV lặp      : {len(cfg.seeds)} lần (seeds {list(cfg.seeds)}) | quy tắc gộp: "
      f"{cfg.record_agg} ({cfg.agg_selection}) | Platt: {cfg.calibrate_aggregation}")
print(f"Thư mục ra  : {OUT}")''')

rep(8, '''    DATA_ROOT = _synthesise_dataset()''',
       '''    DATA_ROOT = _synthesise_dataset(root=str(OUT / "synthetic_demo"))   # [V7] không ghi cứng /kaggle''')

# ---------------------------------------------------------------------------
# 3.1 load_audio (cell 14): chuẩn hoá theo TỆP, tham số hoá phân vị
# ---------------------------------------------------------------------------
rep(14, '''    m = np.max(np.abs(x))
    if m > 1e-9:
        x = x / m                                          # chuẩn hoá đỉnh theo bản ghi''',
        '''    # [V7] chuẩn hoá đỉnh theo TỪNG TỆP (v6 ghi nhầm là "theo bản ghi").
    #      peak_norm_pct=100 -> max|x| như v6; <100 -> phân vị, bền với một xung nhiễu đơn lẻ.
    m = (float(np.max(np.abs(x))) if cfg.peak_norm_pct >= 100
         else float(np.percentile(np.abs(x), cfg.peak_norm_pct)))
    if m > 1e-9:
        x = x / m''')
rep(14, '''print("LƯU Ý TRIỂN KHAI: chuẩn hoá đỉnh theo TOÀN BỘ bản ghi là phép xử lý OFFLINE")''',
        '''print("LƯU Ý TRIỂN KHAI: chuẩn hoá đỉnh theo TOÀN BỘ tệp là phép xử lý OFFLINE")''')

# ---------------------------------------------------------------------------
# 3.3-3.5 EDA: Holm cho mọi họ kiểm định
# ---------------------------------------------------------------------------
rep(16, '''TRAPZ = getattr(np, "trapezoid", None) or getattr(np, "trapz")''',
        '''TRAPZ = getattr(np, "trapezoid", None) or getattr(np, "trapz")

def holm(p):
    # Hiệu chỉnh Holm-Bonferroni (định nghĩa sớm để dùng cho kiểm định dải tần ở mục 3).
    p = np.asarray(p, dtype=float); o = np.argsort(p); m = len(p); adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        v = (m - r) * p[i]; run = max(run, v); adj[i] = min(1.0, run)
    return adj''')
rep(16, '''#     v5 kiểm định trên 229 TỆP trong khi đơn vị thống kê là 116 BẢN GHI
#     (2 tệp của một bản ghi là hai nửa của cùng một lần thu) -> pseudo-replication,
#     p-value bị thiên lệch theo hướng "có ý nghĩa".''',
        '''#     Kiểm định trên TỆP khi đơn vị thống kê là BẢN GHI (các tệp của một bản ghi là các
#     đoạn của cùng một lần thu) là pseudo-replication -> p-value thiên lệch về "có ý nghĩa".''')
rep(17, '''for i, p in enumerate(pv):
    if p == p and p < .05:''',
        '''pv_holm = holm(np.nan_to_num(np.asarray(pv, dtype=float), nan=1.0))   # [V7-S4]
for i, p in enumerate(pv_holm):
    if p < .05:''')
rep(17, '''ax[2].set_title("(c) Relative band energy\\n(* p<0.05, ** p<0.01; n = recordings)")''',
        '''ax[2].set_title("(c) Relative band energy\\n(Holm-adjusted * p<0.05, ** p<0.01; n = recordings)")''')
rep(17, '''    "p_MannWhitney_record": pv,''',
        '''    "p_MannWhitney_record": pv,
    "p_holm": pv_holm,''')
rep(17, '''band_stats["significant"] = band_stats.p_MannWhitney_record < .05''',
        '''band_stats["significant_holm"] = band_stats.p_holm < .05''')
rep(17, '''print("LƯU Ý: v5 kiểm định trên 229 tệp; ở đây n = số bản ghi. p-value thường LỚN HƠN")
print("       (đúng hơn) vì hai nửa của cùng một bản ghi không còn được tính là 2 mẫu.")''',
        '''print(f"Đơn vị thống kê: BẢN GHI (n = {len(band_df)}). Cột p_holm đã hiệu chỉnh {len(BAND_NAMES)} "
      f"kiểm định đồng thời (Holm).")
print(f"Số dải có ý nghĩa SAU hiệu chỉnh: {int(band_stats.significant_holm.sum())}/{len(BAND_NAMES)}")''')
rep(19, '''#     Đơn vị thống kê: BẢN GHI. Kèm CI bootstrap theo cụm để tránh kết luận quá mạnh
#     trên 116 quan sát.''',
        '''#     Đơn vị thống kê: BẢN GHI. Kèm CI bootstrap theo cụm + hiệu chỉnh Holm [V7-S4]
#     để tránh kết luận quá mạnh trên vài chục quan sát.''')
rep(19, '''band_auc = pd.DataFrame(rows)''',
        '''band_auc = pd.DataFrame(rows)
band_auc["p_holm"] = holm(band_auc.p.fillna(1.0).values)      # [V7-S4] họ kiểm định đồng thời''')
rep(19, '''print(f"\\nDải phân biệt mạnh nhất: {best.band} Hz (AUC {best.auc:.3f} "
      f"[{best.auc_lo:.3f}, {best.auc_hi:.3f}], {best.huong})")''',
        '''print(f"\\nDải phân biệt mạnh nhất: {best.band} Hz (AUC {best.auc:.3f} "
      f"[{best.auc_lo:.3f}, {best.auc_hi:.3f}], {best.huong}; p={best.p:.3f}, p_holm={best.p_holm:.3f})")
print(f"Số dải có ý nghĩa SAU hiệu chỉnh Holm: {int((band_auc.p_holm < .05).sum())}/{len(band_auc)}"
      " -> nếu bằng 0, KHÔNG dải đơn lẻ nào đủ bằng chứng để quyết định thiết kế.")''')

# ---------------------------------------------------------------------------
# 4.1 cache frame: khoá phải chứa tần số lấy mẫu (ablation 16 kHz)
# ---------------------------------------------------------------------------
rep(23, '''    key = f"{MANIFEST_FP}_{tag}_{fl}_{hl}_{highpass}_{notch}_{int(apply_filters)}"''',
        '''    key = (f"{MANIFEST_FP}_{tag}_{fl}_{hl}_{highpass}_{notch}_{int(apply_filters)}"
           f"_sr{int(cfg.target_sr)}_pn{cfg.peak_norm_pct:g}")          # [V7] khoá có sr''')
rep(23, '''print(f"Frame / bản ghi: trung vị {int(_nfr.median())} -> design effect ước lượng "''',
        '''print(f"Frame / bản ghi: trung vị {int(_nfr.median())} (min {_nfr.min()}, max {_nfr.max()}) "
      f"-> bản ghi dài có trọng số lớn hơn trong hàm mất mát (xem ablation rec_balanced, mục 9.7)")
print(f"Design effect ước lượng "''')

# ---------------------------------------------------------------------------
# 5.3-5.5 mô hình
# ---------------------------------------------------------------------------
rep(28, '''    def embed(self, x):
        h = self.net(x.unsqueeze(1))''',
        '''    def embed(self, x):
        # [V7] chuẩn hoá theo TỪNG MẪU như các front-end phổ: loại bỏ tín hiệu "độ to" (mức
        # năng lượng khác nhau giữa hai lớp ~ confound) để so sánh công bằng với 4 mô hình kia.
        x = x.float()
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-4)
        h = self.net(x.unsqueeze(1))''')
rep(28, '''    cam_usable = False         # ResNet18 hạ mẫu 32 lần -> bản đồ 2x2: Grad-CAM vô nghĩa''',
        '''    cam_usable = False         # ResNet18 hạ mẫu 32 lần -> bản đồ 4x4 (v7) vẫn quá thô cho Grad-CAM''')
rep(28, '''        if s.shape[-1] < 16 or s.shape[-2] < 16:
            s = F.interpolate(s, size=(max(64, s.shape[-2]), max(64, s.shape[-1])),
                              mode="bilinear", align_corners=False)''',
        '''        # [V7] v6 đưa ảnh 64x63 vào ResNet18 -> bản đồ cuối 2x2, gần như mất cấu trúc T-F.
        ms = int(cfg.transfer_min_size)
        if s.shape[-1] < ms or s.shape[-2] < ms:
            s = F.interpolate(s, size=(max(ms, s.shape[-2]), max(ms, s.shape[-1])),
                              mode="bilinear", align_corners=False)''')
rep(29, '''#     [V6-T6] MỘT module SpecAugment dùng CHUNG cho cả ba nhánh -> mọi nhánh
#     nhìn thấy cùng một dạng nhiễu (v5 chỉ che nhánh Mel).''',
        '''#     [V7-B6] v6 VIẾT là SpecAugment dùng chung cho 3 nhánh nhưng mã chỉ che nhánh Mel.
#     v7: mỗi nhánh có SpecAugment riêng, áp TRƯỚC chuẩn hoá (MFCC: chỉ che thời gian,
#     vì che theo trục hệ số cepstral không có ý nghĩa vật lý).''')
rep(29, '''        self.specaug = SpecAugment()
        self.f_mel = LogMelFrontend(specaug=self.specaug)
        self.f_mfcc = MFCCFrontend()
        self.f_low = LowBandSTFTFrontend()''',
        '''        self.specaug = SpecAugment()
        self.specaug_mfcc = SpecAugment(freq=False)
        self.specaug_low = SpecAugment()
        self.f_mel = LogMelFrontend(specaug=self.specaug)
        self.f_mfcc = MFCCFrontend(specaug=self.specaug_mfcc)
        self.f_low = LowBandSTFTFrontend(specaug=self.specaug_low)''')
rep(29, '''        # SpecAugment nằm trong f_mel (trước chuẩn hoá); hai nhánh kia giữ nguyên
        # tín hiệu vì chúng là biểu diễn bổ sung, không phải bản sao.''',
        '''        # Mỗi front-end tự áp SpecAugment của nó (chỉ khi training, trước chuẩn hoá).''')

# ---------------------------------------------------------------------------
# 7.x huấn luyện
# ---------------------------------------------------------------------------
rep(39, '''# Trên dữ liệu hiện tại, tỉ lệ đó ~0.99 -> tiền đề MIL không được ủng hộ.''',
        '''# Mục 8.7 đo tỉ lệ đó trên dữ liệu thật và in kết luận.''')
rep(42, '''#      [V6-S1/S2] Toàn bộ CI dùng bootstrap theo CỤM bản ghi. v5 bootstrap theo
#      frame -> CI hẹp hơn thực tế ~7-8 lần (8.801 frame nhưng chỉ 116 bản ghi).''',
        '''#      [V6-S1/S2] Toàn bộ CI dùng bootstrap theo CỤM bản ghi. Bootstrap theo frame
#      cho CI hẹp hơn thực tế gần một bậc độ lớn (mục 8.2 đo trực tiếp hệ số này).''')
rep(42, '''print("Metrics OK | bootstrap theo cụm bản ghi | tiêu chí chọn:", list(SELECTION_CRITERIA))''',
        '''def record_auc_simple(oof, agg=None):
    # [V7] AUC mức bản ghi của một vector OOF (gộp theo quy tắc đăng ký trước).
    agg = agg or cfg.record_agg
    mk = ~np.isnan(oof)
    df = pd.DataFrame(dict(r=groups_all[mk], y=y_all[mk], p=oof[mk]))
    fn = {"p_mean": "mean", "p_med": "median"}.get(agg, "mean")
    g = df.groupby("r").agg(y=("y", "first"), p=("p", fn))
    return (float(roc_auc_score(g.y, g.p)) if g.y.nunique() > 1 else np.nan), g

print("Metrics OK | bootstrap theo cụm bản ghi | tiêu chí chọn:", list(SELECTION_CRITERIA))''')

rep(43, '''#   [V6-T9] Fit Platt trên validation-trong, lưu cả xác suất thô và đã hiệu chỉnh.''',
        '''#   [V6-T9] Fit Platt trên validation-trong, lưu cả xác suất thô và đã hiệu chỉnh.
#   [V7-B2] THỰC THI "chỉ chọn checkpoint từ epoch SELECT_FROM_EPOCH" (v6 chỉ khai báo).
#   [V7]    Tuỳ chọn trọng số mức bản ghi (cfg.record_balanced_loss).''')
rep(43, '''    pos_weight = neg / max(pos, 1.0)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))''',
        '''    W_ALL = None
    if cfg.record_balanced_loss and not MIL_ON:
        # mỗi BẢN GHI đóng góp tổng trọng số như nhau -> khớp với đơn vị đánh giá
        _nrec = frames.groupby("record_id").size()
        _w = (1.0 / frames.record_id.map(_nrec).values).astype(np.float32)
        _w = _w / float(_w[np.asarray(tr_fit)].mean())
        W_ALL = torch.as_tensor(_w, device=device)
        _rl = pd.Series(y_all[tr_fit]).groupby(groups_all[tr_fit]).first()
        pos, neg = float(_rl.sum()), float(len(_rl) - _rl.sum())   # cân bằng lớp theo BẢN GHI
    pos_weight = neg / max(pos, 1.0)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device),
                                reduction=("none" if W_ALL is not None else "mean"))''')
rep(43, '''        for xb, yb, _ in dl_tr:''', '''        for xb, yb, ib in dl_tr:''')
rep(43, '''                loss = crit(out[0] if MIL_ON else out, yb_s)''',
        '''                loss = crit(out[0] if MIL_ON else out, yb_s)
                if W_ALL is not None:
                    _ib = torch.as_tensor(ib, device=device).long()
                    loss = (loss * W_ALL.index_select(0, _ib)).mean()''')
rep(43, '''            vv = np.asarray(scores_hist[c][-W:], dtype=float)
            vv = vv[np.isfinite(vv)]
            s = float(vv.mean()) if len(vv) else np.nan
            if np.isfinite(s) and s > best[c]["score"] + 1e-6:''',
        '''            # làm mịn CHỈ trên các epoch được phép chọn (không trộn điểm của pha warm-up)
            _e0 = max(SELECT_FROM_EPOCH, ep - W + 1)
            vv = np.asarray(scores_hist[c][_e0:ep + 1], dtype=float)
            vv = vv[np.isfinite(vv)]
            s = float(vv.mean()) if len(vv) else np.nan
            if ep >= SELECT_FROM_EPOCH and np.isfinite(s) and s > best[c]["score"] + 1e-6:''')
rep(43, '''        if s_out > best_out["auc"] + 1e-5:''',
        '''        if ep >= SELECT_FROM_EPOCH and s_out > best_out["auc"] + 1e-5:   # cùng ràng buộc -> so sánh công bằng''')

# ---------------------------------------------------------------------------
# 8.x kết quả
# ---------------------------------------------------------------------------
rep(51, '''bp = a2.boxplot(data, labels=list(cfg.models), patch_artist=True, widths=.5)''',
        '''# [V7] matplotlib >= 3.9 đổi `labels` -> `tick_labels`; đặt nhãn trục riêng để chạy mọi phiên bản
bp = a2.boxplot(data, patch_artist=True, widths=.5)
a2.set_xticks(range(1, len(cfg.models) + 1)); a2.set_xticklabels(list(cfg.models))''')
rep_re(52, r"\bsel_df\b", "agg_sel_df", min_count=4)
rep(52, '''best_agg = agg_sel_df.groupby("aggregation").inner_auc.mean().idxmax()''',
        '''AGG_INNER_BEST = agg_sel_df.groupby("aggregation").inner_auc.mean().idxmax()
# [V7-S2] quy tắc gộp ĐĂNG KÝ TRƯỚC (mặc định); "inner" = chọn trên validation-trong như v6
best_agg = cfg.record_agg if cfg.agg_selection == "fixed" else AGG_INNER_BEST
print(f"Quy tắc gộp dùng cho báo cáo: {best_agg} ({'đăng ký trước' if cfg.agg_selection == 'fixed' else 'chọn trên validation-trong'})"
      f" | quy tắc validation-trong sẽ chọn: {AGG_INNER_BEST}")''')
rep(52, '''display(Markdown(f"### Table 7. Record-level performance "
                 f"(aggregation rule `{best_agg}`, selected on the inner validation set)"))''',
        '''display(Markdown(f"### Table 7. Record-level performance "
                 f"(aggregation rule `{best_agg}`, "
                 f"{'pre-registered' if cfg.agg_selection == 'fixed' else 'selected on the inner validation set'})"))''')
rep(52, '''display(Markdown("**Basis for the aggregation-rule choice (inner-validation AUC):**"))''',
        '''display(Markdown("**Inner-validation AUC of each aggregation rule (sensitivity analysis):**"))''')
put(52, src(52) + '''
# ---------------------------------------------------------------------
# [V7-B5] DEPLOY_MODEL được mục 10.4 dùng TRƯỚC mục 13 (v6: NameError ở 10.4a).
# Xác định ngay tại đây bằng ĐÚNG tiêu chí của mục 13: AUC mức bản ghi trên validation-TRONG.
# ---------------------------------------------------------------------
_inner_auc = {}
for _n in cfg.models:
    _gi = REC_INNER[_n]
    _inner_auc[_n] = (float(roc_auc_score(_gi.label, _gi[best_agg])) if _gi.label.nunique() > 1
                      else -np.inf)
DEPLOY_MODEL = max(_inner_auc, key=_inner_auc.get)
print(f"\\nMô hình triển khai (chọn trên validation-TRONG, cùng tiêu chí mục 13): {DEPLOY_MODEL}")
''')
rep(53, '''thr_b = youden_threshold(g.label.values, g[best_agg].values) if len(set(g.label)) > 1 else .5''',
        '''# [V7] ngưỡng lấy từ validation-TRONG (v6 vẽ ngưỡng Youden tối ưu trên chính tập báo cáo)
thr_b = youden_threshold(REC_INNER[BEST_MODEL].label.values, REC_INNER[BEST_MODEL][best_agg].values)''')
rep(57, '''    print(" - [V6] ICC cao + tỉ lệ thể hiện dương ~1 (mục 8.7) => điểm số gần như là một HẰNG SỐ")''',
        '''    print(f" - ICC cao (+ tỉ lệ thể hiện dương trung vị {W_LEAK_MED:.2f}, mục 8.7) => điểm số phần lớn là")
    print("   thuộc tính của TỪNG BẢN GHI.")''')
rep(57, '''    print("   của từng bản ghi. Kết luận 'nhận dạng rò rỉ' CHỈ hợp lệ nếu đã loại trừ được")''',
        '''    print("   Kết luận 'nhận dạng rò rỉ' CHỈ hợp lệ nếu đã loại trừ được")''')

# ---------------------------------------------------------------------------
# 9.x thống kê
# ---------------------------------------------------------------------------
rep(60, '''      f"con số này cao vì 8 801 frame chỉ đến từ 116 bản ghi (pseudo-replication).")''',
        '''      f"con số này cao vì {len(frames)} frame chỉ đến từ {len(records)} bản ghi (pseudo-replication).")''')
rep(69, '''        print("(2) BỎ QUA: chưa có thông tin điểm khảo sát (cfg.site_source='none').")''',
        '''        print("(2) BỎ QUA: chưa có thông tin điểm khảo sát (cfg.site_source='none').")
        # [V7-C1] Sinh sẵn tệp mẫu để người thu dữ liệu chỉ việc điền cột `site`.
        _tpl = records[["record_id", "cls", "n_files", "duration_s"]].copy()
        _tpl["site"] = ""
        _tpl.to_csv(OUT / "site_map_template.csv", index=False, encoding="utf-8-sig")
        print(f"    -> Đã tạo {OUT / 'site_map_template.csv'}: điền cột `site` (vị trí/tuyến ống/ngày thu)")
        print("       rồi đặt cfg.site_source='csv', cfg.site_csv=<đường dẫn>, cfg.run_loso_full=True.")''')

# ---------------------------------------------------------------------------
# 10.x diễn giải
# ---------------------------------------------------------------------------
rep(73, '''        logit = self.model(x)
        logit.sum().backward()''',
        '''        # [V7-B1] cuDNN không cho backward của RNN ở chế độ eval -> v6 dừng ở đây với CRNN
        #         ("cudnn RNN backward can only be called in training mode"). Tắt cuDNN cục bộ
        #         giữ nguyên chế độ eval (BatchNorm/Dropout đúng như lúc suy luận).
        with torch.backends.cudnn.flags(enabled=False):
            logit = self.model(x)
            logit.sum().backward()''')
rep(73, '''        top = np.argsort(m_leak)[-3:][::-1]''',
        '''        if (not np.isfinite(m_leak).all()) or float(m_leak.sum()) <= 1e-9:
            # [V7] bản đồ toàn 0 (ReLU triệt tiêu hết) -> argmax vô nghĩa, KHÔNG đưa vào kết luận
            print(f"\\n[{cam_name}] !! Bản đồ Grad-CAM SUY BIẾN (toàn 0) -> loại khỏi kết luận.")
            del mdl; gc.collect()
            continue
        top = np.argsort(m_leak)[-3:][::-1]''')
rep(73, '''    _peak = np.mean([v["peak_hz"] for v in CAM_MASS.values()])
    _share = np.mean([v["share_below_800"] for v in CAM_MASS.values()])''',
        '''    if not CAM_MASS:
        print("!! Không mô hình nào cho bản đồ Grad-CAM hợp lệ -> không rút kết luận từ Grad-CAM.")
    _peak = np.mean([v["peak_hz"] for v in CAM_MASS.values()]) if CAM_MASS else np.nan
    _share = np.mean([v["share_below_800"] for v in CAM_MASS.values()]) if CAM_MASS else np.nan''')
rep(74, '''        if _tiny:
            _cap += (''',
        '''        if base_auc < 0.6:
            # [V7] mô hình gần/kém hơn ngẫu nhiên trên fold này -> ΔAUC khi che không diễn giải được
            _cap += (f" The baseline AUC on this fold ({base_auc:.2f}) is below 0.6, so the occlusion "
                     "profile is NOT interpretable and no band-importance claim is made.")
        elif _tiny:
            _cap += (''')
rep(74, '''        r_sp = float(spstats.spearmanr(o0.auc_drop.values, info).statistic) if len(o0) > 4 else np.nan''',
        '''        r_sp = (float(spstats.spearmanr(o0.auc_drop.values, info).statistic)
                if (len(o0) > 4 and b0 >= 0.6) else np.nan)
        if b0 < 0.6:
            print(f"\\n[{n0}] AUC nền của fold {CAM_FOLD} = {b0:.3f} < 0.6 -> không so sánh occlusion với §3.5.")''')
rep(79, '''# 10.4c t-SNE MỨC BẢN GHI - 116 điểm, đơn vị thống kê đúng''',
        '''# 10.4c t-SNE MỨC BẢN GHI - mỗi bản ghi một điểm, đơn vị thống kê đúng''')
rep(81, '''    print("       (116 bản ghi, ICC cao), không bởi dung lượng mô hình -> thêm tham số hay đổi")''',
        '''    print(f"       ({len(records)} bản ghi, ICC cao), không bởi dung lượng mô hình -> thêm tham số hay đổi")''')
rep(81, '''    print("       Kết hợp với mục 9.5 (CI của CNN chồng lấn baseline cổ điển), đây là bằng chứng")''',
        '''    print("       Kết hợp với mục 9.5 (so sánh GHÉP CẶP CNN vs baseline cổ điển), đây là bằng chứng")''')

# ---------------------------------------------------------------------------
# 11.x xuất bản
# ---------------------------------------------------------------------------
rep(83, '''                 "cross-validation). Confidence intervals are CLUSTER bootstraps over the 116 "''',
        '''                 f"cross-validation). Confidence intervals are CLUSTER bootstraps over the {len(records)} "''')
rep(83, '''                 f"calibrated frame posteriors; Wilson 95% intervals for recall/specificity.",''',
        '''                 f"{'calibrated' if cfg.calibrate_aggregation else 'raw'} frame posteriors; "
                 f"Wilson 95% intervals for recall/specificity.",''')
rep(83, '''                 "RECORD level (n = 116) and paired cluster bootstrap for dAUC.", "stats")])''',
        '''                 f"RECORD level (n = {len(records)}) and paired cluster bootstrap for dAUC.", "stats")])''')

splice(84, 'base_df_ = _gv("base_df")',
       'base_txt, base_verdict = "không chạy", "chưa đánh giá được"',
       '''# [V7-B7] v6 lọc baseline theo tên "CHỈ mức năng lượng (RMS)" trong khi bảng dùng tên
#         "Level only (RMS)" -> baseline 1 đặc trưng có thể bị chọn nhầm làm "tốt nhất".
#         v7 lấy kết luận trực tiếp từ phép so sánh GHÉP CẶP ở mục 9.5.
base_txt = _gv("BEST_BASE_TXT", "không chạy")
base_verdict = _gv("BASELINE_VERDICT_VI", "chưa đánh giá được")
base_verdict_en = _gv("BASELINE_VERDICT_EN", "not evaluated")''')
rep(84, '''_band_txt = ("consistent with a low-frequency-dominated leak signature"
             if narr == "low_band_supported" else
             "NOT concentrated in the very-low-frequency band (< 800 Hz) that is usually "
             "assumed for plastic pipes")
_mil_txt = "does not support" if not mil_ok else "supports"''',
        '''_band_txt = {"low_band_supported": "consistent with a low-frequency-dominated leak signature",
             "low_band_not_supported": "NOT concentrated in the very-low-frequency band (< 800 Hz) "
                                       "that is usually assumed for plastic pipes",
             }.get(narr, "not resolvable between the low (< 800 Hz) and upper bands (the bootstrap "
                         "CI of the difference includes zero)")
_band_txt_vi = {"low_band_supported": "nằm ở vùng tần số thấp",
                "low_band_not_supported": "KHÔNG nằm ở vùng tần số rất thấp (< 800 Hz) như thường "
                                          "được giả định cho ống nhựa",
                }.get(narr, "CHƯA phân định được giữa dải thấp và dải cao (CI của hiệu số chứa 0)")
_wv = _gv("WITNESS_VERDICT", "not_supported")
_mil_txt = {"supported": "supports", "not_supported": "does not support"}.get(_wv, "is inconclusive about")
_mil_txt_vi = {"supported": "ủng hộ", "not_supported": "KHÔNG ủng hộ"}.get(_wv, "chưa đủ để kết luận về")
_rs = (_gv("REPEAT_SUMMARY", {}) or {}).get(rb.model)
if _rs and _rs.get("n_repeats", 1) > 1:
    rep_txt_en = (f"Across {_rs['n_repeats']} repetitions of the cross-validation with different "
                  f"partitions, the record-level AUC of {rb.model} was {_rs['mean']:.3f} +/- {_rs['sd']:.3f}; "
                  f"averaging the out-of-fold posteriors over repetitions gave AUC = {_rs['ens_auc']:.3f} "
                  f"(CI {_rs['ens_lo']:.3f}-{_rs['ens_hi']:.3f}).")
    rep_txt_vi = (f"Qua {_rs['n_repeats']} lần lặp CV với phân hoạch khác nhau, AUC mức bản ghi của "
                  f"{rb.model} là {_rs['mean']:.3f} ± {_rs['sd']:.3f}; trung bình xác suất out-of-fold qua "
                  f"các lần lặp cho AUC = {_rs['ens_auc']:.3f} (CI {_rs['ens_lo']:.3f}-{_rs['ens_hi']:.3f}).")
else:
    rep_txt_en = rep_txt_vi = ""
_dep = _gv("DEPLOY_MODEL", rb.model)''')
rep(84, '''zero; the remaining pairs are statistically indistinguishable in this dataset.''',
        '''zero; the remaining pairs are statistically indistinguishable in this dataset. {rep_txt_en}
The deployment model, chosen on the inner validation sets only, is {_dep}.''')
rep(84, '''best = {base_txt}, {base_verdict}.''', '''best = {base_txt}; {base_verdict_en}.''')
rep(84, '''chứa 0.
''', '''chứa 0. {rep_txt_vi} Mô hình triển khai (chọn chỉ trên validation-trong): {_dep}.
''')
rep(84, '''{'KHÔNG ủng hộ' if not mil_ok else 'ủng hộ'} tiền đề học đa thể hiện; (ii) dải tần mang thông tin
{'nằm ở vùng tần số thấp' if narr == 'low_band_supported' else 'KHÔNG nằm ở vùng tần số rất thấp (< 800 Hz) như thường được giả định cho ống nhựa'};''',
        '''{_mil_txt_vi} tiền đề học đa thể hiện; (ii) dải tần mang thông tin
{_band_txt_vi};''')

# [V7-B9] Đoạn Results của v6 trộn hai mô hình: AUC bản ghi của mô hình tốt nhất mức bản ghi
#         trừ AUC frame của mô hình tốt nhất mức frame. Nay cùng MỘT mô hình.
rep(84, '''d_rec = float(rb.auc) - float(b.AUC)''',
        '''d_rec = float(rb.auc) - float(frame_df.set_index("model").AUC[rb.model])   # [V7-B9] cùng một mô hình''')
rep(84, '''{rb.recall:.3f}, specificity {rb.specificity:.3f}). Aggregation changed frame-level AUC by
{d_rec:+.3f}.''',
        '''{rb.recall:.3f}, specificity {rb.specificity:.3f}). For {rb.model}, aggregation changed the AUC by
{d_rec:+.3f} relative to its own frame-level value.''')
rep(84, '''recording-grouped cross-validation with a nested inner-validation protocol that selects the
stopping epoch, the decision threshold and the record-level aggregation rule without touching the
reported folds.''',
        '''recording-grouped cross-validation with a nested inner-validation protocol that selects the
stopping epoch and the decision threshold without touching the reported folds; the record-level
aggregation rule ({best_agg}) was {'fixed in advance' if cfg.agg_selection == 'fixed' else 'selected on the inner validation sets'}.''')
rep(84, '''                 f"per-recording score is nearly constant (ICC = {icc_mean:.2f}) - it may be "
                 "discriminating acquisition sites. ")''',
        '''                 f"per-recording score is largely recording-specific (ICC = {icc_mean:.2f}) - it may be "
                 "discriminating acquisition sites; nor ")''')
rep(84, '''(iii) Confound screening: {conf_verdict};''',
        '''(iii) Confound screening: {_conf_en};''')
rep(84, '''(logistic regression / SVM / gradient boosting on band-energy and spectral features, same splits):''',
        '''(logistic regression / SVM / gradient boosting on hand-crafted spectral features and on gain-invariant
log-Mel statistics, same splits, paired comparison):''')
rep(84, '''_cf_up = str(conf_verdict).upper()''',
        '''_cf_up = str(conf_verdict).upper()
_conf_en = {"CÓ DẤU HIỆU CONFOUND": "signs of site/level confounding were found",
            "CHƯA LOẠI TRỪ ĐƯỢC CONFOUND": "site confounding could not be excluded (no site information)",
            "Không thấy dấu hiệu confound rõ": "no clear sign of confounding in the indirect checks"
            }.get(str(conf_verdict), str(conf_verdict))''')

# ---------------------------------------------------------------------------
# 13 / 15: đổi tên sel_df -> dep_sel_df (v6 ghi đè sel_df của mục 8.5)
# ---------------------------------------------------------------------------
for i in (89, 90, 100):
    rep_re(i, r"\bsel_df\b", "dep_sel_df")
rep(89, '''#      trong khi tiêu chí chọn (0.9955 vs 0.9842) đã BÃO HOÀ gần 1.0 trên ~23 bản ghi/fold và''',
        '''#      trong khi tiêu chí chọn đã BÃO HOÀ gần 1.0 trên vài chục bản ghi/fold và''')
rep(100, '''if not bool(_gv("MIL_PREMISE_SUPPORTED", False)):
    _limitations.append("Tỉ lệ thể hiện dương ~1: tiền đề 'nhãn mức frame bị nhiễu' KHÔNG được "
                        "dữ liệu ủng hộ; không dùng mô hình này để định vị thời điểm rò rỉ.")
if _gv("BAND_NARRATIVE", "") == "low_band_not_supported":''',
         '''_wv = _gv("WITNESS_VERDICT", "not_supported")
if _wv != "supported":
    _limitations.append(f"Tỉ lệ thể hiện dương trung vị {float(_gv('W_LEAK_MED', np.nan)):.2f} "
                        f"({'không ủng hộ' if _wv == 'not_supported' else 'chưa đủ để kết luận về'}) "
                        "tiền đề 'nhãn mức frame bị nhiễu'; không dùng mô hình này để định vị thời điểm rò rỉ.")
if _gv("BAND_NARRATIVE", "") == "inconclusive":
    _limitations.append("Chưa phân định được dải tần mang thông tin (CI của hiệu số chứa 0); xem mục 3.5b.")
if _gv("BAND_NARRATIVE", "") == "low_band_not_supported":''')
rep(100, '''    "format_version": 3,''', '''    "format_version": 4,''')
rep(100, '''        "peak_normalise": True, "causal_online": False,''',
         '''        "peak_normalise": True, "peak_norm_pct": float(cfg.peak_norm_pct),
        "peak_norm_unit": "file", "causal_online": False,''')
rep(100, '''if len(cfg.seeds) < 2:''', '''if len(cfg.seeds) < 2:  # CV lặp''')

# ---------------------------------------------------------------------------
# Chèn cell mới (chỉ số là vị trí trong v6; chèn TRƯỚC cell đó)
# ---------------------------------------------------------------------------
INSERT_BEFORE = {46: [md(C.MD_REPEATED_CV)]}
new_cells = []
for i, c in enumerate(cells):
    new_cells.extend(copy.deepcopy(INSERT_BEFORE.get(i, [])))
    new_cells.append(c)
nb["cells"] = new_cells

# ---------------------------------------------------------------------------
# Kiểm tra cuối: không còn số liệu cũ bị viết cứng trong văn bản
# ---------------------------------------------------------------------------
STALE = re.compile(r"\b116\b|8[ .]801|\b229\b|~23 bản ghi|0,99\)|trung vị 0,99|ICC ≈ 0,75")
bad = []
for i, c in enumerate(nb["cells"]):
    s = "".join(c["source"])
    if i == len(nb["cells"]) - 1:
        continue  # changelog được phép nhắc số liệu lịch sử của v6
    for ln in s.splitlines():
        if STALE.search(ln):
            bad.append((i, ln.strip()[:120]))
assert not bad, "Còn số liệu cũ bị viết cứng:\n" + "\n".join(map(str, bad))

# id ổn định cho mọi cell (nbformat >= 4.5)
nb["nbformat"], nb["nbformat_minor"] = 4, 5
for i, c in enumerate(nb["cells"]):
    c["id"] = f"v7-{i:03d}"
nb.setdefault("metadata", {})["leak_cnn_version"] = "v7"
DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"OK -> {DST} ({len(nb['cells'])} cell)")
