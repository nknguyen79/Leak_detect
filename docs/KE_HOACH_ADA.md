# Kế hoạch thực hiện Adversarial Domain Adaptation (Thích ứng miền đối nghịch) cho tập `tapdulieuMix_16k`

> Tài liệu đi kèm notebook `notebooks/leak-ada-dann-kaggle-v1.ipynb` (dựng bằng `tools/build_ada.py`).
> Mục tiêu: một thí nghiệm **chạy trọn trên Kaggle (GPU T4 ×2)**, sinh đủ bảng/hình/văn bản nháp để viết bài báo
> *"Adversarial Domain Adaptation for Acoustic Leak Detection in Low-Pressure PVC Pipes across Acquisition Domains"*.

---

## 1. Tóm tắt

Pipeline v7 đánh giá nhận dạng rò rỉ **trong cùng một miền thu âm**. Tập `tapdulieuMix_16k` trộn dữ liệu từ nhiều
nguồn thu (thư mục con = miền), nên câu hỏi khoa học đổi thành: *mô hình học ở miền này có dùng được ở miền kia không,
và học đối nghịch có thu hẹp khoảng cách đó mà **không cần nhãn** ở miền đích không?*

Kế hoạch gồm 4 khối:

1. **Đo dịch chuyển miền trước khi huấn luyện** (băng thông, PSD, chiều của dấu hiệu rò rỉ trong từng miền, trùng lặp
   nội dung, PAD đầu vào, ma trận "huấn luyện A → kiểm thử B" của bộ phân lớp cổ điển).
2. **So sánh 6 phương pháp trên cùng một mạng**: Source-only, AdaBN, Deep CORAL, **DANN**, **CDAN+E**, Oracle (cận trên).
3. **Giao thức quy nạp, chặt chẽ thống kê**: K-fold trên *bản ghi* đích, không dùng nhãn đích để chọn mô hình, 3 seed,
   bootstrap theo bản ghi, ΔAUC ghép cặp + Holm.
4. **Phân tích cơ chế**: động học đối nghịch, PAD/MMD/silhouette của đặc trưng, t-SNE trước/sau, quét λ, ablation,
   chọn mô hình không nhãn (DEV), occlusion theo dải tần, (tuỳ chọn) đường cong bán giám sát.

---

## 2. Phân tích và đánh giá hiện trạng

### 2.1 Những gì đã biết từ v6/v7 (tập `field_rec_8k_V7.2`)

| Đặc điểm | Giá trị | Hệ quả cho thiết kế ADA |
|---|---|---|
| Đơn vị độc lập | 80 bản ghi `Rxxx` (34 leak / 46 noleak), 167 tệp, 123.7 phút | Đơn vị thống kê phải là **bản ghi**, không phải frame |
| Tần số lấy mẫu gốc | 16 kHz | Giữ `target_sr = 16000` (v7 hạ về 8 kHz) |
| Dấu hiệu theo dải | Không dải nào có ý nghĩa sau Holm; dải 283–413 Hz AUC 0.36, 1.9–2.7 kHz AUC 0.62 | Dấu hiệu yếu, phân tán → dễ bị "dấu vân tay" của chuỗi thu lấn át → động cơ cho DA |
| Thông tin điểm khảo sát | Không có | Không loại trừ được confound vị trí; **miền** trong tập Mix là cơ hội đầu tiên đo trực tiếp tính khái quát |
| Nhiều tệp / bản ghi | 60/80 bản ghi cắt thành nhiều tệp | Gộp tệp theo `Rxxx`; trong tập Mix phải **ghép miền** vào mã (`miền|Rxxx`) |

### 2.2 Tập `tapdulieuMix_16k` — giả định và cách notebook tự kiểm chứng

Tập nằm trên máy cục bộ (`D:\OneDrive - ctu.edu.vn\NCS_NKNGUYEN\Nam2026\TapDulieu_2025\tapdulieuMix_16k`) nên kế hoạch
**không giả định cứng** cấu trúc. Notebook:

- tự dò thư mục gốc theo tên `tapdulieuMix_16k` trong `/kaggle/input` (khớp cả slug `tapdulieumix-16k`), in **cây thư mục
  kèm số tệp** (mục 2.2);
- nhận miền (mục 2.3), `cfg.domain_source="auto"` thử lần lượt và lấy cách **đầu tiên** cho ≥ 2 miền hợp lệ:

  | Cách | Bố cục dữ liệu | Miền nhận được |
  |---|---|---|
  | `folder` | `tapdulieuMix_16k/<miền>/<leak\|noleak>/...` hoặc `<leak\|noleak>/<miền>/...` | tên thư mục |
  | `label_folder` | `leak_16k/`, `noleak_8k/`, `leak_tt/`... (nhãn và miền chung một tên thư mục) | `16k`, `8k`, `tt` |
  | `filename` | `leak/ptn_R001_leak01.wav`, `leak/R001_tt_leak01.wav` | `ptn`, `tt` |
  | `bandwidth` | thư mục phẳng, trộn dữ liệu thu 8 kHz (nâng lên 16 kHz) và thu gốc 16 kHz | `chain_8k`, `chain_16k` |

  Khai báo tay: `cfg.domain_roots` (**mỗi dataset một miền**, ví dụ tập Mix + tập hiện trường `field_rec_8k_V7.2`),
  `"regex"` (`cfg.domain_regex`), `"csv"` (cột `domain` + một trong `rel`/`file`/`file_stem`/`record_id`);
  gộp/đổi tên miền bằng `cfg.domain_alias`;
- nếu vẫn chỉ có một miền: **in chẩn đoán** (ví dụ đường dẫn, ký hiệu còn lại trong tên tệp, biểu đồ chuỗi thu,
  danh sách dataset trong `/kaggle/input` có `leak/`+`noleak/`), **ghi `domain_map_template.csv`** để gán miền bằng tay,
  rồi dừng với 4 cách khắc phục cụ thể.

### 2.3 Nguồn gây dịch chuyển miền có thể có trong bài toán này

| Nguồn | Biểu hiện trong tín hiệu | Cách notebook xử lý / đo |
|---|---|---|
| Chuỗi thu khác (8 kHz nâng lên 16 kHz vs 16 kHz gốc) | Không có năng lượng > 4 kHz ở một miền | **Hoà hợp băng thông** tự động (mục 3.2) — loại dấu hiệu miền tầm thường |
| Cảm biến / khớp nối khác | Độ dốc phổ (spectral tilt) khác | Augmentation FIR ngẫu nhiên + căn chỉnh đặc trưng |
| Điểm khảo sát, nhiễu nền (xe cộ, bơm, điện lưới) | Nền phổ, xung nhiễu khác | Đo bằng PSD theo miền × lớp, PAD |
| Điều kiện thủy lực (áp suất, đường kính, kích thước lỗ rò) | Dải tần mang dấu hiệu dịch chuyển | Bảng AUC theo dải **trong từng miền** + kiểm tra **đổi chiều** |
| Tỉ lệ lớp khác nhau | Dịch chuyển nhãn | Bảng tỉ lệ leak theo miền; lấy mẫu nguồn cân bằng lớp; CDAN+E |

### 2.4 Rủi ro khoa học và đánh giá khả thi

| Rủi ro | Vì sao quan trọng | Biện pháp trong kế hoạch |
|---|---|---|
| **Dịch chuyển nhãn** | DANN căn chỉnh P(f) biên → có thể *làm hại* khi P(y) khác (Zhao et al., 2019) | Lấy mẫu nguồn cân bằng; CDAN+E; ablation `no_balance`; PAD theo lớp |
| **Dịch chuyển có điều kiện** | Nếu dấu hiệu đổi chiều giữa miền, không UDA nào đảm bảo cải thiện | Mục 3.5 phát hiện dải "FLIP"; nêu trong Discussion |
| **Trùng nội dung nguồn ↔ đích** | Bản ghi đích có nhãn lọt vào nguồn → thổi phồng kết quả | Mục 3.6 so khớp đường bao 100 ms (r ≥ 0.97) |
| **Chọn mô hình bằng nhãn đích** | Lỗi phổ biến nhất trong tài liệu UDA | Quy tắc chính `last` (đăng ký trước); `oracle` chỉ để đo mức lạc quan |
| **Cỡ mẫu nhỏ** | CI rộng, p-value không ổn định | Bootstrap theo bản ghi, 3 seed, trình bày CI thay vì chỉ trung bình |
| **Rò rỉ transductive** | Dùng chính bản ghi kiểm thử (không nhãn) để căn chỉnh | Giao thức **quy nạp** K-fold trên bản ghi đích (mặc định) |

**Điều kiện tối thiểu để chạy**: ≥ 2 miền; miền đích có ≥ 2 bản ghi mỗi lớp (`cfg.min_records_per_class`); số fold đích
= min(5, số bản ghi của lớp ít nhất). Khuyến nghị ≥ 10 bản ghi mỗi lớp ở miền đích để CI có ý nghĩa.

---

## 3. Định nghĩa bài toán

- Miền nguồn $\mathcal{D}_S=\{(x_i^s, y_i^s)\}$ **có nhãn**; miền đích $\mathcal{D}_T=\{x_j^t\}$ **không nhãn** khi huấn luyện.
- Đầu vào: frame 2 s (bước 1 s) @ 16 kHz; nhãn ở mức bản ghi; quyết định cuối ở mức bản ghi (trung bình xác suất frame).
- **Bài toán chuyển miền** (tự động): 2 miền → A→B và B→A; > 2 miền → leave-one-domain-out (các miền còn lại → miền giữ lại);
  hoặc khai báo tay `cfg.tasks = (("lab", "field"), ("lab+X", "Y"))`.
- Kịch bản: **UDA** (chính); **SSDA** với 1/2/4 bản ghi đích có nhãn mỗi lớp (tuỳ chọn, mục 14).

---

## 4. Thiết kế phương pháp

### 4.1 Kiến trúc dùng chung

| Khối | Cấu trúc | Ghi chú |
|---|---|---|
| Front-end | log-Mel 64 dải, n_fft 2048 (128 ms), hop 512 (32 ms), 20 Hz – f_max chung | Trong mô hình, chạy GPU, fp32; SpecAugment trước chuẩn hoá; chuẩn hoá từng mẫu (bất biến gain) |
| $G_f$ | 4 × [Conv3×3-BN-ReLU ×2, SE, MaxPool, Dropout2d 0.1], 32/64/128/256 kênh, GAP⊕GMP → 512 | Giống `LeakCNN2D` v7 → so sánh được với v7 |
| Bottleneck | FC 512→256, BN, ReLU, Dropout 0.3 → **f** | Không gian căn chỉnh miền |
| $G_y$ | FC 256→1, sigmoid, BCE + label smoothing 0.05 | |
| $G_d$ (DANN) | GRL → FC256-ReLU-Drop0.5 → FC256-ReLU-Drop0.5 → FC1 | Theo Ganin et al. (thu nhỏ cho dữ liệu nhỏ) |
| $G_d$ (CDAN) | GRL → f ⊗ g (512) → cùng MLP | g = [1−p, p], tách gradient |

### 4.2 Hàm mục tiêu

- **Source-only**: $\mathcal{L}_y$ trên lô nguồn cân bằng lớp.
- **DANN**: $\mathcal{L}_y + \mathcal{L}_d$, GRL nhân gradient với $-\lambda_p$,
  $\lambda_p=\lambda_{\max}\big(\tfrac{2}{1+e^{-10p}}-1\big)$, $\lambda_{\max}=1$.
- **CDAN+E**: $\mathcal{L}_d$ trên $f\otimes g$, mỗi mẫu trọng số $1+e^{-H(g)}$ chuẩn hoá trong từng miền.
- **Deep CORAL**: $\mathcal{L}_y + \lambda_p\,\lVert C_s - C_t\rVert_F^2/4d^2$; **MK-MMD** (tuỳ chọn) tương tự.
- **AdaBN**: không huấn luyện thêm; ước lượng lại thống kê BatchNorm của mô hình source-only trên đích không nhãn.
- **Oracle**: $\mathcal{L}_y$ trên nguồn + đích có nhãn (cùng fold) — cận trên, không phải UDA.

### 4.3 Lý do lựa chọn

- DANN là phương pháp đối nghịch chuẩn, có lý thuyết (cận Ben-David) → **phương pháp chính**.
- CDAN+E là biến thể đối nghịch giải quyết đúng điểm yếu của DANN trong bài toán này (dịch chuyển nhãn/có điều kiện).
- AdaBN, CORAL: baseline **không đối nghịch**, rẻ → trả lời "có cần học đối nghịch không?".
- Cùng $G_f$/$G_y$/lịch/augmentation cho mọi phương pháp → khác biệt chỉ đến từ mục tiêu căn chỉnh.

---

## 5. Xử lý dữ liệu

1. **Kiểm kê** mức tệp → mức bản ghi (`record_uid = miền|Rxxx`), dừng khi một bản ghi mang hai nhãn.
2. **Một lượt đọc**: khử DC → high-pass 20 Hz → notch 50 Hz × 6 hài → chuẩn hoá đỉnh theo tệp → cắt frame 2 s/1 s;
   cùng lượt tính PSD Welch (nperseg 4096), băng thông 99.5% năng lượng, tỉ lệ năng lượng > 4 kHz, đường bao 100 ms.
   Cache theo dấu vân tay nội dung + tham số.
3. **Hoà hợp băng thông** (`cfg.bandwidth_harmonize="auto"`): nếu p90 băng thông của miền hẹp nhất < 0.85·Nyquist và
   khác miền khác > 15% → giới hạn `f_max` của mọi front-end về băng thông chung.
4. **Phát hiện trùng nội dung** giữa các bản ghi khác nhau (đặc biệt giữa nguồn và đích).
5. **Augmentation** (cả nguồn và đích, trên GPU): FIR ngẫu nhiên p=0.5 (bất biến cảm biến), nhiễu Gauss SNR 10–30 dB p=0.5,
   dịch vòng p=0.5, SpecAugment 2+2 mặt nạ. Không dùng gain (bị chuẩn hoá triệt tiêu — v7).
6. **Lô huấn luyện**: 64 frame nguồn (32 leak + 32 noleak) + 64 frame đích không nhãn, đi chung một lượt truyền
   (BatchNorm thấy cả hai miền).

---

## 6. Giao thức đánh giá

| Thành phần | Thiết kế | Lý do |
|---|---|---|
| Chia nguồn | 80% `src_train` / 20% `src_val`, phân tầng theo miền × lớp, theo **bản ghi** | Chọn epoch/ngưỡng/DEV không chạm đích |
| Chia đích | **Quy nạp** StratifiedKFold K=5 trên **bản ghi** đích | Mỗi bản ghi đích được kiểm thử đúng một lần bởi mô hình chưa từng thấy nó (kể cả không nhãn) |
| Lặp | 3 seed (phân hoạch + khởi tạo mới), OOF trung bình qua seed | Độ bất định do phân hoạch; ước lượng ổn định |
| Chọn checkpoint | `last` (đăng ký trước), chỉ từ 30% lịch; độ nhạy: `src_val`, `dev`; tham chiếu `oracle` | Không rò nhãn đích |
| Chỉ số chính | **AUC mức bản ghi** (`p_mean`) trên miền đích | Đơn vị kỹ sư vận hành cần |
| Chỉ số phụ | BAcc, F1, MCC, Sens, Spec @0.5; AUC frame; AUC nguồn-val (giữ hiệu năng nguồn); *gap closed* | |
| Suy luận | CI 95% bootstrap bản ghi (B=2000); ΔAUC **ghép cặp** vs source-only; Holm trong từng bài toán; McNemar mức bản ghi | Đúng đơn vị, kiểm soát đa so sánh |
| Assertion cứng | 5 điều kiện (không giao 4 tập, phủ đích đúng một lần, đủ hai lớp) | Notebook dừng nếu vi phạm |

---

## 7. Tham số (đăng ký trước)

| Nhóm | Tham số | Giá trị | Căn cứ |
|---|---|---|---|
| Tín hiệu | sr / frame / hop | 16 kHz / 2 s / 1 s | Giữ nguyên v7, dùng sr gốc |
| Front-end | n_mels / n_fft / hop | 64 / 2048 / 512 | Cùng độ phân giải thời gian–tần số với v7 (1024 @ 8 kHz) |
| Mạng | bottleneck / G_d | 256 / MLP 256-256-1, dropout 0.5 | CDAN/DANN chuẩn, thu nhỏ cho dữ liệu nhỏ |
| Đối nghịch | λ_max / γ | 1.0 / 10 | Ganin et al. 2016 |
| Tối ưu | AdamW lr / wd | 3e-4 / 1e-4 | Như v7 |
| Lịch | epoch × vòng lặp | 30 × 80 = 2400 vòng | Mỗi job ~1–1.5 phút trên T4 |
| Lô | nguồn + đích | 64 + 64 | |
| Ổn định | warm-up / clip / AMP | 5% + cosine / 5.0 / fp16 | Như v7 |
| Đánh giá | eval_every / chọn từ | 2 epoch / epoch 9 | λ_p ≈ 0.9 λ_max ở 30% lịch |
| Thống kê | bootstrap / ngưỡng | 2000 / 0.5 | |

---

## 8. Kết quả, hình và bảng được sinh tự động (ánh xạ vào bài báo)

### 8.1 Bảng (CSV + LaTeX booktabs + Markdown trong `tables/`)

| Bảng (tên tệp) | Nội dung | Vị trí trong bài báo |
|---|---|---|
| `inventory_domain_class` | Bản ghi/tệp/phút theo miền × lớp | Dữ liệu |
| `bandwidth_by_domain` | Băng thông hiệu dụng, năng lượng > 4 kHz | Dữ liệu / Tiền xử lý |
| `band_auc_within_domain` | AUC theo dải trong từng miền + chiều dấu hiệu | Phân tích dịch chuyển miền |
| `duplicate_pairs` *(nếu có)* | Cặp nghi trùng nội dung | Dữ liệu (kiểm soát chất lượng) |
| `pad_input`, `cross_domain_matrix_classical` | PAD đầu vào; AUC huấn luyện A → kiểm thử B | **Động cơ** cho DA |
| `architecture_parameters` | Thành phần mạng và số tham số | Phương pháp |
| `transfer_tasks`, `partitions` | Bài toán, phân hoạch | Thiết lập thí nghiệm |
| `training_cost` | Thời gian mỗi phương pháp | Thiết lập thí nghiệm |
| **`main_results`** | AUC [CI], AUC seeds, BAcc, F1, MCC, ΔAUC [CI], p Holm, gap closed, AUC nguồn-val | **Kết quả chính** |
| `mcnemar_vs_source_only` | McNemar mức bản ghi | Kết quả |
| `feature_discrepancy` | PAD tổng/theo lớp, MMD, silhouette lớp đích | Phân tích cơ chế |
| `selection_rules`, `selection_criterion_validity` | AUC theo quy tắc chọn; tương quan tiêu chí không nhãn | Thảo luận phương pháp luận |
| `lambda_sweep`, `ablation` | Độ nhạy λ; ablation | Phân tích độ nhạy |
| `ssda_curve` *(tuỳ chọn)* | AUC theo số bản ghi đích có nhãn | Thảo luận ứng dụng |
| `band_occlusion` | Mức giảm AUC khi che từng nhóm dải | Diễn giải |
| `hyperparameters` | Toàn bộ thiết lập | Phụ lục / Phương pháp |

### 8.2 Hình (PNG 300 dpi + PDF vector trong `figures/`, chú thích trong `figure_index.csv`)

| Hình | Nội dung |
|---|---|
| `domain_inventory` | Bản ghi theo miền × lớp, thời lượng, tỉ lệ leak (dịch chuyển nhãn) |
| `bandwidth_by_domain` | Băng thông từng bản ghi, năng lượng cao tần |
| `psd_domain_class` | PSD tương đối theo miền × lớp (± CI 95%) và chồng các miền |
| `band_auc_by_domain` | Heatmap AUC theo dải × miền; profile độ dốc phổ |
| `input_domain_shift` | Ma trận AUC chéo miền; PCA; t-SNE đặc trưng đầu vào |
| `spectrogram_examples` | log-Mel đại diện từng miền × lớp |
| `frame_distribution` | Frame theo miền × lớp, mức RMS |
| `architecture_dann` | **Sơ đồ kiến trúc DANN/CDAN** |
| `partition_map` | Bản đồ giao thức quy nạp |
| `schedules` | Lịch λ_p và learning rate |
| **`target_auc_forest`** | **AUC đích [CI] + ΔAUC vs source-only (forest plot)** |
| `auc_heatmap_tasks` | AUC phương pháp × bài toán |
| `roc_target`, `confusion_target` | ROC và ma trận nhầm lẫn mức bản ghi |
| `adversarial_dynamics` | L_y, L_d, độ chính xác G_d (→0.5), λ_p theo epoch |
| `epoch_curves` | AUC đích [giám sát], PAD, AUC nguồn-val theo epoch |
| **`tsne_adaptation`** | **t-SNE trước/sau thích ứng (theo miền, theo lớp)** |
| `pad_bars` | PAD tổng và theo lớp |
| `model_selection` | AUC theo quy tắc chọn; DEV vs AUC đích |
| `lambda_sensitivity`, `ablation` | Độ nhạy λ, ablation |
| `ssda_curve` *(tuỳ chọn)* | Đường cong bán giám sát |
| `band_occlusion` | Độ nhạy theo dải tần trên miền đích |

### 8.3 Văn bản

- `manuscript_draft.md`: đoạn **Methods** và **Results** tiếng Anh sinh từ số liệu thật (chỉ viết "significantly improved"
  khi CI ΔAUC loại 0 **và** p Holm < 0.05) + tóm tắt tiếng Việt.
- `ada_summary.json`: cấu hình, kết quả chính, phiên bản phần mềm, GPU, thời gian chạy.
- `ada_paper_artifacts.zip`: toàn bộ hình/bảng/văn bản để tải về một lần.

---

## 9. Cấu trúc notebook

| Mục | Nội dung |
|---|---|
| 0 | Giới thiệu, câu hỏi nghiên cứu, cơ sở lý thuyết (cận Ben-David, DANN, CDAN, giới hạn) |
| 1 | Imports, **cấu hình trung tâm** (`CFG`), tái lập, xuất hình/bảng |
| 2 | Dò dữ liệu, dữ liệu mô phỏng hai miền (khi không có dữ liệu), kiểm kê tệp/bản ghi/miền, kiểm tra định danh |
| 3 | **EDA dịch chuyển miền**: một lượt đọc + cache, băng thông & hoà hợp, PSD, AUC theo dải trong từng miền, trùng lặp, PAD & ma trận chéo miền, spectrogram |
| 4 | Phân bố frame |
| 5 | Front-end, $G_f$/bottleneck/$G_y$, GRL, $G_d$, CDAN, CORAL, MMD + **kiểm thử đơn vị**, bảng tham số, sơ đồ kiến trúc |
| 6 | Bài toán chuyển miền, phân hoạch quy nạp + 5 assertion, bản đồ phân hoạch |
| 7 | Dữ liệu thường trú GPU, bộ lấy mẫu, augmentation, lịch λ/LR, DEV/PAD, **`train_job`**, hàng đợi 2 GPU, AdaBN, lưu/khôi phục |
| 8 | Thí nghiệm chính |
| 9 | Bảng kết quả chính, forest plot, ROC, nhầm lẫn, McNemar |
| 10 | Động học đối nghịch, đường cong theo epoch |
| 11 | Không gian đặc trưng: PAD/MMD/silhouette, t-SNE, PAD theo lớp |
| 12 | Chọn mô hình không nhãn đích |
| 13 | Quét λ, ablation |
| 14 | (Tuỳ chọn) bán giám sát |
| 15 | Occlusion theo dải tần |
| 16 | Bảng siêu tham số, văn bản nháp, JSON, zip |
| 17 | Diễn giải, hạn chế, danh mục kiểm tra trước khi nộp |

---

## 10. Ngân sách tính toán (Kaggle, T4 ×2)

Mỗi "job" = một lượt huấn luyện (bài toán × seed × fold × phương pháp), 2400 vòng lặp, ước tính **1–1.5 phút/job** trên T4
(mạng ~1.3 M tham số, đầu vào 64×63).

| Khối | Số job (2 miền, K=5, 3 seed, mặc định) |
|---|---|
| Thí nghiệm chính | 2 bài toán × 3 seed × (1 source-only + 5 fold × 4 phương pháp) = **126** |
| Quét λ (3 giá trị, 1 seed) | 15 |
| Ablation (4 biến thể, 1 seed) | 20 |
| SSDA (tuỳ chọn, 3 mức × 2 phương pháp) | 30 |
| **Tổng mặc định** | **≈ 161 job → ~1.5–2.5 giờ** trên 2 GPU (+ ~5–15 phút EDA) |

Rút gọn khi cần: `cfg.seeds=(42,)` (÷3), `cfg.tgt_folds=3`, `cfg.methods` bỏ `coral`/`oracle`,
`cfg.run_ablation=False`. Kết quả đã chạy được **khôi phục tự động** (`ada_results.pkl`) khi chạy lại trong cùng phiên với
cùng cấu hình. Bảng `training_cost` cho thời gian thực đo để hiệu chỉnh ước tính.

---

## 11. Hướng dẫn chạy trên Kaggle

1. Nén thư mục `tapdulieuMix_16k` → `Kaggle → Datasets → New Dataset` (giữ nguyên cấu trúc thư mục con).
2. Mở notebook `leak-ada-dann-kaggle-v1.ipynb` trên Kaggle (`File → Import Notebook`), `Add Data` → dataset vừa tạo.
3. `Settings → Accelerator = GPU T4 x2` (không cần Internet).
4. Chạy **đến hết mục 2.4** trước; kiểm tra:
   - cây thư mục và danh sách miền nhận ra (mục 2.2–2.3);
   - bảng kiểm kê miền × lớp (mục 2.4) — mỗi miền đích cần ≥ 2 bản ghi mỗi lớp.
   Nếu miền nhận sai: sửa `cfg.domain_source`, `cfg.domain_regex` / `cfg.domain_csv`, `cfg.domain_alias`,
   `cfg.exclude_domains` hoặc khai báo tay `cfg.tasks`.
   **Nếu tập Mix chỉ có một miền** (notebook dừng ở mục 2.3 với chẩn đoán): `Add Data` thêm dataset thứ hai (ví dụ
   `tapdulieutt-8k-v7-2`) rồi đặt
   `cfg.domain_roots = {"mix": "tapdulieuMix_16k", "field": "field_rec_8k_V7.2"}` (tên thư mục là đủ; notebook tự tìm
   trong `/kaggle/input`). Khi hai dataset có chung bản ghi (cùng một lần thu ở 8 kHz và 16 kHz), mục 3.6 sẽ báo trùng —
   phải loại trước khi báo cáo.
5. `Run All` (hoặc `Save Version → Save & Run All` để chạy nền). Tải `/kaggle/working/ada_paper_artifacts.zip`.
6. Chạy thử ngoài Kaggle (CPU, dữ liệu mô phỏng):
   `LEAK_OUT=/tmp/ada_smoke python tools/run_smoke.py notebooks/leak-ada-dann-kaggle-v1.ipynb`.

---

## 12. Kế hoạch viết bài báo

**Tên đề xuất**: *Adversarial Domain Adaptation for Acoustic Leak Detection in Low-Pressure PVC Water Pipes across
Acquisition Domains: A Leakage-Free, Recording-Level Evaluation*.

| Phần | Nội dung | Tư liệu từ notebook |
|---|---|---|
| 1. Introduction | Rò rỉ ống PVC áp thấp; khoảng trống: khái quát chéo miền, đánh giá thiếu chặt chẽ | — |
| 2. Related work | Phát hiện rò rỉ bằng học sâu (bảng đối chứng v7); UDA (DANN, CDAN, CORAL, AdaBN); chọn mô hình trong UDA | — |
| 3. Data & domain shift | Mô tả miền; băng thông; PSD; dấu hiệu theo dải; PAD đầu vào; ma trận chéo miền | Bảng 1–6; Hình 1–7 |
| 4. Methods | Tiền xử lý, kiến trúc, DANN/CDAN+E, baseline, lịch λ | Bảng kiến trúc, siêu tham số; Hình kiến trúc, lịch |
| 5. Experimental protocol | Quy nạp K-fold, chọn mô hình không nhãn, thống kê | Bảng phân hoạch; Hình bản đồ phân hoạch |
| 6. Results | Bảng chính; forest plot; ROC; McNemar | Bảng `main_results`; Hình forest/ROC/nhầm lẫn |
| 7. Analysis | Động học; PAD/t-SNE; chọn mô hình; λ; ablation; occlusion | Hình 10–15, các bảng tương ứng |
| 8. Discussion | Khi nào DA giúp/không giúp (dịch chuyển nhãn/có điều kiện); mức lạc quan khi chọn bằng nhãn đích; ý nghĩa triển khai | Bảng chọn mô hình, SSDA |
| 9. Conclusion & limitations | Mục 17 của notebook | — |

**Tạp chí phù hợp (tham khảo)**: *Applied Acoustics*, *Measurement*, *Mechanical Systems and Signal Processing*,
*Sensors* (MDPI), *Water* (MDPI), *Journal of Water Resources Planning and Management* (ASCE).

---

## 13. Lộ trình

| Tuần | Công việc | Đầu ra |
|---|---|---|
| 1 | Tải tập Mix lên Kaggle; chạy mục 1–4; xác nhận miền, băng thông, trùng lặp, bảng dải theo miền | Bảng/hình dữ liệu; quyết định danh sách bài toán |
| 1 | Chạy smoke trên Kaggle (`LEAK_SMOKE=1`) để kiểm tra thời gian | Ước tính thời gian thực |
| 2 | Chạy cấu hình đầy đủ (Save & Run All) | `ada_paper_artifacts.zip` |
| 2 | Đọc kết quả theo danh mục mục 17; nếu có miền metadata chi tiết → chạy thêm với `domain_source="csv"` | Ghi chú phân tích |
| 3 | (Tuỳ chọn) bật `run_ssda`; thử backbone `cnn1d` làm phân tích độ bền | Hình bổ sung |
| 3–4 | Viết bản thảo theo khung mục 12 dùng `manuscript_draft.md` | Bản thảo v1 |

---

## 14. Tài liệu tham khảo chính

1. Ben-David S. et al. A theory of learning from different domains. *Machine Learning* 79 (2010) 151–175.
2. Ganin Y. et al. Domain-adversarial training of neural networks. *JMLR* 17 (2016) 1–35.
3. Long M. et al. Conditional adversarial domain adaptation. *NeurIPS* (2018).
4. Sun B., Saenko K. Deep CORAL: correlation alignment for deep domain adaptation. *ECCV Workshops* (2016).
5. Long M. et al. Learning transferable features with deep adaptation networks. *ICML* (2015).
6. Li Y. et al. Adaptive batch normalization for practical domain adaptation. *Pattern Recognition* 80 (2018) 109–117.
7. Zhao H. et al. On learning invariant representations for domain adaptation. *ICML* (2019).
8. You K. et al. Towards accurate model selection in deep unsupervised domain adaptation. *ICML* (2019).
9. Park D. S. et al. SpecAugment. *Interspeech* (2019).
10. Các công bố về phát hiện rò rỉ bằng học sâu: xem bảng đối chứng ở mục 0 của notebook v7.

---

## 15. Trạng thái kiểm chứng của notebook

| Đã kiểm tra | Kết quả |
|---|---|
| Chạy end-to-end chế độ smoke (CPU, dữ liệu mô phỏng hai miền `lab`/`field`, `LEAK_SMOKE=1`) | 42/42 cell code chạy hết (~5.5 phút), sinh 23 hình + 19 bảng + văn bản nháp |
| Chạy kiểm tra dài hơn (400 vòng lặp/job, 48 bản ghi mô phỏng) cho source-only, DANN, CDAN+E, CORAL, oracle | Mất mát phân lớp hội tụ; bộ phân biệt miền của DANN/CDAN+E dao động quanh mức ngẫu nhiên (L_d ≈ ln 2, độ chính xác 0.35–0.55) → GRL hoạt động; AUC frame đích của source-only giảm dần (0.98 → 0.69) trong khi DANN (0.93), CORAL (0.97) giữ được |
| Nhận diện miền trên 7 cấu hình mô phỏng | `folder`, `label_folder`, `filename`, `bandwidth`, `domain_roots`, `csv`, `regex` đều nhận đúng 2 miền; tập một miền dừng với chẩn đoán + `domain_map_template.csv` |
| Kiểm thử đơn vị trong notebook | GRL đảo dấu gradient đúng hệ số; CORAL/MMD = 0 với hai phân bố trùng nhau; kích thước ánh xạ CDAN; trọng số entropy chuẩn hoá đúng |
| 5 assertion phân hoạch | Đạt cho mọi bài toán × seed |

**Chưa kiểm tra được ở đây (cần lần chạy đầu trên Kaggle):** đường dữ liệu thường trú GPU và chạy song song 2 GPU (môi trường kiểm thử không có GPU),
cấu trúc thư mục thật của `tapdulieuMix_16k`, thời gian chạy thực tế. Số liệu từ dữ liệu mô phỏng **không** dùng cho bài báo
(notebook in cảnh báo `SYNTHETIC DEMO MODE` ở cuối).
