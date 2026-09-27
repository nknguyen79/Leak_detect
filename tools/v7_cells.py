# -*- coding: utf-8 -*-
"""Nội dung các cell được VIẾT LẠI TOÀN BỘ trong v7 (khoá = chỉ số cell trong v6)."""

MARKDOWN = {}
CODE = {}

# =============================================================================
# MARKDOWN
# =============================================================================
MARKDOWN[0] = r'''
# Đánh giá khả năng nhận dạng rò rỉ nước trên đường ống trên cơ sở phân tích âm thanh và ứng dụng mạng học sâu với đường ống PVC áp lực thấp — **v7**

**Acoustic Leak Identification in Low-Pressure PVC Water Distribution Pipes Using Deep Convolutional Neural Networks**

---

### Mục tiêu / Objective

Notebook này là **pipeline nghiên cứu hoàn chỉnh, chạy được trên Kaggle (GPU T4 x2)**, phục vụ trực tiếp việc viết bài báo khoa học:

1. **Đọc & kiểm kê dữ liệu** hiện trường (`leak/` vs `noleak/`, ống PVC, áp suất 1–2 bar), trích mã bản ghi `Rxxx` và gộp mọi tệp cùng bản ghi.
2. **Phân tích tín hiệu (EDA)** ở **mức bản ghi**: PSD Welch, năng lượng theo dải, AUC đơn biến theo dải — mọi họ kiểm định đều **hiệu chỉnh Holm**.
3. **Tiền xử lý**: khử DC, lọc thông cao 20 Hz, notch 50 Hz × 6 hài, tái lấy mẫu, chuẩn hoá đỉnh theo tệp, frame 2 s chồng 50%.
4. **5 kiến trúc**: `LeakCNN2D` (log-Mel), `LeakCNN1D` (sóng thô), `TransferCNN` (ResNet18 ImageNet), `MultiBranchCNN` (Mel + MFCC + STFT dải thấp), `CRNN` (CNN + BiGRU + attention).
5. **Giao thức chống rò rỉ dữ liệu**: CV phân tầng trên **bảng bản ghi** + assertion cứng; **lựa chọn lồng** (epoch, ngưỡng) chỉ trên validation-trong.
6. **CV lặp** (mặc định 3 phân hoạch khác nhau) để đo độ bất định do phân hoạch/khởi tạo, và **trung bình OOF qua các lần lặp** làm ước lượng ổn định nhất.
7. **Thống kê đúng đơn vị**: bootstrap theo cụm bản ghi, McNemar mức bản ghi + Holm, bootstrap **ghép cặp** cho ΔAUC.
8. **Baseline cổ điển** (đặc trưng phổ + thống kê log-Mel) so sánh **ghép cặp** với CNN; **ablation** cấu hình tín hiệu với lựa chọn trên validation-trong.
9. **Diễn giải**: Grad-CAM, occlusion theo dải, t-SNE tô theo bản ghi; **kiểm tra confound** điểm khảo sát.
10. **Xuất bảng LaTeX/Markdown, hình 300 dpi, gói mô hình TorchScript** kèm manifest và giới hạn đã đo.

### Điểm mới của v7 (chi tiết ở mục 16)

| Mã | Sửa đổi |
|---|---|
| **B1** | Grad-CAM của CRNN làm v6 **dừng giữa chừng** (cuDNN RNN backward ở chế độ eval) → mục 10–16 không chạy. Đã sửa. |
| **B2** | "Không chọn checkpoint trong warm-up" giờ **được thực thi** (v6 chỉ khai báo; LeakCNN1D bị chọn ở epoch ~0). |
| **B3** | Không còn tự hạ `low_band_hz` về 1000 Hz dựa trên một dải "mạnh nhất" không có ý nghĩa sau hiệu chỉnh đa so sánh. |
| **B5–B7** | `DEPLOY_MODEL` dùng trước khi định nghĩa; ghi đè `sel_df`; lọc sai tên baseline RMS. |
| **S1** | CV lặp nhiều phân hoạch (mặc định 3 seed). |
| **S2** | Quy tắc gộp mức bản ghi **đăng ký trước** (`p_mean`, xác suất thô); Platt và lựa chọn trên validation-trong chỉ là phân tích độ nhạy. |
| **S3** | Baseline mạnh hơn (thống kê log-Mel) + so sánh **ghép cặp** thay cho "CI có chồng lấn không". |
| **S4** | Hiệu chỉnh Holm cho mọi kiểm định theo dải; giả thuyết "dải thấp" được kiểm định bằng CI bootstrap của hiệu số. |
| **A1** | Ablation bật mặc định: bỏ FIR, loss cân bằng theo bản ghi, 16 kHz, frame 1 s/4 s, bỏ notch/high-pass. |
| **D1** | Gán nhãn theo **từ**, bỏ khoá `"rr"`; dò thư mục dữ liệu báo lỗi rõ khi `cfg.data_root` sai. |

> **Cách dùng nhanh / Quick start**
> 1. `Add Data` → chọn dataset chứa hai thư mục `leak/` và `noleak/`; đặt đúng `cfg.data_root` (mục 1.2).
> 1b. **Kiểm tra cell 2.3b**: nó xác nhận mã bản ghi `Rxxx` được trích đúng và báo mâu thuẫn nhãn. Nếu tên tệp không theo dạng `Rxxx`, chỉnh `cfg.record_regex`.
> 1c. **Nếu có thông tin điểm khảo sát**: điền `site_map_template.csv` (notebook tự sinh ở mục 9.6), đặt `cfg.site_source="csv"` và `cfg.run_loso_full=True`. Đây là thí nghiệm quan trọng nhất còn thiếu.
> 2. `Settings` → `Accelerator` = **GPU T4 x2**, bật **Internet** (tải trọng số ImageNet cho TransferCNN).
> 3. `Run All` (~45–60 phút với cấu hình mặc định). Không tìm thấy dataset → notebook **tự sinh dữ liệu mô phỏng** (SYNTHETIC DEMO MODE).
> 4. Chạy thử nhanh: đặt biến môi trường `LEAK_SMOKE=1` (3 epoch, 2 fold).
'''

MARKDOWN[20] = r'''
### 3.5b Quyết định dải tần dựa trên SỐ LIỆU, không dựa trên giả thuyết

Mục 3.5 đo sức phân biệt của từng dải **trước khi dùng mô hình**. Cell dưới đây dùng
chính số đo đó để:

1. **Kiểm định giả thuyết "năng lượng rò rỉ tập trung ở dải rất thấp (< 800 Hz)"** bằng
   **CI bootstrap theo bản ghi** của hiệu số *sức phân biệt trung bình dải cao − dải thấp*.
   Ba kết luận có thể có: *ủng hộ*, *không ủng hộ*, hoặc **chưa phân định được** (CI chứa 0).
   v6 so sánh hai con số điểm mà không có CI, nên đã phát biểu mạnh hơn bằng chứng.
2. **Không tự hạ** `cfg.low_band_hz`. v6 đã hạ 2800 → 1000 Hz chỉ dựa trên *một* dải "mạnh
   nhất" (283–413 Hz, p = 0,034, không còn ý nghĩa sau hiệu chỉnh Holm) và cắt mất dải
   1,3–2,7 kHz mà chính Grad-CAM của LeakCNN2D sử dụng. v7 giữ giá trị đăng ký trước;
   nếu bật `cfg.low_band_auto`, dải chỉ được **mở rộng** để bao top-3 dải mang thông tin.
'''

MARKDOWN[22] = r'''
## 4. Tiền xử lý và cắt frame / Pre-processing and framing

Mỗi tệp được cắt thành các frame `cfg.frame_sec` giây với bước trượt `cfg.hop_sec` (mặc định 2 s, chồng lấn 50%).

> **Điểm mấu chốt về tính chính xác khoa học:** mọi frame đều mang theo `record_id` của bản
> ghi mẹ. Chỉ số này được dùng làm **group** trong cross-validation. Nếu chia ngẫu nhiên ở
> mức frame, các frame chồng lấn của cùng một bản ghi sẽ xuất hiện ở cả train lẫn validation
> → **rò rỉ dữ liệu** và chỉ số bị thổi phồng (mục 9.3 đo trực tiếp mức thổi phồng này).
>
> **Hệ quả thứ hai:** số frame lớn KHÔNG làm tăng cỡ mẫu hiệu dụng. Với ICC cao và khoảng
> một trăm frame mỗi bản ghi, hệ số phóng đại phương sai (design effect = 1 + (m − 1)·ICC)
> lên tới hàng chục, nên **cỡ mẫu hiệu dụng chỉ cỡ vài lần số bản ghi**. Mục 8.2 in ra ICC,
> design effect và cỡ mẫu hiệu dụng **đo được**; mọi khoảng tin cậy mức frame đều bootstrap
> theo cụm bản ghi (mục 7.2b).
'''

MARKDOWN[25] = r'''
## 5. Biểu diễn đặc trưng và kiến trúc mạng / Feature front-ends and CNN architectures

Thiết kế then chốt: **biểu diễn thời gian–tần số được đặt bên trong mô hình** và tính trên GPU.
Nhờ đó (i) mọi mô hình dùng chung một nguồn dữ liệu trả về dạng sóng thô, (ii) SpecAugment thực
hiện trực tiếp trên GPU, và (iii) Grad-CAM truy vết ngược được về đúng ảnh log-Mel mà mô hình nhìn thấy.

| Mô hình | Đầu vào | Ý tưởng | Tham chiếu tương đương |
|---|---|---|---|
| `LeakCNN2D` | log-Mel 64 dải | CNN 4 khối + Squeeze-Excitation, pooling mean+max | Ma et al. 2024; Zhang et al. 2026 |
| `LeakCNN1D` | dạng sóng (chuẩn hoá theo mẫu) | Bộ lọc học được trên miền thời gian (kernel rộng 80) | Shukla & Piratla 2020 |
| `TransferCNN` | log-Mel → 3 kênh, phóng ≥ 128×128 | Fine-tune ResNet18 (ImageNet) | Li et al. 2025 |
| `MultiBranchCNN` | log-Mel + MFCC + STFT 0–`low_band_hz` | Hợp nhất đa biểu diễn | Zhang et al. 2026 (multi-scale) |
| `CRNN` | log-Mel → BiGRU + attention | Mô hình hoá tính dừng theo thời gian | Cody et al. 2020 |

**Các nguyên tắc ở tầng đặc trưng:**

1. **Chuẩn hoá theo TỪNG MẪU** ở mọi mô hình (v7: cả `LeakCNN1D`) → không mô hình nào dùng được
   "độ to" tuyệt đối của tín hiệu, vốn khác nhau giữa hai lớp và có thể là confound.
2. **SpecAugment áp TRƯỚC chuẩn hoá**, giá trị che = mức im lặng của mẫu, **mặt nạ riêng cho
   từng mẫu** (v6 dùng một mặt nạ chung cho cả batch). `MultiBranchCNN`: mỗi nhánh có
   SpecAugment riêng (v6 *viết* là dùng chung cho 3 nhánh nhưng *mã* chỉ che nhánh Mel).
3. **MFCC chuẩn hoá theo từng hệ số (CMVN)**.
4. **Nhánh STFT dải thấp giữ 0–2800 Hz** (đăng ký trước; xem mục 3.5b).
'''

MARKDOWN[31] = r'''
## 6. Giao thức đánh giá chống rò rỉ dữ liệu / Leakage-free evaluation protocol

Notebook phân tầng **trên BẢNG BẢN GHI** rồi trải về frame (`cfg.fold_mode = "record_stratified"`):

- Mỗi `Rxxx` chỉ mang MỘT nhãn nên nhóm là "thuần nhãn"; phân tầng trực tiếp trên danh sách bản
  ghi giữ tỉ lệ lớp giữa các fold gần như **chính xác** (bảng 4 in biên độ dao động thực tế).
- `StratifiedGroupKFold` (`cfg.fold_mode = "frame_group"`) cân bằng theo **số frame**, nên khi số
  frame mỗi bản ghi khác nhau, tỉ lệ lớp giữa các fold có thể lệch đáng kể.

**4 assertion cứng** được kiểm tra cho **mọi** phân hoạch (kể cả các lần lặp CV) và notebook
**dừng ngay** nếu vi phạm:

| # | Điều kiện | Ý nghĩa |
|---|---|---|
| 1 | `set(groups[tr]) ∩ set(groups[va]) = ∅` | Không bản ghi nào nằm ở cả hai phía |
| 2 | Mỗi bản ghi xuất hiện **đúng một lần** ở validation | Không fold nào lặp/thiếu bản ghi |
| 3 | `len(tr) + len(va) = len(frames)` | Phủ hết frame |
| 4 | Validation có **đủ hai lớp** | Không fold nào chỉ toàn một lớp |

Số fold tự điều chỉnh theo số bản ghi của lớp hiếm nhất; quá ít bản ghi → `LeaveOneGroupOut`.

> **[V7] CV lặp:** `build_splits(seed)` sinh một phân hoạch mới cho mỗi seed trong `cfg.seeds`.
> Chia đúng theo bản ghi **chưa đủ**: vì số bản ghi độc lập nhỏ, mọi khoảng tin cậy mức frame
> **phải** bootstrap theo **cụm bản ghi**, nếu không CI hẹp hơn thực tế gần một bậc độ lớn.
'''

MD_REPEATED_CV = r'''
### 7.5 Huấn luyện toàn bộ mô hình + CV lặp / Training with repeated cross-validation

**[V7-S1]** Với vài chục bản ghi, *một* phân hoạch 5-fold cho kết quả phụ thuộc đáng kể vào
việc bản ghi nào rơi vào fold nào. v6 chỉ chạy một phân hoạch và một seed, nên mọi chênh lệch
AUC giữa các kiến trúc đều lẫn với nhiễu phân hoạch.

Cell dưới đây chạy **`len(cfg.seeds)` lần CV**, mỗi lần một **phân hoạch mới** và một khởi tạo mới:

- **Lần lặp đầu** (seed `cfg.seed`) là lần chạy chính: checkpoint, ngưỡng, diễn giải (mục 10), triển khai (mục 13–15).
- **Bảng 5b** báo cáo trung bình ± độ lệch chuẩn qua các lần lặp. Chênh lệch giữa hai kiến trúc
  nhỏ hơn độ lệch chuẩn này **không diễn giải được**.
- **Trung bình xác suất OOF qua các lần lặp** là ước lượng ổn định nhất và **vẫn trung thực**:
  mỗi dự đoán của một frame đến từ các mô hình **chưa từng thấy bản ghi chứa frame đó**.
'''

MARKDOWN[47] = r'''
## 8. Kết quả và trực quan hoá / Results and visualisation

### Vì sao đánh giá hai cấp, và cấp nào mới là chính

Nhãn của tập dữ liệu này được gán ở **mức bản ghi** (mã `Rxxx`), không phải mức frame.
Điều đó có bốn hệ quả mà bài báo phải nói rõ:

**1. Mức bản ghi là đơn vị thống kê duy nhất sạch.** Mỗi `Rxxx` là một quan sát độc lập.
Số frame lớn không làm tăng cỡ mẫu hiệu dụng.

**2. Mọi khoảng tin cậy ở mức frame PHẢI bootstrap theo cụm bản ghi.** Mục 8.2 in cả CI
theo cụm (đúng) và CI theo frame (sai) cùng **hệ số chênh lệch đo được** giữa hai cách.

**3. Nhãn mức frame chỉ là nhãn mức bản ghi được "kế thừa" xuống frame.** Mục 8.7 **đo**
tỉ lệ thể hiện dương thay vì giả định nó, và kết luận theo ba mức: ủng hộ / không ủng hộ /
chưa kết luận được tiền đề "nhãn frame bị nhiễu".

**4. Mọi so sánh giữa các kiến trúc phải làm ở mức BẢN GHI** (McNemar mức bản ghi,
bootstrap ghép cặp theo bản ghi). McNemar ở mức frame là pseudo-replication: nó coi hàng nghìn
quyết định phụ thuộc lẫn nhau là các "mẫu độc lập".

**Quy tắc gộp mức bản ghi [V7-S2]:** `cfg.record_agg = "p_mean"` trên xác suất **thô**, được
**đăng ký trước**. Bảng 7 vẫn in kết quả của cả bốn quy tắc và của bản hiệu chỉnh Platt để minh bạch.
'''

MARKDOWN[54] = r'''
### 8.7 Ước lượng tỉ lệ thể hiện dương / Witness-rate estimation

Câu hỏi quyết định **toàn bộ khung lập luận MIL**: trong một bản ghi được gán nhãn `leak`,
**bao nhiêu phần trăm frame thật sự mang dấu hiệu rò rỉ?**

Cell này **đo** đại lượng đó (trung vị, trên các bản ghi leak, của tỉ lệ frame vượt ngưỡng)
và **tự rút ra kết luận theo ba mức** [V7]:

| Trung vị | Kết luận | Hệ quả |
|---|---|---|
| < 0,5 | Tiền đề MIL **được ủng hộ** | Nhãn frame nhiễu; có thể thử `mil_mode="bag"` |
| 0,5 – 0,9 | **Chưa kết luận được** | Không được mô tả nhãn frame là "nhiễu nặng", cũng không được khẳng định ngược lại |
| ≥ 0,9 | Tiền đề MIL **không được ủng hộ** | Điểm số gần như hằng số trong bản ghi → bắt buộc xem mục 9.6 (confound) |

(v6 dùng một ngưỡng duy nhất 0,5 và in "trung vị 0,81 ≈ 1" — phát biểu mạnh hơn số liệu.)

**Cảnh báo về tính vòng quanh:** ước lượng này dựa trên chính mô hình, nên nó đo "phần frame
mà mô hình cho là có rò rỉ", không phải chân lý. Muốn có chân lý phải nghe và dán nhãn thời
điểm rò rỉ trên một mẫu bản ghi (xem mục 16).
'''

MARKDOWN[59] = r'''
## 9. Kiểm định thống kê / Statistical significance testing

Chỉ so sánh điểm trung bình là **chưa đủ** để tuyên bố một kiến trúc tốt hơn — nhưng dùng sai
đơn vị thống kê thì còn tệ hơn:

| Kiểm định | Lỗi thường gặp | Cách làm ở đây |
|---|---|---|
| **McNemar mức frame** | Hàng nghìn quyết định frame từ vài chục bản ghi → pseudo-replication; gần như **mọi cặp** đều `p ≈ 0`. | **McNemar ở MỨC BẢN GHI** là kiểm định chính (+ Holm). McNemar mức frame chỉ in để *mô tả*. |
| **Wilcoxon 5 fold** | Với k = 5, p nhỏ nhất **có thể** đạt là `2/2⁵ = 0,0625` → **không bao giờ** bác bỏ được ở 0,05. | In rõ **sàn p**; chỉ chạy khi số fold ≥ 8. |
| **CI cho AUC** | Bootstrap theo frame → hẹp giả tạo. | **Bootstrap theo cụm bản ghi** + **bootstrap ghép cặp theo bản ghi** cho ΔAUC. |

Ba kiểm định được dùng:

- **McNemar mức bản ghi** trên các quyết định ghép cặp — đúng đơn vị (n = số bản ghi).
- **Bootstrap ghép cặp theo cụm bản ghi** cho ΔAUC (A − B) — không phụ thuộc ngưỡng.
- **Wilcoxon signed-rank theo fold** — chỉ khi số fold đủ lớn; in kèm sàn p.

Độ lệch chuẩn giữa các lần lặp CV (Bảng 5b) là thước đo bổ sung: chênh lệch nhỏ hơn nó không diễn giải được.
'''

MARKDOWN[66] = r'''
### 9.5 BASELINE CỔ ĐIỂN — mạng CNN có thực sự thêm giá trị?

Đây là câu hỏi phản biện **đầu tiên**: *"Với vài chục bản ghi, một bộ phân loại cổ điển có làm
được gần bằng CNN không?"* Cell dưới đây trả lời bằng thực nghiệm, **cùng giao thức** (cùng fold
nhóm theo bản ghi, cùng bootstrap theo cụm bản ghi).

**Hai bộ đặc trưng** [V7-S3]:

- `spectral13`: năng lượng tương đối 7 dải + trọng tâm/độ trải/độ phẳng phổ + rolloff 85% + ZCR + RMS (như v6).
- `logmel_stats`: **trung bình và độ lệch chuẩn theo thời gian của 64 dải log-Mel** (128 chiều),
  đã trừ mức dB trung bình của frame → **bất biến với gain**, dùng *đúng* biểu diễn mà CNN nhìn thấy.
  Đây là đối thủ công bằng hơn nhiều so với 13 đặc trưng thủ công.

Ba bộ phân loại (LogReg, SVM-RBF, Gradient Boosting) × hai mức (frame → gộp `p_mean`, và
trực tiếp trên vector trung bình của bản ghi).

**So sánh GHÉP CẶP** [V7]: v6 kết luận "CNN vượt baseline" khi hai CI *riêng rẽ* không chồng lấn
(0,8092 vs 0,8076 — sát nhau tới 0,002). Cách đúng là **CI bootstrap ghép cặp của ΔAUC** trên
cùng tập bản ghi, hiệu chỉnh Holm theo số baseline. Mô hình CNN đại diện là `DEPLOY_MODEL` (chọn
trên validation-trong), không phải mô hình có AUC ngoài cao nhất — tránh thiên lệch có lợi cho CNN.

Ngoài ra in **baseline "chỉ dùng mức năng lượng"** (RMS trung bình bản ghi). Nếu riêng mức to/nhỏ
đã phân biệt tốt hai lớp, mọi kết luận về "học đặc trưng rò rỉ" phải được xem lại.
'''

MARKDOWN[68] = r'''
### 9.6 KIỂM TRA CONFOUND theo điểm khảo sát / site-level confounding

Các dấu hiệu trong chính kết quả của notebook có thể chỉ về cùng một khả năng: **mô hình phân
biệt điểm khảo sát (nơi thu âm) chứ không phân biệt hiện tượng rò rỉ**:

1. **ICC cao** (mục 8.8): phần lớn phương sai điểm số nằm *giữa các bản ghi*.
2. **Tỉ lệ thể hiện dương cao** (mục 8.7): điểm số gần như hằng số trong từng bản ghi.
3. **Mức to/nhỏ tín hiệu khác nhau giữa hai lớp** (mục 9.5).

Cell này **đo** confound ở mức có thể đo được, theo thứ tự mạnh → yếu:

| Phép kiểm tra | Cần gì | Sức mạnh |
|---|---|---|
| **Leave-one-site-out** (huấn luyện lại, bỏ hẳn 1 điểm khảo sát) | `cfg.site_source` | Mạnh nhất — trả lời trực tiếp câu hỏi |
| **AUC trong cùng một điểm** (chỉ dùng điểm có cả hai lớp) | `cfg.site_source` | Mạnh — loại bỏ khác biệt giữa các vị trí |
| **Baseline mức năng lượng / đặc trưng phổ** | không cần gì | Trung bình — phát hiện confound "chỉ học độ to" |
| **Phân bố điểm theo bản ghi** | không cần gì | Yếu — chỉ là dấu hiệu cảnh báo |

> Nếu **không** có thông tin điểm khảo sát, notebook in **cảnh báo rõ ràng** rằng confound chưa
> được loại trừ và **[V7] tự sinh tệp `site_map_template.csv`** để điền cột `site`.
'''

MARKDOWN[70] = r'''
### 9.7 ABLATION CẤU HÌNH TÍN HIỆU & HUẤN LUYỆN (bật mặc định) [V7-A1]

Huấn luyện lại **cùng mô hình (`cfg.ablation_model`, cố định trước), cùng phân hoạch fold, cùng
seed**, chỉ đổi đúng một yếu tố:

| Biến thể | Thay đổi | Câu hỏi |
|---|---|---|
| `no_fir` | tắt augmentation đáp ứng kênh ngẫu nhiên | FIR ngẫu nhiên có xoá mất đặc trưng *độ dốc phổ* — thứ phân biệt hai lớp? |
| `rec_balanced` | trọng số 1/n_frame cho mỗi bản ghi | bản ghi dài có đang chi phối huấn luyện? |
| `sr16k` | lấy mẫu 16 kHz (giữ dải 4–8 kHz), n_fft 2048 | dải > 4 kHz có mang thông tin? |
| `frame4s` / `frame1s` | độ dài frame 4 s / 1 s | vì sao chọn 2 s? |
| `no_notch` / `no_highpass` | bỏ notch 50 Hz / bỏ lọc thông cao | các bộ lọc có cần thiết? |

**Lựa chọn trung thực:** mỗi biến thể có AUC mức bản ghi trên **validation-trong** (dùng để
*chọn*) và trên fold ngoài (dùng để *báo cáo*), cùng **ΔAUC ghép cặp** so với cấu hình gốc. Biến
thể được đề xuất là biến thể có AUC validation-trong cao nhất. Muốn dùng nó làm cấu hình chính,
hãy đổi `cfg` rồi chạy lại toàn bộ notebook; khi đó con số báo cáo vẫn không thiên lệch.
'''

MARKDOWN[72] = r'''
## 10. Diễn giải mô hình / Model interpretability

Ba phân tích trả lời câu hỏi *"mô hình đã học đúng vật lý hay chỉ học nhiễu nền?"*:

1. **Grad-CAM** trên log-Mel: vùng thời gian–tần số nào chi phối quyết định.
2. **Occlusion theo dải tần**: che từng nhóm dải Mel **trước khi lên dB và trước chuẩn hoá**, đo
   mức sụt AUC ⇒ tầm quan trọng nhân quả của mỗi dải.
3. **t-SNE** trên embedding, **tô màu theo BẢN GHI** để phơi ra cấu trúc theo điểm khảo sát, kèm
   các chỉ số đo trong không gian gốc (không đọc khoảng cách trên bản đồ 2D).

**Nguyên tắc:** hình và caption KHÔNG được khẳng định kết luận trước khi đo — mọi nhận định được
**in ra từ số đo**, kèm cả trường hợp số đo phản bác giả thuyết. Grad-CAM/occlusion chạy trên
**fold cố định (fold 0)**, không chọn fold có AUC cao nhất.

**[V7-B1]** Ở v6, Grad-CAM của `CRNN` gây lỗi *"cudnn RNN backward can only be called in training
mode"* và **dừng toàn bộ notebook** — mục 10.2 → 16 chưa từng chạy. v7 tắt cuDNN cục bộ khi tính
gradient cho Grad-CAM, giữ mô hình ở chế độ eval.
'''

MARKDOWN[86] = r'''
## 12. Kết luận, hạn chế và hướng phát triển / Conclusions, limitations, future work

**Đóng góp của pipeline này (phần Discussion):**

1. **Chế độ vật lý chưa được khảo sát**: ống PVC ở 1–2 bar — suy giảm mạnh, SNR thấp. Đa số công
   bố WoS/Scopus dùng ống kim loại ở áp suất ≥ 5 bar nên kết quả không chuyển giao trực tiếp được.
2. **Giao thức đánh giá chống rò rỉ + chống lạc quan do lựa chọn**, kèm **đo** mức thổi phồng do
   rò rỉ dữ liệu và do lựa chọn epoch/ngưỡng/tiêu chí (mục 9.3, 9.4).
3. **Đơn vị thống kê đúng**: CI bootstrap theo cụm bản ghi, McNemar mức bản ghi, so sánh ghép cặp,
   hiệu chỉnh Holm, công khai sàn p của Wilcoxon, CV lặp.
4. **Đo thay vì giả định**: tỉ lệ thể hiện dương (8.7), ICC/design effect (8.2, 8.8), dải tần mang
   thông tin (3.5, 3.5b), tầm quan trọng nhân quả theo dải bằng occlusion (10.2).
5. **Kiểm tra confound** (9.6), **baseline cổ điển ghép cặp** (9.5) và **ablation** cấu hình (9.7).

**Hạn chế phải nêu trung thực** (số liệu cụ thể lấy từ output của chính lần chạy):

- **Số bản ghi hạn chế** → CI rộng; cỡ mẫu hiệu dụng (mục 8.2) chỉ cỡ vài lần số bản ghi. Cần
  **thêm điểm khảo sát**, không phải kéo dài thời lượng thu tại chỗ.
- **Thiếu thông tin điểm khảo sát** → **chưa loại trừ được confound** "mô hình học vị trí thu".
  Đây là hạn chế nghiêm trọng nhất.
- **Nhãn ở mức bản ghi**, không có nhãn thời điểm rò rỉ → không kiểm chứng trực tiếp được tiền đề MIL.
- **Một chuỗi thiết bị thu, một mạng ống** → hiệu lực ngoại suy thấp.
- **Tiền xử lý không nhân quả** (chuẩn hoá đỉnh theo toàn bộ tệp, quyết định ở mức bản ghi)
  → pipeline chạy offline, chưa phải thời gian thực.
'''

MARKDOWN[102] = r'''
---

## 16. CHANGELOG v6 → v7 và những gì CÒN PHẢI LÀM

Phần này liệt kê **chính xác** những gì đã sửa so với `leak-cnn-pvc-kaggle-v6.ipynb`, để người
đọc (và phản biện) truy vết được. Nhóm mã: **B** = lỗi mã, **S** = thống kê, **A** = ablation,
**C** = confound, **D** = dữ liệu. Toàn bộ v7 được sinh tái lập từ v6 bằng `tools/build_v7.py`.
'''

# =============================================================================
# CODE
# =============================================================================
CODE[7] = r'''
# =========================================================
# 2.1 Dò dataset / Auto-discover dataset root
#     [V6-P2] Chuẩn hoá Unicode + ưu tiên THƯ MỤC trước tên tệp khi gán nhãn.
#     [V7-D1] Khớp nhãn theo TỪ (ranh giới "_", ".", chữ số) thay vì chuỗi con; dò thư mục
#             dữ liệu chọn thư mục SÂU NHẤT chứa cả leak/ và noleak/, và báo rõ khi
#             cfg.data_root không tồn tại (v6 âm thầm rơi về quét cả /kaggle/input/datasets).
# =========================================================
def _norm_token(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.lower().replace("-", "_").replace(" ", "_")

def _key_regex(keys):
    ks = sorted({_norm_token(k) for k in keys}, key=len, reverse=True)
    # khoá phải đứng ở đầu từ; sau nó là hết chuỗi, "_", "." hoặc chữ số (leak01, leak_02)
    return re.compile(r"(?:^|[_.])(?:" + "|".join(map(re.escape, ks)) + r")(?=$|[_.\d])")

_NOLEAK_RE = _key_regex(cfg.noleak_keys)
_LEAK_RE = _key_regex(cfg.leak_keys)

def _label_of_token(s):
    s = _norm_token(s)
    if _NOLEAK_RE.search(s):        # kiểm tra 'noleak' TRƯỚC 'leak'
        return 0
    if _LEAK_RE.search(s):
        return 1
    return None

def _label_tokens(p: Path):
    # Thứ tự xét: thư mục chứa tệp (nếu ưu tiên thư mục) -> tên tệp -> các thư mục cha
    parts = [_norm_token(x) for x in p.parts]
    toks = list(reversed(parts))
    if cfg.label_source_priority.startswith("folder"):
        toks = [_norm_token(p.parent.name)] + toks
    return toks

def _label_from_path(p: Path):
    # Trả về (nhãn, token quyết định) để truy vết được nguồn gán nhãn.
    for s in _label_tokens(p):
        lab = _label_of_token(s)
        if lab is not None:
            return lab, s
    return None, None

# --- tự kiểm tra bộ gán nhãn (chạy mỗi lần, tốn vài micro giây) ---
for _t, _want in [("leak", 1), ("noleak", 0), ("no_leak", 0), ("No-Leak", 0), ("R001_leak01", 1),
                  ("R12_noleak_03", 0), ("rò_rỉ", 1), ("khong_ro_ri", 0), ("normal", 0),
                  ("current_error", None), ("field_rec_8k_V7.2", None), ("leakage", None)]:
    assert _label_of_token(_t) == _want, f"bộ gán nhãn sai với '{_t}': {_label_of_token(_t)} != {_want}"

def _dirs_with_both(base: Path, max_depth=6):
    # Mọi thư mục (tính cả base) có thư mục con mang nhãn leak VÀ thư mục con mang nhãn noleak.
    out = []
    if not base.exists():
        return out
    stack = [(base, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            kids = [k for k in d.iterdir() if k.is_dir()]
        except Exception:
            continue
        labs = {_label_of_token(k.name) for k in kids}
        if {0, 1}.issubset(labs):
            out.append(d)
        if depth < max_depth:
            stack += [(k, depth + 1) for k in kids]
    return sorted(out, key=lambda p: (-len(p.parts), str(p)))

def _audio_in(root: Path):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in cfg.audio_ext)

def discover_root(explicit: str = ""):
    if explicit:
        ex = Path(explicit)
        if ex.exists():
            files = _audio_in(ex)
            if files and {0, 1}.issubset({_label_from_path(p)[0] for p in files}):
                return ex, files
            print(f"!! cfg.data_root = {ex} tồn tại nhưng KHÔNG chứa đủ hai lớp leak/noleak.")
        else:
            print(f"!! cfg.data_root = {ex} KHÔNG TỒN TẠI -> tự dò trong /kaggle/input.")
    cands = []
    for base in [Path("/kaggle/input"), Path("./data")]:
        cands += _dirs_with_both(base)
    if not cands:
        return None, []
    if len(cands) > 1:
        # ưu tiên ứng viên có tên trùng với phần cuối của cfg.data_root
        tail = Path(explicit).name.lower() if explicit else ""
        cands.sort(key=lambda p: (0 if tail and p.name.lower() == tail else 1, -len(p.parts), str(p)))
        print(f"!! Tìm thấy {len(cands)} thư mục chứa leak/noleak; dùng thư mục đầu tiên. Danh sách:")
        for c in cands[:10]:
            print("     -", c)
    root = cands[0]
    return root, _audio_in(root)

DATA_ROOT, AUDIO_FILES = discover_root(cfg.data_root)
SYNTHETIC = DATA_ROOT is None

if SYNTHETIC:
    print("!! KHÔNG tìm thấy dataset leak/noleak -> chuyển sang SYNTHETIC DEMO MODE.")
    print("!! No leak/noleak dataset found -> running in SYNTHETIC DEMO MODE.")
else:
    print("Dataset root :", DATA_ROOT)
    print("Audio files  :", len(AUDIO_FILES))
'''

CODE[21] = r'''
# =========================================================
# 3.5b QUYẾT ĐỊNH theo số liệu, có KHOẢNG TIN CẬY
#       [V7-S4] Kiểm định giả thuyết dải thấp bằng CI bootstrap (theo bản ghi) của hiệu số
#               sức phân biệt trung bình: dải cao (>= 800 Hz) - dải thấp (<= 800 Hz).
#       [V7-B3] KHÔNG tự hạ low_band_hz; chỉ mở rộng nếu cfg.low_band_auto=True.
# =========================================================
LOW_HZ = 800.0
_ba = band_auc.reset_index(drop=True).copy()
_lo_idx = np.where(_ba.f_hi.values <= LOW_HZ)[0]
_hi_idx = np.where(_ba.f_lo.values >= LOW_HZ)[0]
_tot = TRAPZ(_P, FREQ_AXIS, axis=1) + 1e-20
_E = np.column_stack([
    TRAPZ(_P[:, (FREQ_AXIS >= a) & (FREQ_AXIS < b)], FREQ_AXIS[(FREQ_AXIS >= a) & (FREQ_AXIS < b)], axis=1) / _tot
    for a, b in zip(_ba.f_lo.values, _ba.f_hi.values)])

def _disc_power(E, y):
    return np.array([abs(2.0 * roc_auc_score(y, E[:, j]) - 1.0) for j in range(E.shape[1])])

def _region_delta(E, y):
    d = _disc_power(E, y)
    return float(d[_hi_idx].mean() - d[_lo_idx].mean())

# Các dải của 3.5 cách đều trên thang LOG -> trung bình không trọng số = trọng số theo log-tần số
# (v6 lấy trọng số theo độ rộng Hz, khiến các dải cao rộng áp đảo phép so sánh).
s_lo = float(_ba.suc_phan_biet.values[_lo_idx].mean()) if len(_lo_idx) else float("nan")
s_hi = float(_ba.suc_phan_biet.values[_hi_idx].mean()) if len(_hi_idx) else float("nan")
delta_obs = s_hi - s_lo

_rng = np.random.default_rng(2024); _bs = []
for _ in range(200 if cfg.smoke_test else 1000):
    i = _rng.integers(0, len(_y), len(_y))
    if len(np.unique(_y[i])) < 2:
        continue
    _bs.append(_region_delta(_E[i], _y[i]))
d_lo, d_hi = (np.percentile(_bs, [2.5, 97.5]) if len(_bs) else (np.nan, np.nan))
best = _ba.loc[_ba.suc_phan_biet.idxmax()]

print("=== KIỂM ĐỊNH GIẢ THUYẾT DẢI TẦN (đơn vị: bản ghi, n = %d) ===" % len(_y))
print(f"Sức phân biệt trung bình |2AUC-1| | dải <= {LOW_HZ:.0f} Hz : {s_lo:.3f} ({len(_lo_idx)} dải)")
print(f"Sức phân biệt trung bình |2AUC-1| | dải >= {LOW_HZ:.0f} Hz : {s_hi:.3f} ({len(_hi_idx)} dải)")
print(f"Hiệu số (cao - thấp) = {delta_obs:+.3f}, CI 95% bootstrap theo bản ghi [{d_lo:+.3f}, {d_hi:+.3f}]")
print(f"Dải mạnh nhất quan sát được: {best.band} Hz (AUC {best.auc:.3f}, {best.huong}, "
      f"p_holm = {best.p_holm:.3f})")

if np.isfinite(d_lo) and d_lo > 0:
    narrative = "low_band_not_supported"
    verdict = "Dữ liệu KHÔNG ủng hộ giả thuyết dải thấp: dải cao phân biệt TỐT HƠN (CI > 0)."
elif np.isfinite(d_hi) and d_hi < 0:
    narrative = "low_band_supported"
    verdict = "Dữ liệu ỦNG HỘ giả thuyết dải thấp: dải thấp phân biệt tốt hơn (CI < 0)."
else:
    narrative = "inconclusive"
    verdict = ("CHƯA PHÂN ĐỊNH ĐƯỢC: CI của hiệu số chứa 0. Với số bản ghi hiện có, dữ liệu KHÔNG "
               "đủ để nói thông tin nằm ở dải thấp hay dải cao.")
LOWBAND_HYPOTHESIS_SUPPORTED = narrative == "low_band_supported"
print("KẾT LUẬN:", verdict)
if narrative == "inconclusive":
    print("  => Trong bài báo: KHÔNG lặp lại giả thuyết 'năng lượng rò rỉ ống nhựa < 800 Hz' như sự")
    print("     thật, và cũng KHÔNG khẳng định điều ngược lại. Phát biểu đúng: sức phân biệt phân tán")
    print("     trên nhiều dải, không dải đơn lẻ nào có ý nghĩa sau hiệu chỉnh đa so sánh.")
elif narrative == "low_band_not_supported":
    print("  => Discussion PHẢI viết rằng trên tập dữ liệu này, dấu hiệu phân biệt KHÔNG tập trung")
    print("     ở dải rất thấp như kỳ vọng lý thuyết cho ống nhựa.")

# ---- Dải cho nhánh STFT dải thấp: đăng ký trước, chỉ được MỞ RỘNG ------------------
_top = _ba.nlargest(3, "suc_phan_biet")
_need = float(_top.f_hi.max())
print(f"\nTop-3 dải mang thông tin: {', '.join(_top.band)} Hz -> cần phủ tới {_need:.0f} Hz")
if cfg.low_band_auto:
    _new = float(min(cfg.f_max, max(cfg.low_band_hz, np.ceil(_need / 100.0) * 100.0)))
    if _new > cfg.low_band_hz + 1e-6:
        print(f"[V7] cfg.low_band_hz: {cfg.low_band_hz:.0f} -> {_new:.0f} Hz (mở rộng để bao top-3 dải)")
        cfg.low_band_hz = _new
    if len(lost) and lost.suc_phan_biet.max() > 0.15:
        _fmin_new = float(min(300.0, max(0.0, np.floor(lost.f_lo.min() / 10.0) * 10.0)))
        if _fmin_new < cfg.f_min:
            print(f"[V7] cfg.f_min: {cfg.f_min:.0f} -> {_fmin_new:.0f} Hz (tránh cắt mất thông tin)")
            cfg.f_min = _fmin_new
elif _need > cfg.low_band_hz:
    print(f"!! cfg.low_band_hz = {cfg.low_band_hz:.0f} Hz KHÔNG phủ hết top-3 dải. Giá trị được giữ vì là "
          "tham số đăng ký trước; nêu trong Limitations hoặc bật cfg.low_band_auto rồi chạy lại.")
else:
    print(f"✓ cfg.low_band_hz = {cfg.low_band_hz:.0f} Hz phủ trọn top-3 dải mang thông tin.")

INFO_BAND_LO, INFO_BAND_HI = float(best.f_lo), float(best.f_hi)
BAND_NARRATIVE = narrative
BAND_TEST = dict(s_lo=s_lo, s_hi=s_hi, delta=delta_obs, ci=[float(d_lo), float(d_hi)], verdict=narrative)
print(f"\nDải dùng cho nhánh STFT dải thấp: 0-{cfg.low_band_hz:.0f} Hz | f_min = {cfg.f_min:.0f} Hz")
'''

CODE[26] = r'''
# =========================================================
# 5.1 Front-end: log-Mel / MFCC / STFT dải thấp (chạy trên GPU)
#     [V6-T6] SpecAugment áp TRƯỚC chuẩn hoá, giá trị che = mức im lặng của mẫu.
#     [V6-T7] MFCC chuẩn hoá theo từng hệ số (CMVN).
#     [V7] Mặt nạ SpecAugment RIÊNG cho từng mẫu; tham số front-end đọc từ cfg LÚC KHỞI TẠO
#          (v6 gắn cứng lúc định nghĩa lớp -> ablation 16 kHz không đổi được sr/n_fft).
# =========================================================
def _fp32_ctx(x):
    # Buộc front-end chạy fp32 ngay cả khi bật AMP: STFT/Mel của tín hiệu đã chuẩn hoá
    # đỉnh có thể vượt dải biểu diễn của fp16 (65504) -> inf -> NaN lan xuống toàn mạng.
    return amp_autocast(x, False)

def _safe_norm(m):
    # Chuẩn hoá theo từng mẫu (không dùng thống kê toàn tập -> không rò rỉ dữ liệu):
    #  - nan_to_num trước khi tính thống kê
    #  - clamp_min độ lệch chuẩn: frame câm/hằng số có sd = 0 -> phép chia 0/0
    m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=-80.0)
    mu = m.mean(dim=(2, 3), keepdim=True)
    sd = m.std(dim=(2, 3), keepdim=True).clamp_min(1e-3)
    return torch.nan_to_num((m - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)

def _safe_norm_percoeff(m):
    # CMVN: chuẩn hoá theo TỪNG hệ số trên trục thời gian.
    m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=-80.0)
    mu = m.mean(dim=3, keepdim=True)
    sd = m.std(dim=3, keepdim=True).clamp_min(1e-3)
    return torch.nan_to_num((m - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)

class SpecAugment(nn.Module):
    # Che ngẫu nhiên dải tần / khoảng thời gian. CHỈ khi training.
    # Giá trị che = mức nhỏ nhất của chính mẫu đó ("im lặng"), áp TRƯỚC chuẩn hoá.
    # [V7] mỗi mẫu một mặt nạ riêng (v6: một mặt nạ chung cho cả batch -> kém đa dạng).
    def __init__(self, f=None, t=None, n=2, freq=True):
        super().__init__()
        self.f = cfg.specaug_freq_mask if f is None else int(f)
        self.t = cfg.specaug_time_mask if t is None else int(t)
        self.n, self.freq = int(n), bool(freq)
        self.generator = None      # generator riêng của fold (tất định theo fold)
    def _band_mask(self, B, L, W, device):
        g = self.generator
        w = torch.randint(0, W + 1, (B,), generator=g)
        s0 = (torch.rand(B, generator=g) * (L - w).clamp_min(1).float()).long()
        ar = torch.arange(L)
        return ((ar[None, :] >= s0[:, None]) & (ar[None, :] < (s0 + w)[:, None])).to(device)
    def forward(self, s):
        if not self.training or (self.f <= 0 and self.t <= 0):
            return s
        B, C, Fq, Tm = s.shape
        fill = s.amin(dim=(2, 3), keepdim=True).detach()
        mask = torch.zeros(B, 1, Fq, Tm, dtype=torch.bool, device=s.device)
        for _ in range(self.n):
            if self.freq and self.f > 0 and Fq > self.f:
                mask = mask | self._band_mask(B, Fq, self.f, s.device)[:, None, :, None]
            if self.t > 0 and Tm > self.t:
                mask = mask | self._band_mask(B, Tm, self.t, s.device)[:, None, None, :]
        return torch.where(mask, fill, s)

def set_specaug_generators(model, generator):
    # Gán generator riêng cho mọi SpecAugment trong mô hình -> tất định theo fold.
    n = 0
    for m in model.modules():
        if isinstance(m, SpecAugment):
            m.generator = generator; n += 1
    return n

class LogMelFrontend(nn.Module):
    def __init__(self, sr=None, n_fft=None, hop=None, n_mels=None, f_min=None, f_max=None,
                 specaug=None):
        super().__init__()
        sr = sr or cfg.target_sr; n_fft = n_fft or cfg.n_fft; hop = hop or cfg.hop_length
        n_mels = n_mels or cfg.n_mels
        f_min = cfg.f_min if f_min is None else f_min
        f_max = min(cfg.f_max if f_max is None else f_max, sr / 2)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
            f_min=f_min, f_max=f_max, power=2.0, center=True)
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)
        self.specaug = specaug
    def wave_to_db(self, x):            # dB thô, chưa che, chưa chuẩn hoá
        with _fp32_ctx(x):
            return self.to_db(self.mel(x.float())).unsqueeze(1)
    def forward(self, x):               # x: [B, T]
        s = self.wave_to_db(x)
        if self.specaug is not None:
            s = self.specaug(s)         # TRƯỚC khi chuẩn hoá
        return _safe_norm(s)

class MFCCFrontend(nn.Module):
    def __init__(self, sr=None, n_mfcc=None, specaug=None):
        super().__init__()
        sr = sr or cfg.target_sr
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sr, n_mfcc=n_mfcc or cfg.n_mfcc, log_mels=True,
            melkwargs=dict(n_fft=cfg.n_fft, hop_length=cfg.hop_length, n_mels=cfg.n_mels,
                           f_min=cfg.f_min, f_max=min(cfg.f_max, sr / 2)))
        self.specaug = specaug
    def forward(self, x):
        with _fp32_ctx(x):
            m = self.mfcc(x.float()).unsqueeze(1)
        if self.specaug is not None:
            m = self.specaug(m)
        if cfg.mfcc_norm == "per_coeff":
            return _safe_norm_percoeff(m)
        return _safe_norm(m)

class LowBandSTFTFrontend(nn.Module):
    # Chỉ giữ 0 - low_band_hz (tham số đăng ký trước, xem mục 3.5b).
    def __init__(self, sr=None, n_fft=None, hop=None, fmax=None, specaug=None):
        super().__init__()
        sr = sr or cfg.target_sr; n_fft = n_fft or cfg.n_fft; hop = hop or cfg.hop_length
        fmax = min(cfg.low_band_hz if fmax is None else fmax, sr / 2)
        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop, power=2.0)
        self.kmax = max(2, int(fmax / (sr / n_fft)) + 1)
        self.specaug = specaug
    def forward(self, x):
        with _fp32_ctx(x):
            s = self.spec(x.float())[:, :self.kmax, :]
            s = torch.log(s.clamp_min(1e-10)).unsqueeze(1)
        if self.specaug is not None:
            s = self.specaug(s)
        return _safe_norm(s)

# Kiểm tra kích thước tensor / sanity check on shapes
with torch.no_grad():
    _x = torch.from_numpy(np.asarray(X_ALL[:2])).float()
    print("waveform      :", tuple(_x.shape))
    print("log-Mel       :", tuple(LogMelFrontend()(_x).shape))
    print("MFCC (CMVN)   :", tuple(MFCCFrontend()(_x).shape))
    print("low-band STFT :", tuple(LowBandSTFTFrontend()(_x).shape),
          f"| giữ 0-{cfg.low_band_hz:.0f} Hz")
    # SpecAugment: mỗi mẫu một mặt nạ, và chỉ hoạt động khi training
    _sa = SpecAugment(f=8, t=8).train(); _sa.generator = torch.Generator().manual_seed(0)
    _s = torch.randn(4, 1, 64, 63); _o = _sa(_s.clone())
    _masked = (_o != _s).flatten(1).float().mean(1)
    assert (_masked > 0).any(), "SpecAugment không che gì"
    assert torch.equal(SpecAugment().eval()(_s.clone()), _s), "SpecAugment che cả khi eval"
    print("SpecAugment   : tỉ lệ ô bị che theo mẫu =", [round(float(v), 3) for v in _masked])
'''

CODE[32] = r'''
# =========================================================
# 6.1 Thiết lập cross-validation — PHÂN TẦNG TRÊN BẢNG BẢN GHI
#     [V7-S1] Tách thành hàm build_splits(seed) để CV lặp (mỗi seed một phân hoạch mới);
#             check_splits() áp 4 assertion cứng cho MỌI phân hoạch.
# =========================================================
rec_lab = frames.groupby("record_id").label.first()
rec_to_frames = frames.groupby("record_id").indices
REC_IDS = rec_lab.index.values
REC_Y = rec_lab.values.astype(int)
n_rec_min = pd.Series(REC_Y).value_counts().min()
K = int(min(cfg.n_splits, n_rec_min))
USE_LOGO = K < 2

def _expand(rec_subset):
    return np.sort(np.concatenate([rec_to_frames[r] for r in rec_subset]))

def build_splits(seed):
    if USE_LOGO:
        return [(_expand(np.delete(REC_IDS, i)), _expand(REC_IDS[[i]])) for i in range(len(REC_IDS))]
    if cfg.fold_mode == "record_stratified":
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=int(seed))
        return [(_expand(REC_IDS[a]), _expand(REC_IDS[b]))
                for a, b in skf.split(np.zeros(len(REC_IDS)), REC_Y)]
    sgkf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=int(seed))
    return list(sgkf.split(np.zeros(len(frames)), y_all, groups_all))

def check_splits(splits):
    seen = set()
    for k, (tr, va) in enumerate(splits):
        gtr, gva = set(groups_all[tr]), set(groups_all[va])
        assert not (gtr & gva), f"DATA LEAKAGE ở fold {k}: {gtr & gva}"
        assert len(np.unique(y_all[va])) == 2 or USE_LOGO, f"Fold {k} thiếu một lớp ở validation"
        assert len(tr) + len(va) == len(frames), f"Fold {k} không phủ hết frame"
        if not USE_LOGO:
            assert not (seen & gva), f"Bản ghi lặp ở hai fold validation: {seen & gva}"
            seen |= gva
    if not USE_LOGO:
        assert seen == set(REC_IDS), "Có bản ghi không xuất hiện ở validation lần nào"
    return True

if USE_LOGO:
    print("!! Quá ít bản ghi -> LeaveOneGroupOut.")
elif K < cfg.n_splits:
    print(f"!! Giảm số fold {cfg.n_splits} -> {K} (lớp hiếm chỉ có {n_rec_min} bản ghi).")

SPLITS = build_splits(cfg.seed)
check_splits(SPLITS)
print(f"✓ {len(SPLITS)} fold | mỗi bản ghi xuất hiện ĐÚNG MỘT LẦN ở validation | "
      f"không có bản ghi nào ở cả hai phía")

split_tab = pd.DataFrame([
    dict(fold=k, n_train_frames=len(tr), n_val_frames=len(va),
         n_train_rec=len(set(groups_all[tr])), n_val_rec=len(set(groups_all[va])),
         val_leak_REC=int(sum(rec_lab[r] for r in set(groups_all[va]))),
         val_leak_rate_rec=round(float(np.mean([rec_lab[r] for r in set(groups_all[va])])), 3),
         val_leak_rate_frame=round(float(y_all[va].mean()), 3))
    for k, (tr, va) in enumerate(SPLITS)])
display(Markdown("### Table 4. Cross-validation partitioning"))
display(split_tab)
spread = split_tab.val_leak_rate_rec.max() - split_tab.val_leak_rate_rec.min()
print(f"Biên độ dao động tỉ lệ lớp giữa các fold (mức bản ghi): {spread:.3f}")
print("  (< 0.10 là tốt; giá trị lớn làm phương sai giữa các fold tăng mạnh)")
split_tab.to_csv(TAB_DIR / "table4_cv_splits.csv", index=False)
'''

CODE[46] = r'''
# =========================================================
# 7.5 HUẤN LUYỆN TOÀN BỘ MÔ HÌNH + CV LẶP / TRAIN ALL MODELS WITH REPEATED CV
#     [V7-S1] Mỗi seed trong cfg.seeds = một PHÂN HOẠCH fold mới + khởi tạo mới.
#             Lần lặp 0 là lần chạy chính (checkpoint, diễn giải, triển khai).
# =========================================================
ALL_RESULTS, OOF, OOF_OPT, INNER, OOF_CAL = {}, {}, {}, {}, {}
INNER_CAL = {}
OOF_REPS = defaultdict(list)                     # OOF của từng lần lặp
SPLITS_REPS = [SPLITS] + [build_splits(s) for s in cfg.seeds[1:]]
for _sp in SPLITS_REPS[1:]:
    check_splits(_sp)
timing, rep_rows = [], []

for ri, sd in enumerate(cfg.seeds):
    print(f"\n######## CV LẶP {ri + 1}/{len(cfg.seeds)} (seed {sd}) ########")
    for name in cfg.models:
        t0 = time.time()
        res = run_cv(name, splits=SPLITS_REPS[ri], seed_base=sd, tag=f"rep{ri}")
        oo = assemble_oof(res, "val")
        OOF_REPS[name].append(oo)
        if ri == 0:
            ALL_RESULTS[name] = res
            OOF[name] = oo                                   # trung thực (epoch chọn bởi inner)
            OOF_OPT[name] = assemble_oof(res, "opt")         # thiên lệch (epoch tốt nhất trên tập báo cáo)
            INNER[name] = pool_inner(res)
            INNER_CAL[name] = pool_inner_cal(res)
            oof_c = np.full(len(frames), np.nan)
            for r in res:
                oof_c[r["val_idx"]] = r["val_probs_cal"]
            OOF_CAL[name] = oof_c
            if cfg.save_models:
                for r in res:
                    torch.save(r["state"], CKPT_DIR / f"{name}_fold{r['fold']}.pt")
            timing.append(dict(model=name, total_s=time.time() - t0,
                               per_fold_s=np.mean([r["train_time_s"] for r in res]),
                               mean_best_epoch=np.mean([r["best_epoch"] for r in res]),
                               min_best_epoch=int(np.min([r["best_epoch"] for r in res])),
                               frac_after_lr_peak=np.mean([r["selected_after_lr_peak"] for r in res]),
                               nested_folds=int(sum(r["nested"] for r in res)),
                               skipped_batches=int(sum(r["n_skipped"] for r in res))))
        mk = ~np.isnan(oo)
        auc_r, _ = record_auc_simple(oo)
        rep_rows.append(dict(model=name, repeat=ri, seed=sd,
                             auc_frame=float(roc_auc_score(y_all[mk], oo[mk])), auc_record=auc_r,
                             mean_best_epoch=float(np.mean([r["best_epoch"] for r in res]))))
        print(f"--> {name} (lặp {ri}, seed {sd}): {time.time()-t0:.1f}s | OOF phủ "
              f"{np.mean(mk)*100:.1f}% | AUC frame {rep_rows[-1]['auc_frame']:.4f} | "
              f"AUC bản ghi ({cfg.record_agg}) {auc_r:.4f}")
        if ri > 0:
            for r in res:
                r["state"] = None
            del res; gc.collect()
            if DEVICE.type == "cuda": torch.cuda.empty_cache()

timing_df = pd.DataFrame(timing).round(3)
display(Markdown("### Table 5. Training cost and checkpoint selection (repeat 0)"))
display(timing_df)
timing_df.to_csv(TAB_DIR / "table5_training_cost.csv", index=False)

if (timing_df.min_best_epoch < SELECT_FROM_EPOCH).any():
    print(f"!! LỖI: có checkpoint được chọn TRƯỚC epoch {SELECT_FROM_EPOCH} - ràng buộc [V7-B2] bị vi phạm.")
else:
    print(f"✓ [V7-B2] Mọi checkpoint được chọn từ epoch {SELECT_FROM_EPOCH} (đỉnh lịch LR) trở đi.")

# ---- Bảng 5b: CV lặp ---------------------------------------------------------
rep_df = pd.DataFrame(rep_rows)
rep_df.round(4).to_csv(TAB_DIR / "table5b_repeated_cv_raw.csv", index=False)
REPEAT_SUMMARY, OOF_ENS, _rows = {}, {}, []
for name in cfg.models:
    ens = np.nanmean(np.vstack(OOF_REPS[name]), axis=0)   # mỗi frame là OOF ở MỌI lần lặp
    OOF_ENS[name] = ens
    auc_e, g_e = record_auc_simple(ens)
    ci = cluster_bootstrap_ci(g_e.index.values, g_e.y.values, g_e.p.values, roc_auc_score,
                              n=cfg.bootstrap_n, seed=0)
    d = rep_df[rep_df.model == name]
    sd_r = float(d.auc_record.std(ddof=1)) if len(d) > 1 else float("nan")
    REPEAT_SUMMARY[name] = dict(n_repeats=int(len(d)), mean=float(d.auc_record.mean()), sd=sd_r,
                                ens_auc=auc_e, ens_lo=ci["lo"], ens_hi=ci["hi"])
    _rows.append(dict(model=name, n_repeats=int(len(d)),
                      AUC_record_mean=float(d.auc_record.mean()), AUC_record_sd=sd_r,
                      AUC_record_min=float(d.auc_record.min()), AUC_record_max=float(d.auc_record.max()),
                      AUC_frame_mean=float(d.auc_frame.mean()),
                      AUC_frame_sd=float(d.auc_frame.std(ddof=1)) if len(d) > 1 else float("nan"),
                      AUC_record_repeat_avg=auc_e, CI_lo=ci["lo"], CI_hi=ci["hi"]))
rep_sum = pd.DataFrame(_rows).sort_values("AUC_record_repeat_avg", ascending=False).reset_index(drop=True)
rep_sum.round(4).to_csv(TAB_DIR / "table5b_repeated_cv.csv", index=False)
display(Markdown(f"### Table 5b. Repeated cross-validation ({len(cfg.seeds)} partitions; "
                 f"record-level AUC with `{cfg.record_agg}` aggregation)"))
display(rep_sum.round(4))
if len(cfg.seeds) > 1:
    _sd = float(np.nanmean(rep_sum.AUC_record_sd))
    print(f"Độ lệch chuẩn trung bình giữa các lần lặp (mức bản ghi): {_sd:.4f}")
    print("=> Chênh lệch AUC giữa hai kiến trúc NHỎ HƠN con số này là nhiễu phân hoạch/khởi tạo,")
    print("   KHÔNG phải khác biệt kiến trúc.")
    print("Cột AUC_record_repeat_avg = AUC khi TRUNG BÌNH xác suất OOF qua các lần lặp: ổn định")
    print("nhất và vẫn trung thực (mỗi dự đoán đến từ mô hình chưa thấy bản ghi đó).")
else:
    print("LƯU Ý: chỉ 1 phân hoạch. Đặt cfg.seeds=(42, 43, 44) để đo độ bất định do phân hoạch.")
'''

CODE[55] = r'''
# =========================================================
# 8.7 Tỉ lệ thể hiện dương trong túi dương / witness rate + KIỂM ĐỊNH TIỀN ĐỀ MIL
#     [V7] Kết luận BA MỨC (ủng hộ / chưa kết luận / không ủng hộ) thay cho một ngưỡng 0,5.
# =========================================================
name = BEST_MODEL
msk = ~np.isnan(OOF[name])
dfw = pd.DataFrame(dict(record_id=frames.record_id.values[msk],
                        label=y_all[msk], p=OOF[name][msk]))
thr_w = THR[name]
wit = (dfw.groupby("record_id")
       .agg(label=("label", "first"), n=("p", "size"),
            frac_above=("p", lambda s: float((s >= thr_w).mean())),
            p_mean=("p", "mean"), p_max=("p", "max"), p_std=("p", "std")).reset_index())
wit.to_csv(TAB_DIR / "table12_witness_rate.csv", index=False)

w_leak = wit.loc[wit.label == 1, "frac_above"]
w_nole = wit.loc[wit.label == 0, "frac_above"]
display(Markdown("### Table 12. Fraction of frames above threshold per recording"))
display(wit.groupby("label").frac_above.describe().round(3))

W_LEAK_MED = float(w_leak.median())
if W_LEAK_MED < 0.5:
    WITNESS_VERDICT = "supported"
elif W_LEAK_MED >= 0.9:
    WITNESS_VERDICT = "not_supported"
else:
    WITNESS_VERDICT = "inconclusive"
MIL_PREMISE_SUPPORTED = WITNESS_VERDICT == "supported"
print(f"Bản ghi LEAK  : trung vị {W_LEAK_MED:.2f} số frame vượt ngưỡng "
      f"(khoảng {w_leak.min():.2f}-{w_leak.max():.2f}, IQR {w_leak.quantile(.25):.2f}-{w_leak.quantile(.75):.2f})")
print(f"Bản ghi NO-LEAK: trung vị {w_nole.median():.2f}")

print()
if WITNESS_VERDICT == "supported":
    print(f"KẾT LUẬN: tiền đề MIL ĐƯỢC ỦNG HỘ (trung vị {W_LEAK_MED:.2f} < 0,5).")
    print("  -> Có cơ sở nói rằng nhãn mức frame bị nhiễu; có thể thử mil_mode='bag'.")
elif WITNESS_VERDICT == "not_supported":
    print(f"KẾT LUẬN: tiền đề MIL KHÔNG ĐƯỢC ủng hộ (trung vị {W_LEAK_MED:.2f} >= 0,9).")
    print("  -> Gần như MỌI frame trong bản ghi leak đều được mô hình coi là 'rò rỉ'.")
    print("  -> KHÔNG được mô tả nhãn mức frame là 'nhiễu nặng'; KHÔNG bật mil_mode='bag'.")
    print("  -> Điểm số gần như hằng số TRONG bản ghi: xem mục 8.8 (ICC) và 9.6 (confound).")
else:
    print(f"KẾT LUẬN: CHƯA KẾT LUẬN ĐƯỢC về tiền đề MIL (trung vị {W_LEAK_MED:.2f} nằm giữa 0,5 và 0,9).")
    print("  -> Phần lớn nhưng không phải mọi frame của bản ghi leak vượt ngưỡng; độ phân tán giữa")
    print("     các bản ghi lớn. KHÔNG khẳng định nhãn frame 'nhiễu nặng', cũng KHÔNG khẳng định ngược lại.")
    print("  -> Muốn phân định: dán nhãn thời điểm rò rỉ trên một mẫu bản ghi (mục 16).")

vc_w = variance_components(np.where(msk)[0], y_all[msk], OOF[name][msk])
print(f"\nPhân rã phương sai ({name}): ICC = {vc_w['icc']:.3f} -> "
      f"{vc_w['icc']*100:.0f}% phương sai nằm GIỮA các bản ghi, "
      f"{(1-vc_w['icc'])*100:.0f}% nằm giữa các frame trong cùng bản ghi.")
if vc_w["icc"] > 0.5 and WITNESS_VERDICT != "supported":
    print("=> Điểm số chủ yếu phân biệt BẢN GHI hơn là các đoạn thời gian trong bản ghi.")
    print("   Đây là dấu hiệu CẦN kiểm tra confound (mục 9.6), chưa phải bằng chứng confound.")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for lab, nm in [(0, "noleak"), (1, "leak")]:
    ax[0].hist(dfw.loc[dfw.label == lab, "p"], bins=40, alpha=.6, density=True,
               color=PAL[nm], label=f"frames in {nm} recordings")
ax[0].axvline(thr_w, color="k", ls="--", lw=1, label=f"threshold={thr_w:.2f}")
ax[0].set_xlabel("frame-level leak probability"); ax[0].set_ylabel("density")
ax[0].set_title("(a) Frame-probability distribution by bag label"); ax[0].legend(fontsize=7)

for lab, nm in [(0, "noleak"), (1, "leak")]:
    ax[1].hist(wit.loc[wit.label == lab, "frac_above"], bins=12, alpha=.65,
               color=PAL[nm], label=nm)
ax[1].axvline(W_LEAK_MED, color="k", ls=":", lw=1.2)
ax[1].axvspan(0.5, 0.9, color=PAL["warn"], alpha=.12, lw=0)
ax[1].set_xlabel("fraction of frames above threshold (per recording)")
ax[1].set_ylabel("number of recordings"); ax[1].legend(fontsize=8)
ax[1].set_title(f"(b) Witness rate per recording\nmedian of leak class = {W_LEAK_MED:.2f} "
                f"(shaded = inconclusive zone)")

w = wit.sort_values(["label", "frac_above"])
ax[2].barh(range(len(w)), w.frac_above,
           color=[PAL["leak"] if l else PAL["noleak"] for l in w.label])
ax[2].set_yticks(range(len(w))); ax[2].set_yticklabels(w.record_id, fontsize=6)
ax[2].set_xlabel("fraction of frames above threshold"); ax[2].set_title("(c) Individual recordings")
_cap_w = {"supported": "A median below 0.5 supports the multiple-instance premise that leak evidence "
                       "occupies only part of a leak recording.",
          "not_supported": "A median of 0.9 or above means almost every frame of a leak recording is "
                           "scored as leak: the noisy-frame-label premise is NOT supported.",
          "inconclusive": "A median between 0.5 and 0.9 does not resolve the multiple-instance premise; "
                          "frame-level annotation of leak onsets is required."}[WITNESS_VERDICT]
savefig(fig, "witness_rate", "Witness rate at record level (fraction of frames above the operating "
        f"threshold in each recording; median of leak recordings = {W_LEAK_MED:.2f}). " + _cap_w)
'''

CODE[67] = r'''
# =========================================================
# 9.5 BASELINE CỔ ĐIỂN - cùng giao thức, so sánh GHÉP CẶP với CNN
#     [V7-S3] thêm bộ đặc trưng thống kê log-Mel (bất biến gain); kết luận dựa trên CI
#             bootstrap GHÉP CẶP của ΔAUC (+ Holm), không dựa trên "hai CI có chồng lấn".
# =========================================================
def frame_psd_matrix(X, n_fft=1024, hop=512, sr=None):
    # Ma trận PSD cho mọi frame, tính vector hoá (nhanh hơn vòng lặp welch).
    sr = sr or cfg.target_sr
    X = np.asarray(X, dtype=np.float32)
    win = np.hanning(n_fft).astype(np.float32)
    nseg = 1 + (X.shape[1] - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(nseg)[:, None]
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    out = np.empty((X.shape[0], len(freqs)), dtype=np.float32)
    for i in range(0, X.shape[0], 256):
        blk = X[i:i + 256]
        S = np.fft.rfft(blk[:, idx] * win, axis=2)
        out[i:i + 256] = (np.abs(S) ** 2).mean(axis=1)
    return out, freqs

def spectral_features(P, freqs, rms=None, zcr=None):
    tot = P.sum(axis=1) + 1e-20
    F_ = {}
    for (a, b), nm in zip(BANDS, BAND_NAMES):
        m = (freqs >= a) & (freqs < b)
        F_[f"band_{nm}"] = (P[:, m].sum(axis=1) / tot).astype(np.float32)
    cen = (P * freqs[None, :]).sum(axis=1) / tot
    F_["centroid"] = cen
    F_["spread"] = np.sqrt((P * (freqs[None, :] - cen[:, None]) ** 2).sum(axis=1) / tot)
    F_["flatness"] = np.exp(np.mean(np.log(P + 1e-12), axis=1)) / (P.mean(axis=1) + 1e-20)
    cs = np.cumsum(P, axis=1) / tot[:, None]
    F_["rolloff85"] = freqs[np.argmax(cs >= 0.85, axis=1)]
    if rms is not None: F_["rms"] = np.asarray(rms, dtype=np.float32)
    if zcr is not None: F_["zcr"] = np.asarray(zcr, dtype=np.float32)
    return pd.DataFrame(F_)

@torch.no_grad()
def logmel_stats_matrix(X, batch=256):
    # Mean + std theo thời gian của từng dải log-Mel, sau khi trừ mức dB trung bình của frame
    # -> bất biến với gain; đúng biểu diễn mà CNN nhìn thấy (trước chuẩn hoá theo mẫu).
    fe = LogMelFrontend().to(DEVICE).eval()
    out = []
    for s in range(0, len(X), batch):
        xb = torch.from_numpy(np.asarray(X[s:s + batch])).float().to(DEVICE)
        db = fe.wave_to_db(xb)[:, 0]
        db = db - db.mean(dim=(1, 2), keepdim=True)
        out.append(torch.cat([db.mean(2), db.std(2)], 1).float().cpu().numpy())
    del fe
    return np.concatenate(out)

print("Đang tính đặc trưng cho mọi frame...")
_t0 = time.time()
P_ALL, FREQS_BL = frame_psd_matrix(X_ALL)
FEAT_ALL = spectral_features(P_ALL, FREQS_BL, rms=frames.rms.values, zcr=frames.zcr.values)
FEAT_ALL.to_csv(TAB_DIR / "features_classical.csv", index=False)
FEATURE_NAMES = list(FEAT_ALL.columns)
FEATURE_SETS = {"spectral13": FEAT_ALL}
if "logmel_stats" in cfg.baseline_feature_sets:
    _lm = logmel_stats_matrix(X_ALL)
    FEATURE_SETS["logmel_stats"] = pd.DataFrame(
        _lm, columns=[f"mel{i:02d}_mean" for i in range(cfg.n_mels)] +
                     [f"mel{i:02d}_std" for i in range(cfg.n_mels)])
FEATURE_SETS = {k: v for k, v in FEATURE_SETS.items() if k in cfg.baseline_feature_sets}
print(f"  xong trong {time.time()-_t0:.1f}s | " +
      ", ".join(f"{k}: {v.shape[1]} chiều" for k, v in FEATURE_SETS.items()))

def _sk_model(kind, seed=0):
    if kind == "logreg":
        return LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    if kind == "svm":
        return SVC(C=2.0, gamma="scale", probability=False, class_weight="balanced")
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                              max_depth=3, random_state=seed)
    raise ValueError(kind)

def _scores_from_model(mdl, Xv):
    if hasattr(mdl, "predict_proba"):
        return mdl.predict_proba(Xv)[:, 1]
    d = mdl.decision_function(Xv)
    return 1.0 / (1.0 + np.exp(-d))

def run_classical(feat_df, splits=None, seed_base=0):
    # Trả về các hàng kết quả + điểm OOF MỨC BẢN GHI của từng baseline (để so sánh ghép cặp).
    splits = SPLITS if splits is None else splits
    names = list(feat_df.columns)
    Xf = np.nan_to_num(feat_df.values.astype(np.float64))
    yf = y_all.astype(int); recf = groups_all
    fr = feat_df.copy(); fr["record_id"] = frames.record_id.values; fr["label"] = yf
    Xr = fr.groupby("record_id")[names].mean()
    yr = fr.groupby("record_id").label.first()
    rec_ids = Xr.index.values; pos = {r: i for i, r in enumerate(rec_ids)}
    rows, rec_scores = [], {}
    for kind in cfg.baseline_models:
        oof_f = np.full(len(frames), np.nan); oof_r = np.full(len(rec_ids), np.nan)
        for k, (tr, va) in enumerate(splits):
            sc = StandardScaler().fit(Xf[tr])
            mdl = _sk_model(kind, seed_base + k).fit(sc.transform(Xf[tr]), yf[tr])
            oof_f[va] = _scores_from_model(mdl, sc.transform(Xf[va]))
            itr = np.array([pos[r] for r in pd.unique(recf[tr])])
            iva = np.array([pos[r] for r in pd.unique(recf[va])])
            sc2 = StandardScaler().fit(Xr.values[itr])
            mdl2 = _sk_model(kind, seed_base + 100 + k).fit(sc2.transform(Xr.values[itr]), yr.values[itr])
            oof_r[iva] = _scores_from_model(mdl2, sc2.transform(Xr.values[iva]))
        mk = ~np.isnan(oof_f)
        ci = cluster_bootstrap_ci(recf[mk], yf[mk], oof_f[mk], roc_auc_score, n=cfg.bootstrap_n, seed=0)
        rows.append(dict(baseline=kind, level="frame", auc=float(roc_auc_score(yf[mk], oof_f[mk])),
                         lo=ci["lo"], hi=ci["hi"], n=int(mk.sum())))
        # (a) frame -> gộp p_mean theo bản ghi (cùng quy tắc với CNN)
        auc_fr, g_fr = record_auc_simple(oof_f)
        ci2 = cluster_bootstrap_ci(g_fr.index.values, g_fr.y.values, g_fr.p.values, roc_auc_score,
                                   n=cfg.bootstrap_n, seed=0)
        rows.append(dict(baseline=kind, level="frame->record", auc=auc_fr, lo=ci2["lo"], hi=ci2["hi"],
                         n=int(len(g_fr))))
        rec_scores[(kind, "frame->record")] = g_fr.p
        # (b) trực tiếp trên vector trung bình của bản ghi
        mr = ~np.isnan(oof_r)
        cir = cluster_bootstrap_ci(rec_ids[mr], yr.values[mr], oof_r[mr], roc_auc_score,
                                   n=cfg.bootstrap_n, seed=0)
        rows.append(dict(baseline=kind, level="record", auc=float(roc_auc_score(yr.values[mr], oof_r[mr])),
                         lo=cir["lo"], hi=cir["hi"], n=int(mr.sum())))
        rec_scores[(kind, "record")] = pd.Series(oof_r, index=rec_ids)
    return pd.DataFrame(rows), rec_scores

if cfg.run_baseline_models:
    _parts, BASE_REC_SCORES = [], {}
    for fs, fdf in FEATURE_SETS.items():
        _t0 = time.time()
        df_, sc_ = run_classical(fdf)
        df_.insert(0, "features", fs); _parts.append(df_)
        for (kind, lvl), s in sc_.items():
            BASE_REC_SCORES[(fs, kind, lvl)] = s
        print(f"  baseline [{fs}] xong trong {time.time()-_t0:.1f}s")
    base_df = pd.concat(_parts, ignore_index=True)

    # baseline "chỉ dùng mức năng lượng" (1 đặc trưng) - kiểm tra confound mức to/nhỏ
    lvl_rec = pd.DataFrame(dict(record_id=frames.record_id.values, rms=frames.rms.values,
                                label=y_all.astype(int))).groupby("record_id").agg(
        rms=("rms", "mean"), label=("label", "first"))
    lvl_auc = float(roc_auc_score(lvl_rec.label, lvl_rec.rms))
    lvl_ci = cluster_bootstrap_ci(lvl_rec.index.values, lvl_rec.label.values, lvl_rec.rms.values,
                                  roc_auc_score, n=cfg.bootstrap_n, seed=0)
    base_df = pd.concat([base_df, pd.DataFrame([dict(
        features="rms", baseline="Level only (RMS)", level="record", auc=lvl_auc,
        lo=lvl_ci["lo"], hi=lvl_ci["hi"], n=len(lvl_rec))])], ignore_index=True)
    display(Markdown("### Table 9c. Classical baselines (same folds, cluster-bootstrap CI)"))
    display(base_df.round(4))
    base_df.to_csv(TAB_DIR / "table9c_classical_baselines.csv", index=False)

    # ---- So sánh GHÉP CẶP: CNN (DEPLOY_MODEL, chọn trên validation-trong) vs từng baseline ----
    CNN_REF = DEPLOY_MODEL
    g_cnn = REC[CNN_REF].set_index("record_id")
    cmp_rows = []
    for (fs, kind, lvl), s in BASE_REC_SCORES.items():
        common = g_cnn.index.intersection(s.dropna().index)
        yv = g_cnn.loc[common, "label"].values.astype(int)
        pa = g_cnn.loc[common, best_agg].values; pb = s.loc[common].values
        d = paired_cluster_delta_ci(common.values, yv, pa, pb, roc_auc_score, n=cfg.bootstrap_n, seed=0)
        cmp_rows.append(dict(cnn=CNN_REF, features=fs, baseline=kind, level=lvl,
                             AUC_cnn=float(roc_auc_score(yv, pa)), AUC_base=float(roc_auc_score(yv, pb)),
                             dAUC=float(roc_auc_score(yv, pa) - roc_auc_score(yv, pb)),
                             dAUC_lo=d["lo"], dAUC_hi=d["hi"], p=d["p_two_sided"], n=int(len(common))))
    BASE_CMP = pd.DataFrame(cmp_rows)
    BASE_CMP["p_holm"] = holm(BASE_CMP.p.fillna(1.0).values)
    BASE_CMP["cnn_better"] = BASE_CMP.dAUC_lo > 0
    BASE_CMP = BASE_CMP.sort_values("AUC_base", ascending=False).reset_index(drop=True)
    display(Markdown(f"### Table 9c-2. Paired comparison at record level: {CNN_REF} (selected on inner "
                     f"validation) vs each baseline (paired cluster bootstrap, Holm over {len(BASE_CMP)} tests)"))
    display(BASE_CMP.round(4))
    BASE_CMP.round(6).to_csv(TAB_DIR / "table9c2_cnn_vs_baselines_paired.csv", index=False)

    _strong = BASE_CMP.iloc[0]
    BEST_BASE_TXT = (f"{_strong.baseline} on {_strong.features} ({_strong.level}, AUC {_strong.AUC_base:.3f})")
    _n_beat = int((BASE_CMP.cnn_better & (BASE_CMP.p_holm < 0.05)).sum())
    print(f"\nCNN tham chiếu: {CNN_REF} (AUC bản ghi {BASE_CMP.AUC_cnn.iloc[0]:.4f})")
    print(f"Baseline mạnh nhất: {BEST_BASE_TXT}")
    print(f"CNN vượt baseline có ý nghĩa (CI ghép cặp > 0 VÀ p_holm < 0.05): {_n_beat}/{len(BASE_CMP)}")
    if _n_beat == len(BASE_CMP):
        BASELINE_VERDICT_VI = ("CNN vượt MỌI baseline cổ điển với CI ghép cặp của ΔAUC nằm hoàn toàn trên 0 "
                               "(hiệu chỉnh Holm)")
        BASELINE_VERDICT_EN = ("the CNN outperformed EVERY classical baseline, with paired-bootstrap CIs "
                               "for dAUC entirely above zero (Holm-corrected)")
    elif bool(BASE_CMP.cnn_better.iloc[0]) and _n_beat > 0:
        BASELINE_VERDICT_VI = (f"CNN vượt baseline mạnh nhất theo CI ghép cặp nhưng chỉ {_n_beat}/{len(BASE_CMP)} "
                               "so sánh còn có ý nghĩa sau hiệu chỉnh Holm - lợi thế của học sâu cần nêu thận trọng")
        BASELINE_VERDICT_EN = (f"the CNN exceeded the strongest baseline by paired CI, but only {_n_beat}/"
                               f"{len(BASE_CMP)} comparisons survived Holm correction; the advantage of deep "
                               "learning must be stated cautiously")
    else:
        BASELINE_VERDICT_VI = ("dữ liệu KHÔNG chứng minh được CNN vượt baseline cổ điển mạnh nhất (CI ghép cặp "
                               "của ΔAUC chứa 0); đóng góp nên đặt ở giao thức đánh giá, không ở kiến trúc")
        BASELINE_VERDICT_EN = ("the data do NOT show that the CNN outperforms the strongest classical "
                               "baseline (paired CI for dAUC includes zero)")
    print("=> " + BASELINE_VERDICT_VI)
    print(f"\nBaseline 'CHỈ mức năng lượng': AUC bản ghi {lvl_auc:.4f} [{lvl_ci['lo']:.4f}, {lvl_ci['hi']:.4f}]")
    if lvl_auc > 0.75:
        print("!! Mức to/nhỏ của tín hiệu ĐÃ phân biệt được hai lớp ở mức cao. Đây là confound")
        print("   nghiêm trọng: mô hình có thể chỉ đang học 'độ ồn của vị trí thu'. Phải nêu rõ.")
    # tương thích tên cũ cho các cell phía sau
    best_base_r = base_df[(base_df.level == "record") & (base_df.baseline != "Level only (RMS)")] \
        .sort_values("auc", ascending=False).iloc[0]
else:
    BEST_BASE_TXT, BASELINE_VERDICT_VI, BASELINE_VERDICT_EN = "không chạy", "chưa đánh giá", "not evaluated"
    print("cfg.run_baseline_models=False -> bỏ qua baseline cổ điển (KHÔNG nên bỏ khi viết bài).")
'''

CODE[71] = r'''
# =========================================================
# 9.7 ABLATION CẤU HÌNH TÍN HIỆU & HUẤN LUYỆN [V7-A1]
#     Cùng mô hình (cố định trước), cùng phân hoạch, cùng seed; chỉ đổi MỘT yếu tố.
#     Chọn biến thể bằng AUC mức bản ghi trên validation-TRONG -> không thiên lệch.
# =========================================================
def record_stratified_splits(frames_df, K_, seed):
    rl = frames_df.groupby("record_id").label.first()
    r2f = frames_df.groupby("record_id").indices
    ids, yy = rl.index.values, rl.values.astype(int)
    K_ = int(min(K_, pd.Series(yy).value_counts().min()))
    skf = StratifiedKFold(n_splits=K_, shuffle=True, random_state=seed)
    ex = lambda sub: np.sort(np.concatenate([r2f[r] for r in sub]))
    return [(ex(ids[a]), ex(ids[b])) for a, b in skf.split(np.zeros(len(ids)), yy)], K_

class _FramesSwap:
    # Đổi tạm bộ frame toàn cục để tái dùng nguyên vẹn bộ máy huấn luyện.
    # Phải tính lại MỌI cấu trúc dẫn xuất từ chỉ số frame.
    _DERIVED = ("X_ALL", "frames", "y_all", "groups_all", "rec_lab", "rec_to_frames",
                "REC_IDS", "REC_Y", "FRAMES_BY_REC")
    def __init__(self, Xv, Fv):
        self.Xv, self.Fv = Xv, Fv
    def __enter__(self):
        g = globals()
        self.saved = {k: g.get(k) for k in self._DERIVED}
        _RESIDENT.clear()
        g["X_ALL"] = self.Xv; g["frames"] = self.Fv
        g["y_all"] = self.Fv.label.values.astype(np.float32)
        g["groups_all"] = self.Fv.record_id.values
        g["rec_lab"] = self.Fv.groupby("record_id").label.first()
        g["rec_to_frames"] = self.Fv.groupby("record_id").indices
        g["REC_IDS"] = g["rec_lab"].index.values
        g["REC_Y"] = g["rec_lab"].values.astype(int)
        g["FRAMES_BY_REC"] = self.Fv.groupby("record_id").indices
        return self.Fv
    def __exit__(self, *exc):
        g = globals()
        for k, v in self.saved.items():
            g[k] = v
        _RESIDENT.clear()
        return False

class _CfgSwap:
    # Đổi tạm các thuộc tính cfg (khôi phục kể cả khi có lỗi).
    def __init__(self, **kw):
        self.kw = kw
    def __enter__(self):
        self.saved = {k: getattr(cfg, k) for k in self.kw}
        for k, v in self.kw.items():
            setattr(cfg, k, v)
        return cfg
    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(cfg, k, v)
        return False

ABL_SPECS = {
    "no_fir":       dict(cfg=dict(aug_fir_p=0.0)),
    "rec_balanced": dict(cfg=dict(record_balanced_loss=True)),
    "sr16k":        dict(cfg=dict(target_sr=16000, n_fft=2048, hop_length=512, f_max=8000.0),
                         frames=dict()),
    "frame4s":      dict(frames=dict(frame_sec=4.0, hop_sec=2.0)),
    "frame1s":      dict(frames=dict(frame_sec=1.0, hop_sec=0.5)),
    "no_notch":     dict(frames=dict(notch=False)),
    "no_highpass":  dict(frames=dict(highpass=0.0)),
}

def _abl_scores(res):
    # điểm mức bản ghi (fold ngoài) + AUC mức bản ghi gộp trên validation-trong
    oof = assemble_oof(res, "val"); mk = ~np.isnan(oof)
    g = record_scores(np.where(mk)[0], y_all[mk], oof[mk], 0.5)[["record_id", "label", best_agg]]
    y_in, p_in, i_in, f_in = pool_inner(res)
    g_in = record_scores(i_in, y_in, p_in, 0.5, folds=f_in)
    auc_in = float(roc_auc_score(g_in.label, g_in[best_agg])) if g_in.label.nunique() > 1 else np.nan
    auc_fr = float(roc_auc_score(y_all[mk].astype(int), oof[mk]))
    return g.set_index("record_id"), auc_in, auc_fr, int(len(frames))

def run_variant(tag, model_name):
    spec = ABL_SPECS[tag]
    with _CfgSwap(**spec.get("cfg", {})):
        if "frames" in spec:
            fkw = spec["frames"]
            Xv, Fv = get_frames(tag, frame_sec=fkw.get("frame_sec"), hop_sec=fkw.get("hop_sec"),
                                highpass=fkw.get("highpass"), notch=fkw.get("notch"))
            with _FramesSwap(Xv, Fv):
                sp, _ = record_stratified_splits(frames, cfg.n_splits, cfg.seed)
                res = run_cv(model_name, splits=sp, seed_base=cfg.seed, tag=tag)
                out = _abl_scores(res)
            del Xv, Fv
        else:
            res = run_cv(model_name, splits=SPLITS, seed_base=cfg.seed, tag=tag)
            out = _abl_scores(res)
    for r in res:
        r["state"] = None
    del res; gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return out

if cfg.run_signal_ablation and cfg.ablation_variants:
    ABL_MODEL = cfg.ablation_model if cfg.ablation_model in cfg.models else BEST_MODEL
    print(f"Ablation trên {ABL_MODEL} | biến thể: {list(cfg.ablation_variants)}")
    # cấu hình gốc: TÁI DÙNG lần chạy chính (cùng phân hoạch SPLITS, cùng seed)
    _mk = ~np.isnan(OOF[ABL_MODEL])
    g_base = record_scores(np.where(_mk)[0], y_all[_mk], OOF[ABL_MODEL][_mk], 0.5) \
        .set_index("record_id")[["label", best_agg]]
    _gi = record_scores(*[INNER[ABL_MODEL][j] for j in (2, 0, 1)], 0.5, folds=INNER[ABL_MODEL][3])
    base_in = float(roc_auc_score(_gi.label, _gi[best_agg]))
    abl_rows = [dict(variant="base", n_frames=int(len(frames)), inner_auc_record=base_in,
                     outer_auc_record=float(roc_auc_score(g_base.label, g_base[best_agg])),
                     outer_auc_frame=float(roc_auc_score(y_all[_mk].astype(int), OOF[ABL_MODEL][_mk])),
                     dAUC_vs_base=0.0, dAUC_lo=np.nan, dAUC_hi=np.nan, p=np.nan)]
    for tag in cfg.ablation_variants:
        if tag not in ABL_SPECS:
            print(f"  !! bỏ qua biến thể không xác định: {tag}"); continue
        t0 = time.time()
        g_v, auc_in, auc_fr, n_fr = run_variant(tag, ABL_MODEL)
        common = g_base.index.intersection(g_v.index)
        yv = g_base.loc[common, "label"].values.astype(int)
        pv_, pb_ = g_v.loc[common, best_agg].values, g_base.loc[common, best_agg].values
        d = paired_cluster_delta_ci(common.values, yv, pv_, pb_, roc_auc_score,
                                    n=min(cfg.bootstrap_n, 1000), seed=0)
        abl_rows.append(dict(variant=tag, n_frames=n_fr, inner_auc_record=auc_in,
                             outer_auc_record=float(roc_auc_score(yv, pv_)), outer_auc_frame=auc_fr,
                             dAUC_vs_base=float(roc_auc_score(yv, pv_) - roc_auc_score(yv, pb_)),
                             dAUC_lo=d["lo"], dAUC_hi=d["hi"], p=d["p_two_sided"]))
        print(f"  [{tag}] {time.time()-t0:.0f}s | inner {auc_in:.4f} | outer {abl_rows[-1]['outer_auc_record']:.4f} "
              f"| Δ vs base {abl_rows[-1]['dAUC_vs_base']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]")
    abl_sig = pd.DataFrame(abl_rows)
    abl_sig["p_holm"] = np.r_[np.nan, holm(abl_sig.p.iloc[1:].fillna(1.0).values)] if len(abl_sig) > 1 else np.nan
    ABL_SELECTED = str(abl_sig.loc[abl_sig.inner_auc_record.idxmax(), "variant"])
    abl_sig["selected_by_inner"] = np.where(abl_sig.variant == ABL_SELECTED, "<<<", "")
    display(Markdown(f"### Table 9e. Signal/training ablation — {ABL_MODEL} (same partition and seed; "
                     f"record-level `{best_agg}`; Δ = paired cluster bootstrap vs base)"))
    display(abl_sig.round(4))
    abl_sig.round(6).to_csv(TAB_DIR / "table9e_signal_ablation.csv", index=False)
    print(f"\nBiến thể được validation-TRONG chọn: {ABL_SELECTED}")
    if ABL_SELECTED == "base":
        print("=> Cấu hình gốc vẫn là lựa chọn tốt nhất theo tiêu chí không thiên lệch.")
    else:
        _r = abl_sig.set_index("variant").loc[ABL_SELECTED]
        print(f"=> Đề xuất: dùng '{ABL_SELECTED}' (ΔAUC ngoài {_r.dAUC_vs_base:+.4f}, CI ghép cặp "
              f"[{_r.dAUC_lo:+.4f}, {_r.dAUC_hi:+.4f}]). Đổi cfg tương ứng rồi chạy lại TOÀN BỘ notebook;")
        print("   con số báo cáo của lần chạy đó vẫn trung thực vì lựa chọn chỉ dựa trên validation-trong.")
    _sig = abl_sig[(abl_sig.variant != "base") & ((abl_sig.dAUC_lo > 0) | (abl_sig.dAUC_hi < 0))]
    if len(_sig):
        print("Biến thể có ΔAUC ghép cặp KHÁC 0 (CI không chứa 0): " +
              ", ".join(f"{r.variant} ({r.dAUC_vs_base:+.3f})" for _, r in _sig.iterrows()))
    else:
        print("Không biến thể nào có ΔAUC ghép cặp khác 0 -> kết quả KHÔNG nhạy với các lựa chọn tiền xử lý"
              " này (đây là bằng chứng về độ vững, nên đưa vào bài báo).")
else:
    abl_sig, ABL_SELECTED = None, None
    print("Bỏ qua ablation (cfg.run_signal_ablation=False).")
'''

CODE[103] = r'''
# =========================================================
# 16.1 Bảng đối chiếu v6 -> v7
# =========================================================
changes = pd.DataFrame([
    ("B1", "Grad-CAM của CRNN",
     "RuntimeError 'cudnn RNN backward can only be called in training mode' -> notebook DỪNG ở 10.1;"
     " mục 10.2-16 (occlusion, t-SNE, bảng LaTeX, đoạn Results, gói triển khai) chưa từng chạy",
     "tắt cuDNN cục bộ khi tính gradient Grad-CAM, mô hình vẫn ở chế độ eval",
     "toàn bộ phần sau của pipeline phải chạy được"),
    ("B2", "Chọn checkpoint trong warm-up",
     "cấu hình khai báo min_epochs_after_lr_peak nhưng best được cập nhật từ epoch 0; "
     "log: 'chọn ep1', LeakCNN1D chọn trung bình epoch 0.8 (gần như chưa huấn luyện)",
     "chỉ cập nhật best khi ep >= SELECT_FROM_EPOCH (= đỉnh LR); làm mịn chỉ trên các epoch hợp lệ; "
     "Bảng 5 kiểm tra ràng buộc",
     "so sánh kiến trúc công bằng; LeakCNN1D không còn bị 'xử thua' do chọn epoch 0"),
    ("B3", "Tự chỉnh low_band_hz",
     "hạ 2800 -> 1000 Hz theo MỘT dải (283-413 Hz, p=0.034, mất ý nghĩa sau Holm), cắt dải 1.3-2.7 kHz "
     "mà Grad-CAM LeakCNN2D dùng (~2.5 kHz)",
     "giá trị đăng ký trước (2800 Hz); auto (tuỳ chọn) chỉ MỞ RỘNG để phủ top-3 dải",
     "không để một kiểm định nhiễu quyết định kiến trúc"),
    ("B4", "Mô tả SpecAugment của MultiBranchCNN",
     "văn bản: dùng chung cho 3 nhánh; mã: chỉ nhánh Mel. Mặt nạ chung cho cả batch",
     "mỗi nhánh một SpecAugment (MFCC: chỉ che thời gian); mặt nạ riêng cho từng mẫu",
     "mã và mô tả phải khớp"),
    ("B5", "DEPLOY_MODEL dùng trước khi định nghĩa",
     "mục 10.4a-e dùng DEPLOY_MODEL, chỉ được gán ở mục 13 -> NameError (bị che bởi B1)",
     "xác định ngay sau mục 8.5 bằng đúng tiêu chí của mục 13 (AUC bản ghi trên validation-trong)",
     "tránh lỗi âm thầm"),
    ("B6", "Ghi đè biến sel_df",
     "mục 13 ghi đè sel_df của mục 8.5", "đổi tên agg_sel_df / dep_sel_df", "vệ sinh mã"),
    ("B7", "Lọc baseline RMS trong đoạn Results",
     "lọc theo tên 'CHỈ mức năng lượng (RMS)' trong khi bảng dùng 'Level only (RMS)'",
     "kết luận lấy trực tiếp từ so sánh ghép cặp mục 9.5", "baseline 1 đặc trưng không bị chọn nhầm"),
    ("B8", "TransferCNN / LeakCNN1D",
     "ResNet18 nhận ảnh 64x63 -> bản đồ 2x2; LeakCNN1D nhận biên độ tuyệt đối (dùng được 'độ to')",
     "phóng log-Mel lên >=128 (bản đồ 4x4); LeakCNN1D chuẩn hoá theo mẫu như mô hình phổ",
     "so sánh kiến trúc công bằng, bớt kênh confound mức năng lượng"),
    ("S1", "Một phân hoạch, một seed",
     "mọi chênh lệch kiến trúc lẫn với nhiễu phân hoạch",
     "CV lặp (mặc định 3 phân hoạch); Bảng 5b: mean±sd và AUC của OOF trung bình qua lần lặp",
     "đo độ bất định mà phản biện chắc chắn hỏi"),
    ("S2", "Ngã rẽ phân tích ở mức bản ghi",
     "quy tắc gộp chọn trên validation-trong + Platt (làm AUC MultiBranch giảm 0.856 -> 0.801)",
     "p_mean trên xác suất thô, ĐĂNG KÝ TRƯỚC; các lựa chọn khác chỉ là phân tích độ nhạy",
     "giảm số quyết định dựa trên tập nhỏ"),
    ("S3", "CNN vs baseline cổ điển",
     "13 đặc trưng thủ công; kết luận bằng 'hai CI không chồng lấn' (0.8092 vs 0.8076)",
     "thêm thống kê log-Mel 128 chiều; CI bootstrap GHÉP CẶP của ΔAUC + Holm; CNN tham chiếu = mô hình "
     "chọn trên validation-trong",
     "kiểm định đúng câu hỏi 'CNN có thêm giá trị?'"),
    ("S4", "Kiểm định theo dải tần",
     "7 + 14 kiểm định không hiệu chỉnh; giả thuyết dải thấp kết luận từ hai số điểm (0.164 vs 0.194)",
     "Holm cho mọi họ kiểm định; CI bootstrap của hiệu số; kết luận 3 mức (có 'chưa phân định')",
     "phát biểu khớp sức mạnh bằng chứng"),
    ("S5", "Tỉ lệ thể hiện dương",
     "ngưỡng duy nhất 0.5; in 'trung vị 0.81 ~ 1'",
     "3 mức: <0.5 ủng hộ / 0.5-0.9 chưa kết luận / >=0.9 không ủng hộ", "không phóng đại"),
    ("A1", "Ablation",
     "tắt mặc định; chỉ độ dài frame + bộ lọc",
     "bật mặc định: no_fir, rec_balanced, sr16k, frame1s/4s, no_notch, no_highpass; chọn theo "
     "validation-trong; ΔAUC ghép cặp so với gốc",
     "trả lời 'vì sao chọn cấu hình này' bằng số liệu, không thiên lệch"),
    ("C1", "Thông tin điểm khảo sát",
     "chỉ cảnh báo", "tự sinh site_map_template.csv để điền", "hạ rào cản cho thí nghiệm leave-one-site-out"),
    ("D1", "Gán nhãn & dò dữ liệu",
     "khớp chuỗi con (khoá 'rr' khớp mọi tên có 'rr'); data_root sai -> âm thầm quét cả /kaggle/input/datasets",
     "khớp theo từ + tự kiểm tra; chọn thư mục sâu nhất chứa leak/noleak, cảnh báo rõ ràng",
     "tránh trộn dữ liệu / gán nhãn sai"),
    ("D2", "Số liệu cũ viết cứng",
     "văn bản còn '116 bản ghi', '8 801 frame', 'ICC 0,75', 'trung vị 0,99', 'mở rộng 2800 Hz'",
     "mọi con số in từ lần chạy; build_v7.py chặn tự động nếu còn số cũ",
     "không để số liệu sai lọt vào bản thảo"),
], columns=["id", "item", "v6 (defect)", "v7 (fixed)", "reason"])
display(Markdown("### Table 17 (internal). v6 to v7 changelog — not intended for the manuscript"))
display(changes)

# ---- Trạng thái tự kiểm tra của lần chạy này ----
print("\n=== TRẠNG THÁI LẦN CHẠY NÀY ===")
print(f"  - Dữ liệu                    : {'MÔ PHỎNG (synthetic)' if SYNTHETIC else DATA_ROOT}")
print(f"  - Đơn vị thống kê            : BẢN GHI (n={len(records)}); CI theo cụm")
print(f"  - Tiêu chí chọn epoch        : {cfg.select_metric}, chỉ từ epoch {SELECT_FROM_EPOCH}")
print(f"  - CV lặp                     : {len(cfg.seeds)} phân hoạch")
print(f"  - Quy tắc gộp mức bản ghi    : {best_agg} ({cfg.agg_selection}); Platt: {cfg.calibrate_aggregation}")
print(f"  - Mô hình triển khai         : {DEPLOY_MODEL} (chọn trên validation-trong)")
print(f"  - Baseline cổ điển           : {BASELINE_VERDICT_VI if cfg.run_baseline_models else 'không chạy'}")
print(f"  - Ablation                   : {'biến thể được chọn = ' + str(ABL_SELECTED) if ABL_SELECTED else 'không chạy'}")
print(f"  - Kiểm tra confound          : có site: {bool(_gv('HAS_SITE', False))} | "
      f"{(_gv('CONFOUND', {}) or {}).get('verdict', 'n/a')}")
print(f"  - Tiền đề MIL                : {_gv('WITNESS_VERDICT', 'n/a')} (trung vị {float(_gv('W_LEAK_MED', np.nan)):.2f})")
print(f"  - Giả thuyết dải thấp        : {_gv('BAND_NARRATIVE', 'n/a')} | CI hiệu số {BAND_TEST['ci']}")

print("\n=== VIỆC CÒN PHẢI LÀM TRƯỚC KHI NỘP BÀI (không thể tự động hoá) ===")
todo = [
 "1. Điền site_map_template.csv (vị trí/tuyến ống/ngày thu cho từng Rxxx), đặt cfg.site_source='csv' và "
 "cfg.run_loso_full=True: đây là thí nghiệm quan trọng nhất còn thiếu.",
 "2. Nghe và dán nhãn THỜI ĐIỂM rò rỉ trên ~10 bản ghi (2 người độc lập, báo cáo thoả thuận) để kiểm "
 "chứng trực tiếp tiền đề MIL.",
 "3. Nếu Bảng 9e đề xuất biến thể khác 'base': đổi cfg tương ứng và chạy lại toàn bộ notebook.",
 "4. Xác minh các tài liệu trong Bảng 11 trên Scopus (cột 'verified').",
 "5. Nếu định triển khai thời gian thực: thay chuẩn hoá đỉnh theo tệp bằng phương án NHÂN QUẢ và đánh giá lại.",
 "6. Notebook suy luận riêng (leak_detector_inference.ipynb) phải đọc manifest format_version 4 "
 "(trường mới: preprocessing.peak_norm_pct).",
]
for t in todo: print("  " + t)
print("\nGhi chú: mọi con số trong notebook này chỉ được trích dẫn kèm ĐƠN VỊ THỐNG KÊ và "
      "KHOẢNG TIN CẬY THEO CỤM.")
'''
