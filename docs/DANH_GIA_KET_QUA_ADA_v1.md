# Đánh giá phản biện kết quả ADA v1 trên `tapdulieuMix_16k` và lộ trình nâng cấp lên mức Q1/Q2

> Nguồn: notebook đã chạy trên Kaggle (2 × T4, 234.7 phút, 42/42 cell, pandas 2.3.3, torch 2.11),
> cấu hình mặc định của `leak-ada-dann-kaggle-v1` (chỉ vá tay cell 10.2). Mọi con số dưới đây lấy từ output của
> lần chạy đó, trừ mục 3.4.3 (thí nghiệm kiểm chứng bổ sung trên dữ liệu mô phỏng).

---

## 0. Tóm tắt điều hành

1. **Kết quả bề mặt hấp dẫn nhưng chưa đủ tin cậy để công bố.** Ở hướng `R → S`, DANN tăng AUC mức bản ghi
   0.735 → 0.861 (ΔAUC +0.126 [0.086, 0.169]), CDAN+E 0.852. Ở hướng `S → R`, **không phương pháp nào** cải thiện
   (DANN 0.611, CDAN+E 0.632 so với source-only 0.620). Chỉ AdaBN có xu hướng tăng (+0.080, Holm p = 0.064).
2. **Ba mối đe doạ tính hợp lệ (validity) phải xử lý trước khi viết bài:**
   - (V1) **Sai đơn vị độc lập ở chuỗi S**: 165 tệp `Sxx_leakNN` bị coi là 165 bản ghi độc lập. Hệ quả: rò rỉ giữa
     các tệp cùng phiên. Dấu hiệu: AUC nguồn-validation = **1.000 ở cả 63 job** `S → R`; oracle trên S = **1.000**;
     CI phía S hẹp giả tạo.
   - (V2) **Nghi vấn "lối tắt" (shortcut) trong chuỗi S**: S tách lớp hoàn hảo bằng đặc trưng cổ điển (AUC 1.000)
     nhờ các dải không mang tín hiệu rò rỉ (< 90 Hz, > 3 kHz), trong khi R chỉ đạt 0.841. Đây là lời giải thích hợp lý
     nhất cho việc `S → R` thất bại.
   - (V3) **Phần "đối nghịch" của DANN gần như không hoạt động**: bộ phân biệt miền ở mức may rủi ngay từ epoch 1;
     đặc trưng vẫn tách miền gần hoàn hảo (PAD 1.92); quét λ gần như phẳng. Phần lớn mức cải thiện nhiều khả năng đến
     từ việc **BatchNorm được huấn luyện trên lô trộn nguồn + đích**, không phải từ căn chỉnh đối nghịch.
3. **Có những phát hiện thật sự có giá trị khoa học**, nếu được kiểm chứng lại sau khi sửa V1–V3:
   - Tính bất đối xứng của chuyển miền (kiểm soát → hiện trường thất bại; hiện trường → kiểm soát thành công).
   - Dải 0.3–1.8 kHz là "chữ ký rò rỉ chuyển giao được"; dải < 300 Hz gây hại khi chuyển miền.
   - Tiêu chí chọn mô hình không nhãn **tương quan âm** với hiệu năng đích.
   - Augmentation kênh ngẫu nhiên có đóng góp đáng kể.
4. **Định vị bài báo Q1/Q2 khả thi nhất:** *"Điều gì thực sự tạo ra lợi ích của thích ứng miền trong nhận dạng rò rỉ
   âm học ống PVC? Một đánh giá không rò rỉ dữ liệu, cùng phương pháp thích ứng có cơ sở vật lý và nhất quán mức
   bản ghi"*. Bài báo có thể gồm hai đóng góp: (a) đóng góp phương pháp luận/cơ chế, (b) đóng góp phương pháp mới.
   Chi tiết ở mục 6.

---

## 1. Dữ liệu thực tế mà notebook đã phát hiện

| | Chuỗi **R** (notebook đặt tên `unnamed`) | Chuỗi **S** (notebook đặt tên `s`) |
|---|---|---|
| Tên tệp | `R001_leak01.wav` … | `S01_leak01.wav`, `S01_leak02.wav` … |
| Tần số lấy mẫu gốc | 16 kHz | **48 kHz** |
| Bản ghi / tệp | 80 bản ghi / 167 tệp (34 leak / 46 noleak) | 165 tệp được coi là 165 "bản ghi" (103 / 62) |
| Thời lượng | 123.7 phút, 25–189 s/bản ghi | 160.4 phút, ~60 s/tệp |
| Đồng nhất với tập v7? | **Trùng khớp hoàn toàn** với `field_rec_8k_V7.2` của v6/v7 (cùng 80 bản ghi, 167 tệp, 55.75 + 67.98 phút) | Dữ liệu mới |
| Tỉ lệ leak | 0.425 | 0.624 (dịch chuyển nhãn ~20 điểm %) |
| Trùng nội dung R↔S | Không (mục 3.6) | |

**Việc cần làm ngay:** đặt lại tên miền có nghĩa, ví dụ
`cfg.domain_alias = {"unnamed": "R_field16k", "s": "S_series48k"}`. Cũng cần ghi rõ **S là dữ liệu gì**: thu trên
mô hình thí nghiệm hay hiện trường mới; thiết bị và cảm biến; áp suất, lưu lượng, trạng thái bơm; kích thước lỗ rò;
khoảng cách cảm biến–lỗ rò. Không có các thông tin này thì không viết được mục *Data* của bài báo Q1.

---

## 2. Kết quả chính (trích từ lần chạy)

| Bài toán | Phương pháp | AUC bản ghi [CI 95%] | ΔAUC so với SO [CI] | Holm p | BAcc@0.5 | AUC nguồn-val |
|---|---|---|---|---|---|---|
| R → S | Source-only | 0.735 [0.646, 0.815] | – | – | 0.619 | 0.873 |
| | AdaBN | 0.771 | +0.036 [0.021, 0.053] | < 0.001 | 0.693 | 0.873 |
| | Deep CORAL | 0.801 | +0.067 [0.045, 0.090] | < 0.001 | 0.719 | 0.879 |
| | **DANN** | **0.861** [0.800, 0.916] | **+0.126** [0.086, 0.169] | < 0.001 | 0.737 | 0.860 |
| | CDAN+E | 0.852 | +0.117 [0.080, 0.157] | < 0.001 | 0.751 | 0.870 |
| | Oracle | 1.000 [1.000, 1.000] ⚠ | | | 1.000 | |
| S → R | Source-only | 0.620 [0.490, 0.742] | – | – | **0.515** | **1.000** ⚠ |
| | AdaBN | **0.700** | +0.080 [0.016, 0.153] | 0.064 | 0.587 | 1.000 |
| | Deep CORAL | 0.625 | +0.004 | 1.000 | 0.519 | 1.000 |
| | DANN | 0.611 | −0.009 [−0.117, 0.105] | 1.000 | 0.602 | 1.000 |
| | CDAN+E | 0.632 | +0.012 | 1.000 | 0.617 | 1.000 |
| | Oracle | 0.864 [0.778, 0.936] | | | 0.755 | |

Bổ sung (1 seed, bài toán R → S):

- **Quét λ:** 0.1 → 0.869; 0.3 → 0.891; 1.0 → 0.891; 3.0 → 0.880. Source-only cùng seed ≈ 0.73.
- **Ablation:** bỏ toàn bộ augmentation −0.085 [−0.125, −0.047]; bỏ FIR −0.041; bỏ cân bằng lớp −0.032;
  bỏ trọng số entropy của CDAN −0.007 (không có ý nghĩa).
- **Chọn mô hình:** quy tắc `last` tốt nhất trong các quy tắc không dùng nhãn. Chọn bằng nhãn đích (`oracle`) thổi
  phồng AUC trung bình +0.029, tối đa +0.091.

---

## 3. Phản biện chi tiết

### 3.1 [Nghiêm trọng – V1] Sai đơn vị độc lập ở chuỗi S → giả lặp (pseudo-replication) và rò rỉ

**Bằng chứng**

- Log mục 2.3b: *"165 tệp không khớp regex `(R\d{1,4})` → lấy tên tệp làm mã bản ghi"*. `S01_leak01` và `S01_leak02`
  thành hai "bản ghi" độc lập, trong khi với chuỗi R, `R001_leak01…04` được gộp đúng thành một bản ghi.
- **AUC nguồn-validation = 1.000 ở cả 63 job có nguồn là S**, ngay từ epoch 2 (Hình 16). Với nguồn là R (phân nhóm
  đúng), con số tương ứng chỉ khoảng 0.70–0.80.
- Bộ phân lớp cổ điển trong S (CV phân tầng theo "bản ghi"): AUC **1.000**. Trong R: 0.841.
- Oracle trên S: AUC đích **1.000 ở cả 15 job**, đạt 1.0 từ epoch 4.
- t-SNE (Hình 19): frame đích S tạo thành **các cụm rất chặt theo từng tệp**, tức "dấu vân tay phiên thu" rất mạnh.

**Hệ quả**

1. CI cho bài toán có đích S **hẹp hơn thực tế**: nếu mỗi `Sxx` gồm 2 tệp thì số đơn vị độc lập ≈ 83 chứ không phải
   165, và độ rộng CI phải tăng khoảng √2 ≈ 1.4 lần. Các p-value "< 0.001" ở R → S vì vậy lạc quan.
2. Ở R → S, **tệp anh em** của tệp đang kiểm thử (cùng `Sxx`) nằm trong tập đích không nhãn dùng để căn chỉnh. Đây
   là một dạng rò rỉ *transductive* nhẹ: không có nhãn, nhưng mô hình đã thấy chính phiên thu đó.
3. Oracle = 1.000 do học thuộc phiên thu, nên "gap closed 47%" **không còn ý nghĩa**.
4. Ở S → R, AUC nguồn-validation luôn bằng 1.000 nên nguồn-validation **mất hoàn toàn vai trò** chọn mô hình hay
   chọn ngưỡng.

**Sửa (bắt buộc):** đặt `cfg.record_regex = r"([RS]\d{1,4})"`. Nếu cùng một `Sxx` xuất hiện **cả ở leak lẫn noleak**,
thì `Sxx` là **điểm đo/kịch bản**, không phải bản ghi. Khi đó:
- phải gom nhóm cross-validation theo `Sxx` (cả hai lớp vào cùng fold), dùng StratifiedGroupKFold;
- cặp leak/noleak tại cùng điểm đo là **thiết kế ghép cặp rất giá trị**: nó loại bỏ nhiễu do vị trí, và đúng là
  thí nghiệm mà v7 nói còn thiếu.

### 3.2 [Nghiêm trọng – V2] Chuỗi S có dấu hiệu "lối tắt" (shortcut)

**Bằng chứng**

- AUC theo dải **trong S** (Bảng 3):
  - dải thấp: 20–33 Hz 0.67, 33–54 Hz **0.82**, 54–89 Hz 0.72;
  - dải trung bị đảo (noleak to hơn): 147–659 Hz 0.18–0.34;
  - dải cao: 2.9–4.9 kHz 0.78, 4.9–8 kHz **0.82**.
  Trong R, mọi dải đều nằm trong khoảng 0.40–0.61 và không dải nào có ý nghĩa.
- Spectrogram S (Hình 6) có **các vạch âm sắc nằm ngang cố định** (động cơ/bơm) ở cả hai lớp.
- Bộ phân lớp cổ điển trong S đạt 1.000. Nếu S thật sự đo "rò rỉ" ở điều kiện thực, điều này khó xảy ra, vì chính
  tập R chỉ đạt 0.84.

**Giả thuyết cần kiểm chứng:** bản ghi noleak của S được thu ở **trạng thái thủy lực khác** (không có dòng chảy,
bơm tắt hoặc chạy khác, thời điểm khác, mức khuếch đại khác). Nếu đúng, mô hình học được "có dòng chảy / không có
dòng chảy" chứ không phải "có rò rỉ". Phản biện Q1 gần như chắc chắn sẽ hỏi điều này.

**Kiểm chứng đề xuất** (rẻ, làm được ngay trong notebook):

1. Phân lớp S **chỉ bằng dải 20–100 Hz**, rồi **chỉ bằng dải > 3 kHz**: nếu AUC ≈ 1 thì lối tắt được xác nhận.
2. Phân lớp S **chỉ bằng RMS/mức tín hiệu** trước chuẩn hoá.
3. So sánh trong cùng `Sxx` (nếu có cặp leak/noleak): nếu AUC trong cặp thấp hơn hẳn AUC giữa các cặp, mô hình đang
   phân biệt **điểm đo** chứ không phải rò rỉ.

### 3.3 [Quan trọng] Dịch chuyển chuỗi thu ở **dải thấp** chưa được hoà hợp

- PSD (Hình 3): R **suy giảm mạnh dưới ~100 Hz** (khoảng −55 dB tại 25 Hz), trong khi S vẫn mạnh (khoảng −12 dB).
  Hai miền chênh nhau tới ~40 dB, đây là dấu vết của bộ lọc thông cao trong chuỗi thu R (cảm biến hoặc máy ghi).
- Quy tắc hoà hợp băng thông của v1 chỉ xét **phía cao** (năng lượng 99.5%) nên đã kết luận
  "Băng thông các miền tương đương → giữ f_max = 8000 Hz". Đây là **thiếu sót thiết kế** của notebook.
- Occlusion (Hình 24): che dải **48–278 Hz làm AUC đích *tăng*** ở mọi phương pháp thích ứng (−0.03 đến −0.08).
  Dải này có hại khi chuyển miền.

**Sửa:** hoà hợp hai phía. Lấy `f_min` = tần số mà miền có cắt thông cao đã suy giảm > 10 dB (ở đây ~100–150 Hz),
và `f_max` ≈ 7.5 kHz. Lý tưởng hơn, tạo augmentation **ngẫu nhiên hoá chuỗi thu**: thông cao ngẫu nhiên 20–200 Hz,
độ dốc phổ ±6 dB/octave, FIR ngẫu nhiên (mục 5.2.4).

### 3.4 [Quan trọng – V3] Căn chỉnh đối nghịch thực chất **không hoạt động**; lợi ích đến từ đâu?

**3.4.1 Bằng chứng từ lần chạy**

| Quan sát | Số liệu | Diễn giải |
|---|---|---|
| Mất mát và độ chính xác bộ phân biệt (Hình 15) | L_d ≡ ln 2, acc ≈ 0.50 **từ epoch 1**, khi λ_p mới ≈ 0.17 | Bộ phân biệt chưa bao giờ "thắng"; trò chơi đối nghịch không có lực đẩy |
| PAD đặc trưng (Bảng 14) | SO 1.89, DANN **1.92**, CDAN+E 1.88, CORAL 1.86 | Đặc trưng **vẫn tách miền gần hoàn hảo** sau DANN |
| MMD (Bảng 14) | SO 0.142 → DANN 0.068 → CDAN+E 0.043 | Chỉ các mômen toàn cục được kéo gần |
| Silhouette lớp đích | SO 0.08 → DANN 0.17 | Cấu trúc lớp ở đích tốt lên |
| Quét λ (seed 42) | λ = 0.1 cho 0.869 so với λ = 1 cho 0.891; SO ≈ 0.73 | **~85% lợi ích đã có ở λ = 0.1**, khi gradient đảo gần như bằng 0 |
| Deep CORAL | loss căn chỉnh ≈ 0.005 (Hình 15b), vẫn +0.067 AUC | Một hàm phạt gần như bằng 0 vẫn tạo lợi ích lớn |
| AdaBN (sau huấn luyện) | +0.036 | Thống kê BN giải thích một phần |

**3.4.2 Giả thuyết cơ chế:** mọi phương pháp dùng đích (CORAL, DANN, CDAN) đều đưa **lô nguồn + đích vào cùng một lượt
truyền**. Vì vậy BatchNorm được *huấn luyện* với thống kê trộn hai miền ("joint-BN"): mạng học đặc trưng **dưới
chuẩn hoá có biết đích**. AdaBN chỉ ước lượng lại BN *sau* huấn luyện nên được ít hơn. Giả thuyết này giải thích cùng
lúc: CORAL có lợi dù loss ≈ 0, quét λ phẳng, và PAD không giảm.

**3.4.3 Kiểm chứng bổ sung (dữ liệu mô phỏng, CPU):** huấn luyện DANN với λ = 0. Độ chính xác bộ phân biệt tăng
0.51 → 0.76–0.80, nên bộ phân biệt **không lỗi** và học được. Với λ = 1, nó bị giữ ở ≈ 0.5 suốt quá trình, trong khi
PAD tuyến tính trên đặc trưng ở chế độ eval vẫn bằng 2.0. Kết luận: bộ trích đặc trưng chỉ cần **"lừa" một bộ phân
biệt yếu** (dropout 0.5 × 2, lr ×1, lô nhiễu do augmentation), mà không thật sự tạo ra đặc trưng bất biến miền.

**Hệ quả cho bài báo:** không được viết "DANN cải thiện nhờ học đặc trưng bất biến miền" khi PAD không giảm. Cần
**thí nghiệm đối chứng cơ chế** (mục 5.1.4). Nếu joint-BN với λ = 0 đạt gần DANN, thì đây là **một phát hiện đáng
công bố**: trong nhận dạng rò rỉ âm học, lợi ích UDA chủ yếu đến từ chuẩn hoá có biết đích, không phải từ căn chỉnh
đối nghịch.

### 3.5 [Quan trọng] Bất đối xứng và chỉ có hai bài toán

- **R → S** thành công, **S → R** thất bại. Theo cận Ben-David (ε_T ≤ ε_S + ½d + λ\*), ở S → R sai số nguồn ≈ 0
  nhưng **λ\* (sai số của giả thuyết lý tưởng chung) lớn**: đặc trưng mạnh nhất trong S (< 90 Hz, > 3 kHz) không
  mang thông tin trong R. Không phương pháp căn chỉnh biên nào khắc phục được điều này.
- Chiều ngược lại, R là miền khó (trần ≈ 0.84–0.86: CV cổ điển 0.841, oracle 0.864, v7 ≈ 0.85). Đặc trưng học từ
  R tổng quát hơn, và S lại "dễ", nên R → S hưởng lợi.
- Với **2 miền / 2 bài toán**, mọi kết luận "DA hiệu quả" chỉ là **một điểm dữ liệu**. Bài báo Q1 cần ≥ 4–6 bài
  toán chuyển miền, hoặc một benchmark công khai bổ sung (mục 5.3.1).

### 3.6 [Trung bình] Thống kê

- p = 0.000 phải ghi là **p < 0.0005**, vì B = 2000 lần bootstrap không cho phép p nhỏ hơn mức đó.
- 3 seed cho độ lệch chuẩn lớn: SO R → S 0.732 ± 0.076; DANN 0.843 ± 0.064. **Quét λ và ablation chỉ 1 seed** nên
  các kết luận Holm p = 0.045 (bỏ FIR, bỏ cân bằng lớp) **mong manh**. Cần ≥ 3 seed.
- Holm hiện chỉ áp trong từng bài toán. Khi có nhiều bài toán, cần thêm kiểm định tổng hợp (Friedman + Nemenyi,
  biểu đồ critical-difference) theo Demšar (2006).
- Bootstrap cần **phân tầng hai cấp** (phiên/điểm đo → tệp) sau khi sửa V1.

### 3.7 [Trung bình] Định nghĩa oracle và các chỉ số tham chiếu

- Oracle hiện tại vừa bị rò rỉ (V1) vừa trộn nguồn + đích. Nên báo cáo **hai cận trên**: *target-only* (CV nhóm
  đúng trên đích) và *source+target* (nhóm đúng).
- "Gap closed" chỉ có ý nghĩa sau khi oracle hợp lệ.

### 3.8 [Phát hiện tích cực] Chọn mô hình không có nhãn đích

- Tương quan Spearman trong từng lượt huấn luyện giữa tiêu chí và AUC đích:
  - −BCE nguồn-val: **−0.22 đến −0.39**, tức validation trên nguồn càng tốt thì đích càng kém (quá khớp miền nguồn);
  - DEV ≈ 0; PAD ≈ 0; entropy đích +0.16 đến +0.25 (độ lệch chuẩn ≈ 0.55).
- `last` tốt nhất trong các quy tắc hợp lệ. Mức lạc quan khi chọn bằng nhãn đích: +0.015 đến +0.091.
- Đây là bằng chứng định lượng cho nhận định "**validation trên miền nguồn đánh lừa khi có dịch chuyển miền**",
  một đóng góp phương pháp luận đáng đưa vào bài báo.

### 3.9 [Trung bình] Ngưỡng và hiệu chỉnh xác suất

- SO S → R: BAcc@0.5 = 0.515, F1 = 0.057, tức gần như dự đoán toàn bộ là noleak. Đây là dịch chuyển xác suất tiên
  nghiệm và hiệu chỉnh. DANN/CDAN+E **không tăng AUC nhưng tăng BAcc lên 0.60–0.62**: thích ứng cải thiện
  **hiệu chỉnh** chứ không cải thiện **thứ hạng**. Nên báo cáo rõ hai khía cạnh này.
- Cho triển khai: cần ước lượng tỉ lệ lớp đích (EM của Saerens et al., 2002; BBSE của Lipton et al., 2018) để đặt
  ngưỡng, và báo cáo ECE.

### 3.10 [Nhẹ] PAD bão hoà nên không phân biệt được các phương pháp

PAD 1.86–1.98 cho mọi biểu diễn, kể cả oracle. Nên bổ sung: AUC của bộ phân lớp miền có kiểm soát dung lượng, MMD
theo lớp, khả năng tách lớp tuyến tính ở đích (linear probe), và độ chính xác bộ phân biệt đo **ở chế độ eval trên
tập giữ lại**.

### 3.11 Điểm mạnh nên giữ

Giao thức quy nạp (sau khi sửa V1); không trùng nội dung R ↔ S; thống kê mức bản ghi; quy tắc chọn đăng ký trước;
ablation cho thấy augmentation quan trọng; λ ổn định; occlusion cho thấy thích ứng **dời sự chú ý sang 0.3–1.8 kHz**
(DANN: che 682–1107 Hz làm AUC giảm 0.16, source-only chỉ giảm 0.06). Kết hợp với bảng AUC theo dải (dải 1–3 kHz,
leak cao hơn, và dải 250–650 Hz, noleak cao hơn, **cùng chiều ở cả hai miền**), đây là **chữ ký rò rỉ chuyển giao
được**, có thể đối chiếu với vật lý suy giảm sóng trong ống PVC.

---

## 4. Tổng hợp cơ chế

```
            S (48 kHz, kiểm soát?)                              R (hiện trường 16 kHz)
  đặc trưng mạnh: <90 Hz, >3 kHz, vạch âm sắc        đặc trưng yếu, phân tán; cắt thông cao <100 Hz
  ── có thể là lối tắt / dấu vân tay chuỗi thu ──    ── tín hiệu rò rỉ thật nhưng SNR thấp ──
                 │                                                   │
   S → R: học lối tắt (src-val = 1.0) → không tồn tại ở R → λ* lớn → mọi DA thất bại
   R → S: học đặc trưng chung 0.3–1.8 kHz → chuyển được; joint-BN + DA kéo thống kê → +0.13 AUC
```

---

## 5. Đề xuất nâng cấp (notebook v2) theo mức ưu tiên

### 5.1 P0 – Bắt buộc trước khi viết bài (sửa tính hợp lệ)

| # | Việc | Cách làm | Kỳ vọng |
|---|---|---|---|
| 1 | Định danh bản ghi S đúng | `record_regex = r"([RS]\d{1,4})"`; nếu `Sxx` chứa cả hai lớp → nhóm CV theo điểm đo (StratifiedGroupKFold), thêm `group_regex` tách biệt với `record_regex` | AUC nguồn-val S < 1.0; oracle S < 1.0; CI rộng ra đúng mức |
| 2 | Metadata và tên miền | CSV `file, domain, site, pressure, leak_size, sensor, flow_state`; `domain_alias` | Viết được mục *Data*; mở ra nhiều bài toán chuyển miền |
| 3 | Kiểm chứng lối tắt S | Phân lớp theo dải 20–100 Hz, > 3 kHz, RMS; so sánh trong cặp và giữa các cặp | Xác nhận hoặc bác bỏ V2, quyết định vai trò của S |
| 4 | Hoà hợp chuỗi thu hai phía | `f_min` từ PSD (~100–150 Hz), `f_max` 7.5 kHz; quy tắc tự động dùng cả phía thấp | Loại dấu hiệu miền tầm thường; dự kiến cải thiện S → R |
| 5 | Đối chứng cơ chế | Phương pháp mới **`joint_bn`** (λ = 0, lô trộn) và **DSBN** (BN riêng từng miền); bộ phân biệt **mạnh hơn** (lr ×10 như bản gốc CDAN, dropout 0.1, rộng 1024), đo acc/PAD của bộ phân biệt **ở chế độ eval** | Tách bạch đóng góp của chuẩn hoá và căn chỉnh đối nghịch |
| 6 | Thống kê | ≥ 3 seed cho mọi thí nghiệm phụ; bootstrap hai cấp; ghi p < 0.0005 | Kết luận ablation vững |

### 5.2 P1 – Nâng đóng góp phương pháp (để có "điểm nhấn")

1. **Self-training nhất quán mức bản ghi (Record-Consistent Self-Training, RCST).** Ý tưởng then chốt và đặc thù cho
   bài toán: mọi frame của một bản ghi đích **chung một nhãn** (dù chưa biết nhãn đó). Vì vậy có thể:
   - lấy **nhãn giả mức bản ghi** từ trung bình xác suất các frame (tin cậy hơn nhiều so với nhãn giả từng frame),
     chọn bản ghi tự tin theo ngưỡng tăng dần và cân bằng lớp, rồi gán nhãn đó cho mọi frame của bản ghi;
   - thêm **hàm phạt nhất quán trong bản ghi**: KL giữa dự đoán từng frame và trung bình của bản ghi.
   Cách này khai thác cấu trúc túi (MIL) vốn có của dữ liệu âm thanh mà các phương pháp UDA tổng quát bỏ qua. Ràng
   buộc: chỉ dùng trên `tgt_train`, nên giao thức quy nạp vẫn được giữ.
2. **Căn chỉnh đối nghịch mức bản ghi:** bộ phân biệt nhận **embedding bản ghi** (attention-MIL gộp K frame của một
   bản ghi) thay cho từng frame. Cách này giảm ảnh hưởng của "dấu vân tay phiên" thấy rõ trong t-SNE.
3. **Augmentation ngẫu nhiên hoá chuỗi thu có cơ sở vật lý** (domain generalization), mở rộng FIR (đã chứng minh
   +0.041):
   - thông cao ngẫu nhiên 20–200 Hz, nhắm trực tiếp vào phát hiện 3.3;
   - độ dốc phổ ±6 dB/octave;
   - bộ lọc suy giảm theo khoảng cách trong ống PVC, dạng exp(−α·f·d);
   - trộn nền: chèn nhiễu từ bản ghi noleak của miền kia ở SNR ngẫu nhiên (cross-domain background mixing).
4. **Baseline hiện đại (2019–2024)** để phản biện không chê "so với baseline cũ":
   - MCC (Minimum Class Confusion, Jin et al., 2020);
   - SHOT hoặc TENT: không cần dữ liệu nguồn / thích ứng lúc kiểm thử, **sát với triển khai thực tế** cho công ty
     cấp nước;
   - tự huấn luyện kiểu FixMatch/AdaMatch;
   - tuỳ chọn: SDAT hoặc MDD.
   AdaBN đã cho tín hiệu tốt nhất ở S → R, nên DSBN và TENT rất đáng thử.
5. **Xử lý dịch chuyển nhãn:** ước lượng tỉ lệ lớp đích (EM/BBSE) để đặt ngưỡng; IW-CDAN (Tachet des Combes et al.,
   2020).

### 5.3 P2 – Mở rộng thực nghiệm cho Q1

1. **Thêm miền:** tách S theo điểm đo, áp suất, kích thước lỗ rò, cảm biến (nếu có metadata); tách R theo điểm khảo
   sát (v7 đã chỉ ra đây là thí nghiệm còn thiếu); bổ sung một tập công khai về âm/rung rò rỉ đường ống (Mendeley
   Data, IEEE DataPort; cần kiểm tra giấy phép và cảm biến). Mục tiêu: **≥ 4–6 bài toán chuyển miền**,
   leave-one-domain-out, biểu đồ critical-difference.
2. **Đường cong bán giám sát** (1/2/4/8 bản ghi đích có nhãn mỗi lớp): thông điệp thực tiễn kiểu "chỉ cần k bản ghi
   hiện trường có nhãn là đạt X".
3. **Chỉ số vận hành:** ECE và đường độ tin cậy; tỉ lệ báo động giả mỗi ngày ở ngưỡng đã chọn; thời lượng ghi tối
   thiểu để quyết định (AUC theo số giây/bản ghi); độ trễ và kích thước mô hình trên thiết bị biên.
4. **Diễn giải đối chiếu vật lý:** occlusion và Grad-CAM trên mọi bài toán; đối chiếu dải 0.3–1.8 kHz với lý thuyết
   suy giảm sóng trong ống PVC (Brennan, Muggleton, Gao). Đây là "điểm nhấn" liên ngành rất được tạp chí Q1 chuyên
   ngành đánh giá cao.

### 5.4 Ma trận thí nghiệm v2 và ngân sách

| Giai đoạn | Nội dung | Số job (ước tính) | Thời gian 2 × T4 |
|---|---|---|---|
| A – Hợp lệ | P0.1–P0.4; SO, AdaBN, CORAL, DANN, CDAN+E, oracle ×2; 3 seed | ~130 | ~2.5 h |
| B – Cơ chế | joint-BN, DSBN, DANN-strong-disc, kiểm chứng lối tắt; 3 seed | ~90 | ~1.8 h |
| C – Phương pháp mới | RCST, MIL-adversarial, augmentation chuỗi thu, MCC, TENT/SHOT, ablation 3 seed | ~200 | ~4 h |
| D – Mở rộng | thêm miền, SSDA, 5 seed cho bảng chính | tuỳ dữ liệu | 1–2 phiên |

Đồ thị theo epoch cho thấy AUC đích ổn định từ epoch ~20, nên có thể giảm `epochs` 30 → 20 để tiết kiệm 1/3 thời
gian. Có thể chia thành nhiều phiên Kaggle nhờ `ada_results.pkl` (lưu ra dataset để chạy tiếp).

---

## 6. Định vị bài báo và các đóng góp

**Tên đề xuất (chọn sau khi có kết quả v2):**
- *"What Drives Domain Adaptation Gains in Acoustic Leak Detection? A Leakage-Free, Recording-Level Study on
  Low-Pressure PVC Pipes"*
- *"Physics-Informed, Recording-Consistent Domain Adaptation for Cross-Site Acoustic Leak Detection in Low-Pressure
  PVC Water Pipes"*

**Đóng góp dự kiến (mỗi đóng góp gắn với bằng chứng):**

1. **Giao thức đánh giá UDA không rò rỉ dữ liệu** (quy nạp, nhóm theo phiên/điểm đo, chọn mô hình không nhãn đích),
   kèm định lượng mức lạc quan của các giao thức thường gặp: frame-split, transductive, chọn bằng nhãn đích.
2. **Phân tích cơ chế:** tách đóng góp của chuẩn hoá (joint-BN/AdaBN/DSBN) và căn chỉnh đối nghịch; chỉ ra rằng
   "lừa được bộ phân biệt" ≠ "bất biến miền" (PAD không giảm).
3. **Giải thích bất đối xứng** bằng lối tắt và dịch chuyển có điều kiện, có kiểm chứng theo dải tần; khuyến nghị
   thực hành: huấn luyện từ dữ liệu hiện trường, không từ dữ liệu kiểm soát.
4. **Phương pháp mới:** tự huấn luyện nhất quán mức bản ghi + căn chỉnh mức bản ghi + ngẫu nhiên hoá chuỗi thu có cơ
   sở vật lý, vượt các baseline 2019–2024 trên ≥ 4 bài toán.
5. **Diễn giải vật lý:** dải 0.3–1.8 kHz là chữ ký rò rỉ chuyển giao được trong ống PVC áp thấp.

**Tạp chí phù hợp (cần kiểm tra quartile JCR/SJR của năm nộp):** *Mechanical Systems and Signal Processing*,
*Applied Acoustics*, *Measurement*, *Engineering Applications of Artificial Intelligence*, *Structural Health
Monitoring* (Q1); *Journal of Pipeline Systems Engineering and Practice*, *Sensors*, *Water*, *Urban Water Journal*
(Q1/Q2 tuỳ năm).

---

## 7. Cây quyết định sau giai đoạn A–B

- **Nếu sau khi sửa V1, AUC trong S vẫn ≈ 1 và kiểm chứng lối tắt dương tính** → S không phải benchmark rò rỉ hợp lệ.
  Dùng S làm "miền kiểm soát" để minh hoạ rủi ro lối tắt, và lấy R (cùng các miền hiện trường khác) làm trọng tâm.
- **Nếu joint-BN ≈ DANN** → trọng tâm bài báo là phát hiện cơ chế (đóng góp 2) cùng phương pháp dựa trên chuẩn hoá và
  nhất quán bản ghi (RCST + DSBN/TENT).
- **Nếu DANN mạnh (bộ phân biệt mạnh hơn) > joint-BN rõ rệt và PAD giảm** → giữ câu chuyện đối nghịch, bổ sung căn
  chỉnh mức bản ghi.
- **Nếu hoà hợp dải thấp và augmentation chuỗi thu cải thiện S → R** → đây là bằng chứng mạnh cho đóng góp có cơ sở
  vật lý.

---

## 8. Danh mục kiểm tra trước khi nộp

- [ ] Đơn vị độc lập đúng cho mọi miền (không còn `fallback:stem`), AUC nguồn-val < 1.0
- [ ] Metadata miền và giao thức thu S được mô tả đầy đủ; kiểm chứng lối tắt đã báo cáo
- [ ] Hoà hợp chuỗi thu cả dải thấp lẫn dải cao
- [ ] Đối chứng joint-BN / DSBN / bộ phân biệt mạnh; PAD hoặc tiêu chí thay thế đo ở chế độ eval
- [ ] ≥ 4 bài toán chuyển miền; ≥ 3 seed (bảng chính 5 seed); bootstrap hai cấp; Friedman/Nemenyi
- [ ] Baseline 2019–2024 (MCC, TENT/SHOT, tự huấn luyện)
- [ ] Oracle hợp lệ (target-only và S+T nhóm đúng)
- [ ] Báo cáo cả xếp hạng (AUC) lẫn hiệu chỉnh và ngưỡng (BAcc, ECE, ước lượng tỉ lệ lớp)
- [ ] Văn bản tự sinh chỉ dùng sau khi mọi mục trên đã đạt

---

## 9. Trạng thái triển khai trong notebook v2 (`notebooks/leak-ada-dann-kaggle-v2.ipynb`)

Thông tin bổ sung từ tác giả dữ liệu: chuỗi S được thu trên **mô hình thí nghiệm** mô phỏng hệ thống ống gần với thực tế
(áp suất, loại ống, lưu lượng); khi thu bản ghi noleak **nước vẫn chảy**; số `Sxx` được đánh lại trong mỗi lớp và
`Sxx_leak` / `Sxx_noleak` **không** thu tại cùng một điểm đo. Hệ quả:

- Giả thuyết "không có dòng chảy" (mục 3.2) bị **bác bỏ**. Giả thuyết còn lại hợp lý nhất: các dấu hiệu < 90 Hz và vạch
  âm sắc của S là **đáp ứng thủy lực của mô hình thí nghiệm khi có rò** (rò làm tăng lưu lượng → đổi điểm làm việc của
  bơm → đổi âm sắc/nhiễu dòng chảy tần thấp). Hiện tượng này có thật trong vòng kín có bơm riêng nhưng hầu như không
  xảy ra ở mạng hiện trường, nên **không chuyển giao được**. Mục 3.9 của v2 định lượng giả thuyết này.
- Mã bản ghi đúng: `Sxx` **ghép với lớp** (`S01_leak01` + `S01_leak02` = một bản ghi; `S01_leak` ≠ `S01_noleak`).
  Đây là lựa chọn an toàn: nếu các tệp thật ra độc lập, việc gộp chỉ làm CI bảo thủ hơn, không gây chệch.

| Đề xuất | Trạng thái v2 |
|---|---|
| P0.1 Định danh bản ghi S | ✅ `record_regex = ([RS]\d{1,4})`, `record_id_by_class = True`; miền đặt tên `R_field` / `S_lab` |
| P0.2 Metadata | ⏳ cần tác giả bổ sung (điểm đo, áp suất, kích thước lỗ rò, cảm biến) để mở thêm bài toán chuyển miền |
| P0.3 Kiểm chứng lối tắt | ✅ mục 3.9: AUC trong miền và chéo miền cho từng vùng tần số |
| P0.4 Hoà hợp chuỗi thu hai phía | ✅ mục 3.2: so PSD trung bình đã căn mức (không dùng nhãn), dung sai 6 dB, trần f_min 300 Hz |
| P0.5 Đối chứng cơ chế | ✅ `joint_bn`, bảng phân rã cơ chế (mục 9.2b), bộ phân biệt 1024 / lr ×10, độ chính xác bộ phân biệt ở chế độ eval, ablation `weak_disc` |
| P0.6 Thống kê | ✅ ablation 3 seed, p ghi `< 1/B`, Holm chỉ trên phương pháp UDA |
| P1.1 RCST (nhãn giả mức bản ghi + nhất quán trong bản ghi) | ✅ `rcst` + ablation `rcst_no_pl`, `rcst_no_cons`; theo dõi độ chính xác nhãn giả mức bản ghi so với mức frame |
| P1.2 Căn chỉnh đối nghịch mức bản ghi (MIL) | ⏳ chưa làm (bước tiếp theo nếu RCST cho kết quả tốt) |
| P1.3 Ngẫu nhiên hoá chuỗi thu | ✅ thông cao 20–200 Hz, độ dốc ±3 dB/oct, thông thấp 3.5–8 kHz; ablation `no_chain_aug`, `so_no_chain_aug` |
| P1.4 Baseline hiện đại | ✅ MCC, TENT; ⏳ SHOT, FixMatch/AdaMatch |
| P1.5 Dịch chuyển nhãn (EM/BBSE, IW-CDAN) | ⏳ chưa làm |
| P2 Thêm miền, SSDA, chỉ số vận hành | ⏳ SSDA có sẵn (`cfg.run_ssda=True`); thêm miền cần metadata |

### 9.1 Bản v2.1 — vận hành lượt chạy dài trên Kaggle

Báo cáo từ lần chạy v2 đầu tiên: mục 8 in 2/246 job (source-only, 93–94 s/job — nhanh hơn v1 do v2 rút còn 20 epoch)
rồi "không thấy đáp ứng". Mã huấn luyện đa luồng giống v1 (v1 đã chạy trọn 126 job), nên nguyên nhân nhiều khả năng là
phiên tương tác ngừng nhận output, không phải treo. Tuy nhiên v2 không có cách nào phân biệt hai trường hợp, và
một số điểm vận hành cần sửa:

| Vấn đề | Sửa trong v2.1 |
|---|---|
| Không phân biệt được "đang chạy" với "treo"; output chỉ đến từ luồng GPU | Luồng chính in nhịp tim `⏱` (job + epoch từng GPU); `run_log.txt` có dấu thời gian; stack trace mọi luồng khi một GPU không tiến triển quá `cfg.stall_min` phút |
| Ngắt ô để lại luồng GPU chạy ngầm (luồng daemon), chạy lại ô -> 4 job tranh 2 GPU | Sự kiện dừng: hai luồng thoát ở vòng lặp kế tiếp, kết quả đã xong được lưu |
| Lưu kết quả mỗi 10 job (mất tới ≈ 12 phút công khi phiên dừng) | Lưu theo thời gian (`cfg.save_every_min`, mặc định 3 phút) |
| Không chạy tiếp được ở phiên Kaggle mới; đổi `seeds` làm mất mọi job cũ (dấu vân tay gồm cả phạm vi chạy) | Dấu vân tay chỉ gồm thứ làm đổi kết quả một job; tự tìm `ada_results.pkl` trong `/kaggle/input` -> chạy theo giai đoạn |
| `set_seed()` trong job gọi `manual_seed_all` -> đặt lại RNG của GPU kia giữa job đang chạy (không tái lập) | Gieo riêng RNG CPU + RNG của đúng GPU đó |
| Ablation/quét λ với seed không có trong `cfg.seeds` -> `KeyError` (thiếu phân hoạch) và so sánh ghép cặp lệch seed | Chỉ dùng seed có trong `cfg.seeds` |
| Ước lượng 4.5–5 giờ ở bản v2 hơi thấp | ≈ 324 job, ≈ 5–6 giờ trên 2 × T4 (≈ 1.5–2 phút/job/GPU) |

Tín hiệu khoa học sớm từ 2 job đầu (lab → field, source-only): AUC nguồn-val = 1.000 dù đã tách theo bản ghi, nhưng
AUC đích mức bản ghi chỉ 0.52–0.58. Trong miền S, hai lớp tách hoàn toàn nhưng tín hiệu đó không chuyển sang miền R.
Điều này khớp với giả thuyết dấu hiệu đặc thù mô hình thí nghiệm (mục 9), cần đối chiếu với bảng mục 3.9 của lượt
chạy đầy đủ.
