# Phân tích và quy trình nhận dạng rò rỉ nước trên ống nhánh PVC áp lực thấp bằng đặc trưng âm thanh thủ công và học máy

*Tài liệu phương pháp đi kèm notebook `notebooks/leak-ml-handcrafted-kaggle-v1.ipynb`.*
*Tên đề xuất cho bài báo:* **"Frequency Band, Pre-processing and a Minimal Handcrafted Feature Set for Acoustic
Leak Detection on Low-Pressure PVC Service Pipes"**.

---

## Mục lục

1. [Tóm tắt](#1-tóm-tắt)
2. [Mục tiêu và câu hỏi nghiên cứu](#2-mục-tiêu-và-câu-hỏi-nghiên-cứu)
3. [Phân tích bài toán](#3-phân-tích-bài-toán)
4. [Tổng quan quy trình](#4-tổng-quan-quy-trình)
5. [Bước 1 — Kiểm kê, định danh bản ghi, chia dữ liệu](#5-bước-1--kiểm-kê-định-danh-bản-ghi-chia-dữ-liệu)
6. [Bước 2 — Kiểm soát chất lượng và xử lý tín hiệu](#6-bước-2--kiểm-soát-chất-lượng-và-xử-lý-tín-hiệu)
7. [Bước 3 — Trích xuất đặc trưng thủ công](#7-bước-3--trích-xuất-đặc-trưng-thủ-công)
8. [Bước 4 — RQ1: dải tần chứa tín hiệu rò rỉ](#8-bước-4--rq1-dải-tần-chứa-tín-hiệu-rò-rỉ)
9. [Bước 5 — RQ2: tín hiệu xử lý hay tín hiệu thô](#9-bước-5--rq2-tín-hiệu-xử-lý-hay-tín-hiệu-thô)
10. [Bước 6 — RQ3: tập đặc trưng tối thiểu](#10-bước-6--rq3-tập-đặc-trưng-tối-thiểu)
11. [Bước 7 — Bốn mô hình học máy](#11-bước-7--bốn-mô-hình-học-máy)
12. [Bước 8 — Đánh giá và kiểm định thống kê](#12-bước-8--đánh-giá-và-kiểm-định-thống-kê)
13. [Bước 9 — RQ4: độ bền, diễn giải, tính ứng dụng](#13-bước-9--rq4-độ-bền-diễn-giải-tính-ứng-dụng)
14. [Hạn chế và các mối đe doạ tính hợp lệ](#14-hạn-chế-và-các-mối-đe-doạ-tính-hợp-lệ)
15. [Cấu trúc bài báo và ánh xạ hình/bảng](#15-cấu-trúc-bài-báo-và-ánh-xạ-hìnhbảng)
16. [Chạy trên Kaggle và tuỳ chỉnh](#16-chạy-trên-kaggle-và-tuỳ-chỉnh)
17. [Tài liệu tham khảo](#17-tài-liệu-tham-khảo)

---

## 1. Tóm tắt

Tập dữ liệu gồm các bản ghi âm thanh thu bằng **gậy nghe kết hợp micro độ nhạy cao** trên **ống nhánh PVC Φ27**
(từ ống phân phối tới đồng hồ khách hàng), **áp suất 0,5–2 bar**, gán nhãn `leak`/`noleak`. Quy trình đề xuất gồm:

1. **Kiểm kê theo bản ghi** (mã `Rxxx` là đơn vị độc lập) và tách một **tập kiểm tra khoá** 20 % số bản ghi.
2. **Chuỗi xử lý tín hiệu 7 điều kiện** (C0 thô → C1 lọc → C2 loại bất thường → C3 loại âm không liên quan →
   C4/C5/C6 ba kiểu khử nhiễu), mọi quy tắc đều *không nhìn nhãn* và dùng ngưỡng *tương đối với chính tệp*.
3. **~168 đặc trưng thủ công** thuộc 7 nhóm (thời gian, hình dạng phổ, năng lượng dải 1/3 octave, cepstral,
   wavelet, LPC, độ phức tạp/điều biến) trên khung 2 s.
4. **Bốn thí nghiệm** trả lời bốn câu hỏi nghiên cứu (dải tần; xử lý vs thô; tập đặc trưng tối thiểu; độ bền và
   tính ứng dụng) với **4 mô hình**: SVM-RBF, Random Forest, XGBoost, k-NN.
5. **Giao thức không rò rỉ dữ liệu**: CV lặp phân tầng trên bảng bản ghi, chọn đặc trưng và siêu tham số **trong
   từng fold** (CV lồng), đánh giá ở **mức bản ghi** với CI bootstrap, so sánh ghép cặp DeLong + Holm.

Điểm mới so với các công trình cùng hướng: (i) đối tượng ống nhánh PVC đường kính nhỏ, áp lực thấp, gậy nghe;
(ii) đo **trực tiếp** tác động của từng bước tiền xử lý thay vì mặc định "xử lý thì tốt hơn"; (iii) chỉ ra bằng thực
nghiệm rủi ro của khử nhiễu kiểu tiếng nói với tín hiệu rò rỉ *dừng*; (iv) tìm **dải tần thiết yếu** và **tập đặc
trưng tối thiểu** bằng giao thức không thiên lệch; (v) báo cáo **thời gian nghe tối thiểu** có giá trị vận hành.

---

## 2. Mục tiêu và câu hỏi nghiên cứu

| Mã | Câu hỏi | Kết quả cần có trong bài báo |
|---|---|---|
| **RQ1** | Khoảng tần số nào chứa tín hiệu rò rỉ? | Phổ trung bình theo lớp, phổ hiệu, phổ AUC theo dải; "dải thiết yếu" $[f_{HP}^*, f_{LP}^*]$ |
| **RQ2** | Âm thanh đã xử lý (lọc, loại bất thường, loại âm không liên quan, khử nhiễu) hay âm thanh thô cho nhận dạng tốt hơn? | Bảng AUC 7 điều kiện × 4 mô hình, ΔAUC ghép cặp so với thô, khẳng định trên tập kiểm tra |
| **RQ3** | Tập đặc trưng nhỏ nhất nào không làm giảm đáng kể hiệu năng? | Đường cong AUC(k), $K_{MIN}$, $K_{MAIN}\in[20,50]$, danh sách đặc trưng, độ ổn định |
| **RQ4** *(đề xuất bổ sung cho mục "4. …")* | Mô hình có bền, diễn giải được và dùng được ngoài hiện trường? | CV lồng + tập kiểm tra khoá; SHAP/permutation; kiểm tra confound mức tín hiệu; mức thổi phồng do chia sai; **thời gian nghe tối thiểu**; chi phí tính toán |

Các mục tiêu bổ sung khác nên cân nhắc (đã có sẵn điểm móc trong notebook hoặc cần thêm dữ liệu):

- **Thời gian nghe tối thiểu** (đã cài đặt, mục 10.2): người vận hành cần áp gậy bao nhiêu giây để đạt độ chính xác
  gần tối đa — kết quả có giá trị thực tiễn cao cho bài báo.
- **So sánh với CNN** (notebook v7 cùng kho mã): cùng tập bản ghi, nên dùng cùng phân hoạch fold để so sánh ghép cặp.
- **Tổng quát hoá theo điểm khảo sát/áp suất**: cần bổ sung siêu dữ liệu (mã điểm, áp suất đo, khoảng cách tới
  điểm rò, vị trí đặt gậy) để chạy leave-one-site-out và phân tích theo áp suất — đây là điểm yếu lớn nhất của dữ
  liệu hiện tại (xem §14).

---

## 3. Phân tích bài toán

### 3.1 Vật lý phát sinh và lan truyền tiếng rò rỉ trên ống PVC Φ27, 0,5–2 bar

**Nguồn âm.** Nước thoát qua lỗ/khe rò tạo **tia rối** (turbulent jet) và dao động của dòng tại mép lỗ; ở áp lực
0,5–2 bar, khả năng xâm thực (cavitation) thấp. Vận tốc tia ước lượng theo Torricelli
$v \approx C_d\sqrt{2\Delta p/\rho}$: với $\Delta p$ = 0,5–2 bar và $C_d$ ≈ 0,6–1 ta có $v$ ≈ 6–20 m/s. Công suất
âm của tia rối tăng rất nhanh theo vận tốc, nên **ở 0,5 bar tín hiệu rò rỉ yếu → SNR thấp** — đây là thách thức
chính của dữ liệu này. Tần số đặc trưng của ồn tia tỉ lệ với $v/d$ (quy luật Strouhal); với lỗ cỡ milimét, ước
lượng bậc độ lớn cho ra dải **hàng trăm Hz đến vài kHz** (chỉ để định hướng giả thuyết, không phải kết luận).

**Lan truyền.** Trong ống nhựa chứa nước, sóng chiếm ưu thế là sóng chất lỏng kết cấu với tốc độ theo công thức
Korteweg $c = \sqrt{(K/\rho)/(1 + K D/(E e))}$. Với PVC ($E$ ≈ 3 GPa), nước ($K$ ≈ 2,2 GPa) và $D/e$ ≈ 10–13
của ống Φ27, $c$ ≈ 450–500 m/s — thấp hơn nhiều so với ống kim loại (> 1 000 m/s). Thành ống đàn nhớt và đất xung
quanh gây **suy giảm mạnh, tăng theo tần số**: ống nhựa hoạt động như **bộ lọc thông thấp** và dải hữu ích co về
tần số thấp khi khoảng cách tăng (Hunaidi & Chu 1999; Muggleton et al. 2002; Gao et al. 2004, 2005; Almeida et al.
2014). Trên **ống nhánh ngắn** (vài mét tới đồng hồ), các thành phần tần số trung–cao (1–3 kHz) có thể vẫn còn —
điều mà phân tích thăm dò trước đó của dữ liệu này đã gợi ý (§3.3). Vì vậy **không nên hạ tần số lấy mẫu xuống
8 kHz** trước khi trả lời RQ1: notebook giữ 16 kHz (dải phân tích 20–7 600 Hz).

**Chuỗi đo.** Gậy nghe tiếp xúc cơ học với đồng hồ/van/ống; micro thu cả rung truyền qua gậy lẫn âm trong không
khí. Gậy có cộng hưởng riêng và hàm truyền phụ thuộc lực ép/điểm tiếp xúc → mỗi lần đo mang một "dấu vân tay"
phổ riêng. Nếu thiết bị cố định, dấu vân tay này gần như là bộ lọc hằng (không gây hại); nếu điểm đặt/lực ép thay
đổi có hệ thống giữa hai lớp, nó trở thành **confound**.

### 3.2 Các nguồn nhiễu và "âm thanh không liên quan"

| Nguồn | Tính chất | Dải | Xử lý trong quy trình |
|---|---|---|---|
| Trôi DC, hạ âm, rung tay cầm | dừng/chậm | < 20 Hz | C1: thông cao 20 Hz |
| Hài lưới điện 50 Hz | dừng, âm sắc | 50·k Hz | C1: notch 50 Hz × 6 hài (Q = 30) |
| Ồn nội micro/ADC, rít cao tần | dừng, băng rộng | > 5 kHz | C1: thông thấp 7 kHz; **không** tách được khỏi rò rỉ bằng khử nhiễu dừng |
| Va chạm/cọ gậy nghe, gõ | xung, kurtosis cao | băng rộng | C2: kẹp xung mẫu + loại khung (sự kiện năng lượng, kurtosis) |
| Bão hoà, mất tín hiệu | lỗi thu | — | QC tệp + C2 |
| Tiếng nói, chim | không dừng, hữu thanh | 100–4 000 Hz | C3: khung hữu thanh (ACF) + lệch log-phổ (LSD) |
| Xe cộ, bước chân, thi công | không dừng, băng rộng | thấp–trung | C2 (sự kiện năng lượng) / C3 (LSD); C4 cho phần còn sót |
| Bơm, động cơ chạy liên tục | dừng, âm sắc | hàng trăm Hz | không loại (hằng trong bản ghi); đặc trưng âm sắc giúp mô hình phân biệt |
| **Dòng tiêu thụ hợp lệ qua đồng hồ** | dừng, băng rộng — **giống rò rỉ** | như rò rỉ | **không xử lý được bằng DSP** → quy trình hiện trường: khoá vòi phía sau đồng hồ, kiểm tra kim lưu lượng nhỏ đứng yên, ghi chú |

Hệ quả thiết kế: **tín hiệu rò rỉ là ồn băng rộng dừng**. Mọi phương pháp khử nhiễu "kiểu tiếng nói" (trừ phổ,
cổng phổ, wavelet ngưỡng phổ quát) coi *thành phần dừng* là nhiễu nền → **có nguy cơ xoá chính tiếng rò rỉ**. Quy trình
vì vậy tách hai việc: (a) *loại bỏ* thứ không phải rò rỉ bằng tiêu chí **không dừng** (C2, C3, C4), và (b) *đo* tác động
của khử nhiễu kinh điển (C5, C6) thay vì mặc định dùng chúng.

### 3.3 Đặc điểm tập dữ liệu hiện có (từ lần chạy thăm dò v6 trong kho mã)

| Thuộc tính | Giá trị |
|---|---|
| Bản ghi độc lập | **80** (34 leak / 46 noleak) |
| Tệp | 167 (1–5 tệp/bản ghi, trung vị 2) |
| Tần số lấy mẫu | 16 kHz, mono |
| Thời lượng bản ghi | 25–189 s; tổng 123,7 phút |
| Thông tin điểm khảo sát, áp suất | **chưa có** |

Kết quả thăm dò trước (CNN v6, cùng dữ liệu): năng lượng tương đối dải 283–413 Hz cao hơn ở noleak (AUC 0,36) và dải
1,9–2,7 kHz cao hơn ở leak (AUC 0,62) — **không dải nào còn ý nghĩa sau hiệu chỉnh đa so sánh**; baseline 13 đặc
trưng phổ đạt AUC bản ghi ≈ 0,69; CNN log-Mel tốt nhất ≈ 0,89; chỉ riêng mức RMS ≈ 0,61; hệ số tương quan nội lớp
(ICC) của điểm số ≈ 0,70 (điểm số gần như hằng trong mỗi bản ghi).

Hàm ý: (1) cỡ mẫu thực là **80, không phải hàng nghìn khung** → mọi lựa chọn (xử lý, số đặc trưng, siêu tham số) phải
nằm trong CV hoặc tách khỏi tập kiểm tra; (2) thông tin phân biệt có thể **phân tán ở cả dải thấp và dải 1–3 kHz**;
(3) baseline đặc trưng thủ công trước đây còn rất sơ sài (13 đặc trưng, không xử lý, không chọn) → còn nhiều dư địa;
(4) phải đo và thảo luận confound mức tín hiệu/điểm khảo sát.

---

## 4. Tổng quan quy trình

```
            ┌──────────────────────────── DỮ LIỆU ────────────────────────────┐
 leak/ noleak/ ──► kiểm kê theo Rxxx ──► tách TEST khoá 20 % (phân tầng) ──► DEV 80 %
            └─────────────────────────────────────────────────────────────────┘
                                   │ (mỗi tệp, không nhìn nhãn)
 QC tệp ─► C0 thô ─► C1 band-pass 20–7000 Hz + notch 50 Hz×6 ─► C2 loại khung bất thường ─► C3 loại âm không liên quan
                                                                 (bão hoà, mất TH,           (LSD tới phổ trung vị tệp,
                                                                  sự kiện năng lượng,          tỉ lệ khối hữu thanh)
                                                                  kurtosis; kẹp xung)
                                                       ─► C4 kẹp trung vị STFT | C5 trừ phổ | C6 wavelet VisuShrink
                                   │
               khung 2 s, không chồng ─► ~168 đặc trưng thủ công / khung
                                   │
 E1 (RQ1) phổ & quét dải ── E2 (RQ2) 7 điều kiện × 4 mô hình ── E3 (RQ3) xếp hạng đồng thuận trong fold, AUC(k)
                                   │
 E4 CV lồng trên DEV + mô hình cuối → TEST khoá (1 lần) ── E5 (RQ4) SHAP, confound, rò rỉ dữ liệu, thời gian nghe
```

---

## 5. Bước 1 — Kiểm kê, định danh bản ghi, chia dữ liệu

- **Nhãn** lấy từ tên thư mục; **mã bản ghi** trích bằng regex `R\d{1,4}` trên tên tệp. Một mã mang hai nhãn → dừng.
- **Đơn vị thống kê là bản ghi.** Các khung 2 s của cùng một bản ghi tương quan rất mạnh; chia fold theo khung sẽ để
  "anh em sinh đôi" của khung kiểm tra nằm trong tập huấn luyện → chỉ số thổi phồng (Kaufman et al. 2012; Roberts
  et al. 2017). Notebook đo trực tiếp mức thổi phồng này (mục 10.2).
- **Tập kiểm tra khoá (hold-out)**: 20 % bản ghi (≈ 16), phân tầng theo lớp, seed cố định. Không tham gia bất kỳ
  lựa chọn nào; chỉ dùng một lần ở E4. CI trên 16 bản ghi rộng — đó là cái giá của một ước lượng độc lập; ước lượng
  chính vẫn là **CV lồng trên DEV**.
- **CV lặp**: 3 lần × 5 fold, phân tầng trực tiếp trên **bảng bản ghi**. Cùng một kế hoạch fold được dùng cho mọi
  thí nghiệm → mọi so sánh là **ghép cặp** (cùng bản ghi, cùng fold).
- **Đầu ra mức bản ghi** = trung bình xác suất các khung (quy tắc đăng ký trước); ngưỡng 0,5 trên mô hình đã cân bằng lớp.
- **Trọng số mẫu**: mỗi bản ghi có tổng trọng số như nhau (bản ghi dài không lấn át), mỗi lớp tổng trọng số như nhau.

---

## 6. Bước 2 — Kiểm soát chất lượng và xử lý tín hiệu

Mọi quy tắc dưới đây **không dùng nhãn** và **không dùng thống kê chéo giữa các tệp** → có thể tính trước cho toàn bộ
dữ liệu mà không gây rò rỉ thông tin sang tập kiểm tra.

### 6.1 QC mức tệp

| Chỉ số | Định nghĩa | Loại tệp khi |
|---|---|---|
| Tỉ lệ bão hoà | $\frac{1}{N}\sum \mathbb{1}(|x_n|\ge 0{,}999)$ | > 1 % |
| Tỉ lệ mất tín hiệu | tỉ lệ mẫu bằng 0 | > 50 % |
| Thời lượng | — | < 1 khung |
| DC, mức RMS (dBFS), "SNR" ≈ P95 − P10 mức khối 50 ms | chỉ báo cáo | — |

### 6.2 C1 — Lọc điều kiện hoá

- Khử DC; **Butterworth bậc 4 band-pass 20–7 000 Hz, pha không** (`sosfiltfilt`) — loại trôi nền/hạ âm và vùng gần
  Nyquist, không làm lệch pha (giữ nguyên hình dạng xung cho C2).
- **Notch IIR** tại 50, 100, …, 300 Hz, Q = 30 (băng chặn ≈ 1,7–10 Hz) — loại hài lưới điện, mất rất ít năng lượng rò rỉ.
- Dải C1 được **cố định trước** (không chọn theo E1) để tránh dùng dữ liệu hai lần.

### 6.3 C2 — Loại tín hiệu bất thường (xung, va chạm, bão hoà)

1. **Kẹp xung mức mẫu**: $\sigma_{MAD} = 1{,}4826\,\mathrm{median}|x-\mathrm{median}(x)|$; kẹp $|x| > 8\sigma_{MAD}$
   (với ồn Gauss xác suất vượt ≈ 10⁻¹⁵ → chỉ tác động lên xung thật).
2. **Loại khung** (khung 2 s) nếu một trong các điều kiện:
   - *Sự kiện năng lượng*: mức khối 50 ms mạnh nhất trong khung $L_{max}$ so với trung vị mức khối của **cả tệp**:
     $z = (L_{max} - \tilde L)/(1{,}4826\,\mathrm{MAD}) > 3{,}5$ **và** $L_{max} - \tilde L > 6$ dB
     (modified z-score của Iglewicz & Hoaglin 1993; điều kiện dB tránh nhạy quá mức với ồn dừng có MAD rất nhỏ).
   - *Xung*: kurtosis của khung > 8 (ồn Gauss = 3).
   - *Bão hoà* > 0,1 % mẫu; *mất tín hiệu*: RMS < 10⁻⁵ hoặc > 20 % mẫu bằng 0.

### 6.4 C3 — Loại âm thanh không liên quan tới rò rỉ

Rò rỉ tạo nền **dừng** kéo dài suốt bản ghi; âm thanh không liên quan (tiếng nói, xe, chim…) **xuất hiện rồi biến mất**.
Hai tiêu chí tương đối so với chính tệp:

- **Khoảng cách log-phổ (LSD)** giữa phổ 1/3 octave của khung và **phổ trung vị** của tệp:
  $\mathrm{LSD}_f = \sqrt{\frac{1}{B}\sum_b (S_{f,b} - \tilde S_b)^2}$ (dB); loại nếu modified-z > 3,5 và LSD > 3 dB.
- **Mức hữu thanh**: tín hiệu lọc 60–1 000 Hz, khối 40 ms bước 20 ms, đỉnh ACF chuẩn hoá (không chệch) ở trễ 2–14 ms
  (70–500 Hz) > 0,5 → khối hữu thanh (Rabiner & Schafer 1978; Boersma 1993). Loại khung nếu tỉ lệ khối hữu thanh > 30 %
  **và** cao hơn trung vị của tệp ≥ 0,2 — điều kiện thứ hai giữ lại **tiếng rít rò rỉ liên tục** (âm sắc nhưng dừng).
- **Bảo vệ**: mỗi tệp luôn giữ ≥ 30 % số khung (≥ 3 khung) — không bản ghi nào biến mất.
- **Kiểm tra confound của chính bộ lọc**: so sánh tỉ lệ khung bị loại giữa hai lớp (Mann–Whitney mức bản ghi).
  Nếu khác biệt có ý nghĩa, việc loại khung có thể tự tạo ra khác biệt giữa lớp → phải thảo luận.

### 6.5 Khử nhiễu — ba cách tiếp cận được so sánh

| Điều kiện | Phương pháp | Công thức | Giả thuyết |
|---|---|---|---|
| **C4 – MBG** (đề xuất) | kẹp biên độ STFT về trung vị theo thời gian | $G(f,t) = \min\!\left(1, \beta\,\tilde{M}(f)/|X(f,t)|\right)$, $\beta$ = 2, $\tilde M(f)$ = trung vị theo thời gian | giữ thành phần **dừng** (rò rỉ), triệt phần nhô lên **nhất thời** còn sót sau C2–C3 (ý tưởng tương tự lọc trung vị nền của FitzGerald 2010; Rafii & Pardo 2013) |
| **C5 – SS** | trừ phổ công suất (Boll 1979; Berouti et al. 1979), nhiễu ước lượng bằng thống kê cực tiểu từ chính tệp | $G = \sqrt{\max(1 - \alpha\hat N(f)/|X|^2,\ \beta_f^2)}$, $\alpha$ = 2, sàn −20 dB, $\hat N$ = phân vị 20 %/(−ln 0,8) | cách "khử nhiễu nền" phổ biến nhất (noisereduce, Sainburg et al. 2020 cùng họ) — **dự đoán làm yếu tín hiệu rò rỉ** |
| **C6 – WT** | wavelet db8, 6 mức, ngưỡng mềm phổ quát $\sigma\sqrt{2\ln N}$ (Donoho & Johnstone 1994) | $\sigma$ = MAD(chi tiết mức 1)/0,6745 | phổ biến trong các bài AE/rò rỉ ống — **dự đoán xoá phần lớn ồn băng rộng** |

Phát biểu kiểm định được: *"Khử nhiễu kiểu tiếng nói (C5, C6) làm giảm AUC so với C3; khử nhiễu bảo toàn thành phần dừng
(C4) không làm giảm hoặc cải thiện AUC."* Đây là một đóng góp phương pháp có thể dùng làm thông điệp chính của bài báo
nếu số liệu ủng hộ.

### 6.6 Bảng tham số (đăng ký trước, sửa ở `cfg` mục 1.2 của notebook)

| Nhóm | Tham số | Giá trị |
|---|---|---|
| Khung | `frame_sec`, `hop_sec`, `target_sr` | 2 s, 2 s (không chồng), 16 kHz |
| C1 | `hp_hz`, `lp_hz`, `filt_order`, `notch_hz`×`notch_harmonics`, `notch_q` | 20, 7 000, 4, 50×6, 30 |
| C2 | `spike_k`, `block_ms`, `z_thr`, `event_db`, `kurt_thr`, `frame_clip_frac` | 8, 50, 3,5, 6, 8, 0,001 |
| C3 | `lsd_z_thr`, `lsd_min_db`, `voicing_thr`, `voicing_frac_thr`, `voicing_excess` | 3,5, 3, 0,5, 0,3, 0,2 |
| Bảo vệ | `min_keep_frac`, `min_keep_frames` | 0,3, 3 |
| C4/C5/C6 | `mbg_beta`; `ss_alpha`, `ss_floor`, `ss_quantile`; `wt_wavelet`, `wt_level` | 2; 2, 0,1, 0,2; db8, 6 |

---

## 7. Bước 3 — Trích xuất đặc trưng thủ công

Khung 2 s (32 000 mẫu). PSD Welch cửa sổ Hann 4 096 mẫu (256 ms, Δf = 3,9 Hz), chồng 50 %; STFT 1 024 mẫu (64 ms) cho
cepstral/flux/contrast; STFT 256 mẫu cho spectral kurtosis. Dải phân tích 20–7 600 Hz.

| Nhóm (số lượng) | Đặc trưng | Công thức / ghi chú | Tham chiếu |
|---|---|---|---|
| **Time-domain (18)** | RMS (dB)\*, đỉnh (dB)\*, crest $=x_{pk}/x_{rms}$, impulse $=x_{pk}/\overline{|x|}$, shape $=x_{rms}/\overline{|x|}$, clearance $=x_{pk}/(\overline{\sqrt{|x|}})^2$, skewness, kurtosis, ZCR, Hjorth mobility/complexity, entropy năng lượng 20 khối, độ lệch chuẩn mức khối 50 ms, CV và kurtosis của đường bao Hilbert, thời gian suy giảm 1/e và điểm 0 đầu tiên của ACF, đỉnh ACF ở trễ 2–20 ms | rò rỉ: gần Gauss, dừng (kurtosis ≈ 3, crest thấp, đường bao phẳng) | Randall 2011; Hjorth 1970; Giannakopoulos 2015 |
| **Spectral shape (30)** | centroid, spread, skewness, kurtosis phổ, entropy phổ, flatness, roll-off 50/85/95 %, độ dốc (dB/octave), decrease, crest phổ, tần số đỉnh, số đỉnh âm sắc & độ nhô lớn nhất, flux (TB, SD), spectral contrast 7 octave, spectral kurtosis (TB, max, tần số max, 3 dải) | flatness $= \exp(\overline{\ln P})/\overline{P}$; SK $= \langle|X|^4\rangle/\langle|X|^2\rangle^2 - 2$ | Peeters 2004; Jiang et al. 2002; Antoni 2006 |
| **Band energy (35)** | năng lượng **tương đối** 25 dải 1/3 octave (25 Hz–6,3 kHz), 3 tỉ số dải (<1 k/>1 k, 100–800 Hz, 1–3 kHz), mức **tuyệt đối** 7 dải octave\* | $10\log_{10}(E_b/E_{tot})$ — bất biến với hệ số khuếch đại | Hunaidi & Chu 1999; Almeida et al. 2014 |
| **Cepstral (41)** | MFCC 1–13 (TB, SD), MFCC0\*, SD delta-MFCC, LFCC 1–13 (TB) | 40 bộ lọc Mel / tuyến tính 20–7 600 Hz, DCT-II | Davis & Mermelstein 1980; Zhou et al. 2011 |
| **Wavelet (26)** | năng lượng tương đối DWT db4 7 mức (A7, D7…D1) + entropy; WPD db4 mức 4 (16 dải 500 Hz) + entropy | phân bố năng lượng đa phân giải | Mallat 1989; Coifman & Wickerhauser 1992 |
| **LPC (11)** | hệ số $a_1…a_{10}$ (Burg) + sai số dự báo (dB) | mô hình cộng hưởng ống/gậy | Makhoul 1975; Cody, Dey & Narasimhan 2020 |
| **Complexity & modulation (7)** | permutation entropy (m = 5, trễ 1 và 4), chiều fractal Higuchi & Katz, năng lượng điều biến đường bao 0,5–4 / 4–16 / 16–64 Hz | tiếng nói điều biến ~4 Hz; rò rỉ gần như không | Bandt & Pompe 2002; Higuchi 1988; Katz 1988; Atlas & Shamma 2003 |

\* *Đặc trưng phụ thuộc mức tín hiệu* (10 đặc trưng) — hữu ích nếu hệ số khuếch đại micro cố định giữa các lần đo,
nhưng có thể là confound nếu không. E5 kiểm tra riêng.

Tổng cộng **~168 đặc trưng/khung** (số chính xác in ở mục 4.1 của notebook, phụ thuộc tần số lấy mẫu). Thêm 51 mức dải
1/6 octave/khung chỉ dùng cho quét dải E1.

---

## 8. Bước 4 — RQ1: dải tần chứa tín hiệu rò rỉ

Điều kiện dùng: **C3** (đã loại bất thường và âm không liên quan, *chưa* khử nhiễu để không làm méo phổ).

| Phân tích | Cách làm | Đọc kết quả |
|---|---|---|
| **E1a – LTAS** | PSD mức bản ghi (TB các khung), chuẩn hoá theo tổng năng lượng, trung vị + IQR theo lớp | hình dạng phổ tổng quát, vùng hai lớp tách nhau |
| **E1a – Phổ hiệu** | $\Delta(f) = \mathrm{median}_{leak} - \mathrm{median}_{noleak}$ (dB, dải 1/6 octave), CI bootstrap theo bản ghi | vùng Δ > 0 có CI không chứa 0: năng lượng tương đối **tăng** khi có rò rỉ |
| **E1b – Phổ AUC** | AUC đơn biến mức bản ghi cho từng dải 1/6 octave (tương đối và tuyệt đối), Mann–Whitney, BH-FDR và Holm; Bảng 4 cho 1/3 octave kèm Cliff's δ | AUC > 0,5: cao hơn ở leak; < 0,5: cao hơn ở noleak |
| **E1d – Quét dải bằng mô hình** | RF chỉ dùng thông tin trong cửa sổ: (i) octave trượt bước 1/3 octave; (ii) thông thấp tích luỹ $[20, f_c]$; (iii) thông cao tích luỹ $[f_c, 7600]$; đặc trưng = mức tương đối các dải 1/6 octave trong cửa sổ + mức tổng cửa sổ; CV lặp nhóm theo bản ghi | **dải thiết yếu** $[f_{HP}^*, f_{LP}^*]$: $f_{LP}^*$ = tần số cắt thông thấp nhỏ nhất và $f_{HP}^*$ = tần số cắt thông cao lớn nhất còn giữ AUC ≥ AUC(toàn dải) − 0,02 |

Cách phát biểu kết quả trong bài báo: *"The leak-related information on these PVC service pipes is concentrated in
[f_HP*, f_LP*] Hz: a model restricted to this band achieves AUC = … [CI] versus … for the full 20–7600 Hz band; the
relative energy in … Hz is higher in leak recordings (AUC …, q_BH …) whereas … Hz is higher in no-leak recordings."*
Nếu $f_{HP}^* \ge f_{LP}^*$, thông tin **lặp lại** ở cả hai phía phổ (mỗi phía tự nó đủ) — cũng là một phát hiện
đáng báo cáo (không tồn tại một dải hẹp duy nhất).

E1 là phân tích **mô tả** — không tham số nào của E2–E4 được chọn từ nó — nên được phép dùng toàn bộ 80 bản ghi
(`cfg.e1_scope = "all"`) để tăng độ mạnh thống kê. Nếu phản biện yêu cầu chặt hơn, đặt `"dev"`.

---

## 9. Bước 5 — RQ2: tín hiệu xử lý hay tín hiệu thô

**Thiết kế có kiểm soát**: cùng danh mục đặc trưng, cùng fold (ghép cặp), cùng 4 mô hình với **siêu tham số mặc định
cố định**; chỉ thay đổi điều kiện tín hiệu C0…C6. Hai chế độ đặc trưng: `all` và `mrmr30` (30 đặc trưng chọn lại trong
từng fold). Chỉ dùng DEV.

- **Chỉ số chính**: AUC mức bản ghi (trung bình 3 lần lặp CV); kèm AUC của điểm OOF trung bình + CI bootstrap,
  balanced accuracy, sensitivity, specificity, F1, MCC.
- **So sánh**: (i) mỗi điều kiện vs C0 (thô); (ii) **từng bước** so với bước liền trước (C1 vs C0, C2 vs C1, C3 vs C2,
  C4/C5/C6 vs C3) → đóng góp riêng của từng khâu. Kiểm định DeLong (Holm theo từng mô hình), CI bootstrap ghép cặp của
  ΔAUC, t-test hiệu chỉnh Nadeau–Bengio trên AUC từng fold.
- **Quy tắc chọn điều kiện tốt nhất (đăng ký trước)**: AUC trung bình 4 mô hình ở chế độ `mrmr30`.
- **Khẳng định trên TEST (E4c)**: huấn luyện trên DEV ở C0 và ở điều kiện tốt nhất, so sánh ghép cặp trên cùng các
  bản ghi TEST.

Các kịch bản kết quả và cách diễn giải:

| Kết quả | Diễn giải |
|---|---|
| C1–C4 > C0 có ý nghĩa | xử lý giúp; báo cáo đóng góp từng bước |
| C1 ≈ C0, C2/C3 > C1 | lọc cố định ít tác dụng (đặc trưng tương đối vốn bền với nhiễu điện), **loại nhiễu xung/không liên quan** mới quan trọng |
| C5, C6 < C3 | **khử nhiễu dừng xoá tín hiệu rò rỉ** — khuyến nghị không dùng cho bài toán này |
| Mọi Δ nhỏ, CI chứa 0 | đặc trưng + mô hình đủ bền; kết luận "xử lý không cần thiết" chỉ hợp lệ khi CI hẹp (≤ ±0,03) |

Lưu ý: tối đa 16 bản ghi TEST → E4c có độ mạnh thấp; kết luận chính dựa trên DEV (64 bản ghi, 15 fold ghép cặp).

---

## 10. Bước 6 — RQ3: tập đặc trưng tối thiểu

1. **Liên quan đơn biến (E3a)**: AUC mức bản ghi, Cliff's δ, Mann–Whitney + BH-FDR cho 168 đặc trưng.
2. **Dư thừa (E3b)**: ma trận |ρ Spearman|, phân cụm phân cấp; số cụm ở |ρ| ≥ 0,9 = số chiều "hiệu dụng".
3. **Xếp hạng trong từng fold** bằng 5 phương pháp thuộc 3 họ, gộp **Borda** (Saeys et al. 2008):
   - lọc: **ANOVA-F**, **thông tin tương hỗ** (Kraskov et al. 2004; Ross 2014);
   - lọc đa biến: **mRMR-FCQ** — độ liên quan F, độ dư thừa |r|, điểm $F_j / \overline{|r_{j,S}|}$ (Ding & Peng 2005; Peng et al. 2005);
   - nhúng: độ quan trọng **Extra-Trees** (Geurts et al. 2006), |hệ số| **hồi quy logistic L1** (Tibshirani 1996).

   Việc xếp hạng **lặp lại trên phần huấn luyện của mỗi fold** là bắt buộc — chọn trên toàn bộ rồi mới CV gây thiên
   lệch lựa chọn nghiêm trọng (Ambroise & McLachlan 2002; Varma & Simon 2006).
4. **Đường cong AUC(k)**, k ∈ {3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, tất cả} × 4 mô hình. Hai quy tắc đăng ký trước:
   - $K_{MIN}$ = k nhỏ nhất có AUC trung bình 4 mô hình ≥ AUC tốt nhất − 0,01 → **câu trả lời RQ3**
     (biến thể "1-SE" của Breiman et al. 1984, với ngưỡng tuyệt đối dễ diễn giải);
   - $K_{MAIN}$ = k tốt nhất trong [20, 50] → bộ làm việc chính.
5. **So sánh phương pháp chọn** (k = 10, 20, 30; SVM và RF) và **độ ổn định** Nogueira et al. (2018):
   $\hat\Phi = 1 - \frac{\frac{1}{p}\sum_f s_f^2}{\frac{\bar k}{p}(1-\frac{\bar k}{p})}$ (1 = luôn chọn cùng tập).
6. **Danh sách cuối**: xếp hạng đồng thuận trung bình qua 15 fold DEV → top-$K_{MAIN}$, top-$K_{MIN}$; kèm nhóm,
   dải tần, chiều ảnh hưởng, tần suất được chọn.

Cách phát biểu: *"A subset of K_MIN features (… band-energy, … wavelet, … cepstral) retains AUC within 0.01 of the
best configuration, while the 168-feature set forms only ~N clusters at |ρ| ≥ 0.9."*

---

## 11. Bước 7 — Bốn mô hình học máy

| Mô hình | Vì sao chọn | Cài đặt | Lưới siêu tham số (CV lồng) |
|---|---|---|---|
| **SVM-RBF** (Cortes & Vapnik 1995) | bộ phân loại phổ biến nhất với đặc trưng thủ công trong nhận dạng rò rỉ (Kang et al. 2018; Xu et al. 2021; Fares et al. 2023) | chuẩn hoá z, $\gamma = g/k$ | C ∈ {0,3, 1, 3, 10, 30}, g ∈ {0,3, 1, 3} (Hsu et al. 2003) |
| **Random Forest** (Breiman 2001) | bền với đặc trưng dư thừa, ít cần chỉnh (Ning et al. 2021; Tijani et al. 2022) | 300 cây | max_features ∈ {√p, 0,3}, min_samples_leaf ∈ {1, 5, 20} |
| **XGBoost** (Chen & Guestrin 2016) | gradient boosting — thường mạnh nhất trên dữ liệu bảng | 300 cây, `hist`, subsample 0,8 | max_depth ∈ {3, 5}, η ∈ {0,05, 0,1}, min_child_weight ∈ {1, 5} |
| **k-NN** (Cover & Hart 1967) | mốc phi tham số kinh điển (El-Zahab et al. 2018; Ullah et al. 2023) | Euclid trên dữ liệu chuẩn hoá | k ∈ {11, 21, 41, 81, 161}, trọng số ∈ {đều, khoảng cách} |

Ghi chú kỹ thuật:
- **Cân bằng**: SVM/RF/XGB nhận trọng số mẫu (bản ghi × lớp); k-NN không hỗ trợ trọng số nên xác suất được **hiệu chỉnh
  tiên nghiệm** về 50/50 trước khi dùng ngưỡng 0,5.
- **k-NN ở mức khung**: k phải lớn hơn số khung/bản ghi (~45) để "hàng xóm" đến từ nhiều bản ghi khác nhau — lưới k
  được chọn theo lý do này.
- SVM dùng $\sigma(\text{decision})$ làm điểm (thứ tự giống hệt, không cần Platt nội bộ).
- Có sẵn `LR` và `MLP` trong notebook nếu muốn mở rộng (`cfg.models`).

---

## 12. Bước 8 — Đánh giá và kiểm định thống kê

| Mục | Lựa chọn | Tham chiếu |
|---|---|---|
| Chỉ số chính | **AUC mức bản ghi** (không phụ thuộc ngưỡng) | — |
| Chỉ số phụ | AP, accuracy, **balanced accuracy**, sensitivity, specificity, precision, F1, **MCC**; AUC mức khung | Chicco & Jurman 2020; Saito & Rehmsmeier 2015 |
| CI | bootstrap phân tầng theo bản ghi (2 000 lần); mức khung: bootstrap **theo cụm** bản ghi | Efron & Tibshirani 1993 |
| So sánh hai mô hình/điều kiện | **DeLong** cho AUC tương quan (cài đặt nhanh Sun & Xu 2014) + CI bootstrap ghép cặp của ΔAUC + t-test hiệu chỉnh **Nadeau–Bengio** trên AUC từng fold | DeLong et al. 1988; Nadeau & Bengio 2003 |
| Đa so sánh | **Holm** (họ so sánh nhỏ), **Benjamini–Hochberg** (quét nhiều dải/đặc trưng) | Holm 1979; Benjamini & Hochberg 1995 |
| Ước lượng không thiên lệch | **CV lồng**: chọn đặc trưng + siêu tham số trong vòng trong, nhóm theo bản ghi | Varma & Simon 2006; Cawley & Talbot 2010 |
| Ước lượng độc lập | **TEST khoá** dùng một lần | — |

Không dùng kiểm định Wilcoxon trên 5 fold (p nhỏ nhất đạt được = 0,0625, không bao giờ bác bỏ được H₀ ở 0,05).

---

## 13. Bước 9 — RQ4: độ bền, diễn giải, tính ứng dụng

| Phân tích | Mục đích | Cách đọc |
|---|---|---|
| Permutation importance trên TEST; **SHAP** (TreeExplainer) cho mô hình cây tốt nhất | đặc trưng nào quyết định | đặc trưng quan trọng có nằm trong dải thiết yếu E1 không? (nhất quán nội tại) |
| **Bản đồ tần số của độ quan trọng** | nối RQ3 với RQ1 | các đoạn ngang theo dải của đặc trưng |
| Chỉ dùng mức RMS | mô hình có chỉ "nghe độ to"? | AUC gần mô hình đầy đủ → nghi confound mức/gain |
| Chọn lại top-$K_{MAIN}$ **không có** đặc trưng phụ thuộc mức | bền với hệ số khuếch đại? | ΔAUC ≈ 0 → kết luận không phụ thuộc gain |
| Chia fold ngẫu nhiên theo **khung** vs theo **bản ghi** | định lượng lỗi phương pháp phổ biến | "B − A" = mức thổi phồng; nên đưa vào Discussion |
| **Thời gian nghe tối thiểu**: AUC khi chỉ dùng n khung liên tục (2–90 s) | giá trị vận hành | số giây nghe để đạt AUC trong phạm vi 0,02 của nghe toàn bộ |
| Chi phí tính toán | khả thi trên thiết bị hiện trường | ms/khung cho trích đặc trưng và suy luận |
| Phân tích lỗi theo bản ghi | hiểu trường hợp khó | thời lượng, tỉ lệ khung bị loại, mức RMS của bản ghi sai |

---

## 14. Hạn chế và các mối đe doạ tính hợp lệ

1. **Cỡ mẫu nhỏ (80 bản ghi, ~16 bản ghi TEST)**: CI rộng; tránh kết luận từ khác biệt nhỏ hơn độ rộng CI.
   Báo cáo đầy đủ CI và p đã hiệu chỉnh.
2. **Thiếu thông tin điểm khảo sát/áp suất**: không loại trừ được khả năng mô hình học "dấu vân tay vị trí"
   (ICC ≈ 0,7 ở phân tích trước). Khuyến nghị bổ sung CSV `record_id, site, pressure_bar, distance_m, contact_point`
   để chạy leave-one-site-out và phân tích theo áp suất.
3. **Nhiễu nhãn tiềm ẩn**: `noleak` có thể chứa dòng tiêu thụ hợp lệ (âm giống rò rỉ); `leak` có thể có đoạn rò rỉ
   yếu khi áp suất thấp. Ghi rõ quy trình hiện trường trong bài báo.
4. **Hệ số khuếch đại**: nếu micro/gậy dùng mức khuếch đại khác nhau giữa các lần đo, đặc trưng mức tuyệt đối là
   confound — đã tách riêng và kiểm tra (E5).
5. **Ngưỡng tiền xử lý**: cố định trước theo nguyên lý thống kê (modified-z 3,5; kurtosis 8…), không tinh chỉnh theo
   kết quả. Có thể thêm phân tích độ nhạy ngưỡng (ví dụ z_thr ∈ {3, 3,5, 4}) như tài liệu bổ sung.
6. **Chọn điều kiện tốt nhất trên DEV rồi báo cáo DEV**: có thiên lệch lựa chọn nhỏ (7 điều kiện) — TEST khoá và E4c
   là kiểm chứng độc lập.
7. **Tổng quát hoá** sang ống khác (HDPE, gang), áp suất khác, thiết bị khác: chưa được kiểm chứng.

---

## 15. Cấu trúc bài báo và ánh xạ hình/bảng

**IMRaD gợi ý** (hội nghị 6–8 trang → rút gọn E3d, E5; tạp chí đầy đủ):

1. *Introduction* — rò rỉ trên ống nhánh, khó khăn ống nhựa áp lực thấp, khoảng trống (§3, bảng tài liệu mục 0 notebook), RQ1–RQ4.
2. *Materials* — thiết bị gậy nghe + micro, ống PVC Φ27, 0,5–2 bar, quy trình thu, tập dữ liệu (Bảng 1, Hình 1).
3. *Methods* — xử lý tín hiệu (Hình 2–3, Bảng 2), đặc trưng (Bảng 3), phân tích dải (E1), thiết kế E2, chọn đặc trưng
   (E3), mô hình & giao thức đánh giá (§11–12).
4. *Results* — RQ1 (Hình 5–6, Bảng 4–5), RQ2 (Hình 7–8, Bảng 6–7, 15), RQ3 (Hình 9–10, Bảng 8–12), mô hình cuối
   (Hình 11–12, Bảng 13–14), RQ4 (Hình 13–15, Bảng 16–21).
5. *Discussion* — vật lý (dải thiết yếu vs lý thuyết ống nhựa), vì sao khử nhiễu dừng hại/không hại, so sánh với CNN
   (notebook v7) và tài liệu, ý nghĩa vận hành (thời gian nghe), hạn chế (§14).
6. *Conclusions*.

| Tệp đầu ra | Nội dung | Vị trí đề xuất |
|---|---|---|
| `Fig01_dataset_overview` | số bản ghi/lớp, thời lượng, số tệp | Materials |
| `Fig02_processing_examples`, `Fig03_denoiser_psd` | minh hoạ loại khung và tác động khử nhiễu | Methods |
| `Fig04_rejection_stats`, `T02_*` | tỉ lệ loại theo lý do/lớp | Methods/Results |
| `Fig05_ltas_discriminability`, `T04_*` | LTAS, phổ hiệu, phổ AUC | Results – RQ1 |
| `Fig06_band_sweep`, `T05_band_sweep` | quét dải, dải thiết yếu | Results – RQ1 |
| `Fig07_E2_heatmap`, `Fig08_E2_delta_vs_raw`, `T06`, `T07` | 7 điều kiện × 4 mô hình, ΔAUC | Results – RQ2 |
| `Fig09_univariate_redundancy`, `T08` | liên quan đơn biến, dư thừa | Results – RQ3 |
| `Fig10_feature_selection`, `T09–T12` | AUC(k), K_MIN, K_MAIN, danh sách đặc trưng, độ ổn định | Results – RQ3 |
| `Fig11_roc_pr`, `Fig12_test_confusion_scores`, `T13`, `T14` | mô hình cuối, DEV lồng và TEST | Results |
| `T15_E4c_raw_vs_best_test` | khẳng định RQ2 trên TEST | Results – RQ2 |
| `Fig13_shap_summary`, `Fig14_importance_frequency_map`, `T16` | diễn giải | Discussion |
| `T17_robustness`, `T18_leakage_demo`, `Fig15_listening_time`, `T19–T21` | độ bền, rò rỉ dữ liệu, thời gian nghe, chi phí, lỗi | Results – RQ4 / Discussion |
| `RESULTS_summary.md`, `key_results.json` | đoạn kết quả tự sinh và số liệu chính | nháp Results/Abstract |

---

## 16. Chạy trên Kaggle và tuỳ chỉnh

1. Tạo notebook mới trên Kaggle → `File → Import Notebook` → chọn `notebooks/leak-ml-handcrafted-kaggle-v1.ipynb`.
2. `Add Data` → gắn dataset có `leak/` và `noleak/`. Sửa `cfg.data_root` ở mục 1.2 (mặc định trỏ tới dataset
   `tapdulieutt-8k-v7-2/field_rec_8k_V7.2` đã dùng cho v7; nếu sai, notebook tự dò trong `/kaggle/input`).
3. `Accelerator = None` (CPU là đủ); bật Internet nếu môi trường thiếu `xgboost`/`PyWavelets`.
4. `Run All` — khoảng 20–40 phút trên CPU Kaggle (thử nghiệm trên 80 bản ghi mô phỏng ≈ 87 phút âm thanh, 4 lõi: 16 phút,
   trong đó trích đặc trưng 7 điều kiện ≈ 3 phút). Đặc trưng được cache (`leak_ml/cache/`), chạy lại chỉ mất phần học máy.
5. Tải `leak_ml_outputs.zip` (hình 300 dpi PNG + PDF, bảng CSV + LaTeX, mô hình `.joblib`, `RESULTS_summary.md`).

| Muốn… | Sửa |
|---|---|
| chạy nhanh để thử | biến môi trường `LEAK_SMOKE=1`, hoặc `cfg.seeds = (42,)`, `cfg.k_grid` ngắn hơn |
| không dùng tập kiểm tra khoá | `cfg.holdout_frac = 0` (chỉ CV lồng) |
| E1 chỉ trên DEV | `cfg.e1_scope = "dev"` |
| thêm mô hình | `cfg.models = ("SVM", "RF", "XGB", "KNN", "LR", "MLP")` |
| khung ngắn/dài hơn | `cfg.frame_sec`, `cfg.hop_sec` (đặc trưng tự tính lại, cache theo cấu hình) |
| bớt điều kiện tín hiệu | `cfg.conditions` (phải giữ `C0_RAW` và điều kiện dùng cho E1) |
| độ nhạy ngưỡng tiền xử lý | chạy lại với `z_thr`, `kurt_thr`, `lsd_z_thr` khác và so sánh `T06` |

Kiểm tra ngoài Kaggle (CPU):

```bash
pip install numpy scipy pandas scikit-learn librosa soundfile PyWavelets xgboost shap matplotlib nbclient ipykernel
python tools/build_ml.py                                   # dựng lại .ipynb từ tools/ml_nb/*.py
LEAK_OUT=/tmp/leak_ml_smoke python tools/run_smoke.py notebooks/leak-ml-handcrafted-kaggle-v1.ipynb
```

---

## 17. Tài liệu tham khảo

**Âm học rò rỉ và ống nhựa**

- Almeida, F. C. L., Brennan, M. J., Joseph, P. F., Whitfield, S., Dray, S., & Paschoalini, A. T. (2014). On the acoustic filtering of the pipe and sensor in a buried plastic water pipe and its effect on leak detection: An experimental investigation. *Sensors*, 14(3), 5595–5610.
- Brennan, M. J., Gao, Y., & Joseph, P. F. (2007). On the relationship between time and frequency domain methods in time delay estimation for leak detection in water distribution pipes. *Journal of Sound and Vibration*, 304(1–2), 213–223.
- Butterfield, J. D., Krynkin, A., Collins, R. P., & Beck, S. B. M. (2017). Experimental investigation into vibro-acoustic emission signal processing techniques to quantify leak flow rate in plastic water distribution pipes. *Applied Acoustics*, 119, 146–155.
- Gao, Y., Brennan, M. J., Joseph, P. F., Muggleton, J. M., & Hunaidi, O. (2004). A model of the correlation function of leak noise in buried plastic pipes. *Journal of Sound and Vibration*, 277(1–2), 133–148.
- Gao, Y., Brennan, M. J., Joseph, P. F., Muggleton, J. M., & Hunaidi, O. (2005). On the selection of acoustic/vibration sensors for leak detection in plastic water pipes. *Journal of Sound and Vibration*, 283(3–5), 927–941.
- Hunaidi, O., & Chu, W. T. (1999). Acoustical characteristics of leak signals in plastic water distribution pipes. *Applied Acoustics*, 58(3), 235–254.
- Hunaidi, O., Chu, W., Wang, A., & Guan, W. (2000). Detecting leaks in plastic pipes. *Journal AWWA*, 92(2), 82–94.
- Martini, A., Troncossi, M., & Rivola, A. (2015). Automatic leak detection in buried plastic pipes of water supply networks by means of vibration measurements. *Shock and Vibration*, 2015, 165304.
- Martini, A., Troncossi, M., & Rivola, A. (2017). Leak detection in water-filled small-diameter polyethylene pipes by means of acoustic emission measurements. *Applied Sciences*, 7(1), 2.
- Muggleton, J. M., Brennan, M. J., & Pinnington, R. J. (2002). Wavenumber prediction of waves in buried pipes for water leak detection. *Journal of Sound and Vibration*, 249(5), 939–954.
- Puust, R., Kapelan, Z., Savic, D. A., & Koppel, T. (2010). A review of methods for leakage management in pipe networks. *Urban Water Journal*, 7(1), 25–45.

**Học máy cho phát hiện rò rỉ**

- Banjara, N. K., Sasmal, S., & Voggu, S. (2020). Machine learning supported acoustic emission technique for leakage detection in pipelines. *International Journal of Pressure Vessels and Piping*, 188, 104243.
- Cody, R. A., Tolson, B. A., & Orchard, J. (2020). Detecting leaks in water distribution pipes using a deep autoencoder and hydroacoustic spectrograms. *Journal of Computing in Civil Engineering*, 34(2), 04020001.
- Cody, R., Dey, P., & Narasimhan, S. (2020). Linear prediction for leak detection in water distribution networks. *Journal of Pipeline Systems Engineering and Practice*, 11(1), 04019043.
- El-Zahab, S., Mohammed Abdelkader, E., & Zayed, T. (2018). An accelerometer-based leak detection system. *Mechanical Systems and Signal Processing*, 108, 58–72.
- Fan, H., Tariq, S., & Zayed, T. (2022). Acoustic leak detection approaches for water pipelines. *Automation in Construction*, 138, 104226.
- Fares, A., Tijani, I. A., Rui, Z., & Zayed, T. (2023). Leak detection in real water distribution networks based on acoustic emission and machine learning. *Environmental Technology*, 44(25), 3850–3866.
- Kang, J., Park, Y.-J., Lee, J., Wang, S.-H., & Eom, D.-S. (2018). Novel leakage detection by ensemble CNN-SVM and graph-based localization in water distribution systems. *IEEE Transactions on Industrial Electronics*, 65(5), 4279–4289.
- Ning, F., Cheng, Z., Meng, D., & Wei, J. (2021). A framework combining acoustic features extraction method and random forest algorithm for gas pipeline leak detection and classification. *Applied Acoustics*, 182, 108255.
- Shukla, H., & Piratla, K. (2020). Leakage detection in water pipelines using supervised classification of acceleration signals. *Automation in Construction*, 117, 103256.
- Tijani, I. A., Abdelmageed, S., Fares, A., Fan, H., & Zayed, T. (2022). Improving the leak detection efficiency in water distribution networks using noise loggers. *Science of the Total Environment*, 821, 153530.
- Ullah, N., Ahmed, Z., & Kim, J.-M. (2023). Pipeline leakage detection using acoustic emission and machine learning algorithms. *Sensors*, 23(6), 3226.
- Xu, T., Zeng, Z., Huang, X., Li, J., & Feng, H. (2021). Pipeline leak detection based on variational mode decomposition and support vector machine using an interior spherical detector. *Process Safety and Environmental Protection*, 153, 167–177.

**Xử lý tín hiệu và khử nhiễu**

- Antoni, J. (2006). The spectral kurtosis: A useful tool for characterising non-stationary signals. *Mechanical Systems and Signal Processing*, 20(2), 282–307.
- Berouti, M., Schwartz, R., & Makhoul, J. (1979). Enhancement of speech corrupted by acoustic noise. *Proc. IEEE ICASSP*, 208–211.
- Boersma, P. (1993). Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound. *Proceedings of the Institute of Phonetic Sciences*, 17, 97–110.
- Boll, S. (1979). Suppression of acoustic noise in speech using spectral subtraction. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 27(2), 113–120.
- Donoho, D. L., & Johnstone, I. M. (1994). Ideal spatial adaptation by wavelet shrinkage. *Biometrika*, 81(3), 425–455.
- FitzGerald, D. (2010). Harmonic/percussive separation using median filtering. *Proc. 13th International Conference on Digital Audio Effects (DAFx-10)*.
- Iglewicz, B., & Hoaglin, D. C. (1993). *How to Detect and Handle Outliers*. ASQC Quality Press.
- Martin, R. (2001). Noise power spectral density estimation based on optimal smoothing and minimum statistics. *IEEE Transactions on Speech and Audio Processing*, 9(5), 504–512.
- Rabiner, L. R., & Schafer, R. W. (1978). *Digital Processing of Speech Signals*. Prentice-Hall.
- Rafii, Z., & Pardo, B. (2013). REpeating Pattern Extraction Technique (REPET): A simple method for music/voice separation. *IEEE Transactions on Audio, Speech, and Language Processing*, 21(1), 73–84.
- Sainburg, T., Thielk, M., & Gentner, T. Q. (2020). Finding, visualizing, and quantifying latent structure across diverse animal vocal repertoires. *PLOS Computational Biology*, 16(10), e1008228.
- Welch, P. (1967). The use of fast Fourier transform for the estimation of power spectra. *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70–73.

**Đặc trưng âm thanh**

- Atlas, L., & Shamma, S. A. (2003). Joint acoustic and modulation frequency. *EURASIP Journal on Advances in Signal Processing*, 2003(7), 668–675.
- Bandt, C., & Pompe, B. (2002). Permutation entropy: A natural complexity measure for time series. *Physical Review Letters*, 88(17), 174102.
- Coifman, R. R., & Wickerhauser, M. V. (1992). Entropy-based algorithms for best basis selection. *IEEE Transactions on Information Theory*, 38(2), 713–718.
- Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 28(4), 357–366.
- Giannakopoulos, T. (2015). pyAudioAnalysis: An open-source Python library for audio signal analysis. *PLOS ONE*, 10(12), e0144610.
- Higuchi, T. (1988). Approach to an irregular time series on the basis of the fractal theory. *Physica D*, 31(2), 277–283.
- Hjorth, B. (1970). EEG analysis based on time domain properties. *Electroencephalography and Clinical Neurophysiology*, 29(3), 306–310.
- Jiang, D.-N., Lu, L., Zhang, H.-J., Tao, J.-H., & Cai, L.-H. (2002). Music type classification by spectral contrast feature. *Proc. IEEE ICME*, 113–116.
- Katz, M. J. (1988). Fractals and the analysis of waveforms. *Computers in Biology and Medicine*, 18(3), 145–156.
- Makhoul, J. (1975). Linear prediction: A tutorial review. *Proceedings of the IEEE*, 63(4), 561–580.
- Mallat, S. G. (1989). A theory for multiresolution signal decomposition: The wavelet representation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 11(7), 674–693.
- Peeters, G. (2004). *A large set of audio features for sound description (similarity and classification) in the CUIDADO project*. IRCAM technical report.
- Randall, R. B. (2011). *Vibration-based Condition Monitoring*. Wiley.
- Zhou, X., Garcia-Romero, D., Duraiswami, R., Espy-Wilson, C., & Shamma, S. (2011). Linear versus mel frequency cepstral coefficients for speaker recognition. *Proc. IEEE ASRU*, 559–564.

**Chọn đặc trưng**

- Ambroise, C., & McLachlan, G. J. (2002). Selection bias in gene extraction on the basis of microarray gene-expression data. *PNAS*, 99(10), 6562–6566.
- Ding, C., & Peng, H. (2005). Minimum redundancy feature selection from microarray gene expression data. *Journal of Bioinformatics and Computational Biology*, 3(2), 185–205.
- Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning*, 63(1), 3–42.
- Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3, 1157–1182.
- Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. *Physical Review E*, 69, 066138.
- Nogueira, S., Sechidis, K., & Brown, G. (2018). On the stability of feature selection algorithms. *Journal of Machine Learning Research*, 18(174), 1–54.
- Peng, H., Long, F., & Ding, C. (2005). Feature selection based on mutual information: Criteria of max-dependency, max-relevance, and min-redundancy. *IEEE TPAMI*, 27(8), 1226–1238.
- Ross, B. C. (2014). Mutual information between discrete and continuous data sets. *PLOS ONE*, 9(2), e87357.
- Saeys, Y., Abeel, T., & Van de Peer, Y. (2008). Robust feature selection using ensemble feature selection techniques. *ECML PKDD 2008*, LNCS 5212, 313–325.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society B*, 58(1), 267–288.

**Mô hình, đánh giá và thống kê**

- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society B*, 57(1), 289–300.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). *Classification and Regression Trees*. Wadsworth.
- Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*, 11, 2079–2107.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proc. ACM SIGKDD*, 785–794.
- Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics*, 21, 6.
- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27.
- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves. *Biometrics*, 44(3), 837–845.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Hsu, C.-W., Chang, C.-C., & Lin, C.-J. (2003). *A practical guide to support vector classification*. Technical report, National Taiwan University.
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 15.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- Lundberg, S. M., Erion, G., Chen, H., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67.
- Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239–281.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432.
- Sun, X., & Xu, W. (2014). Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves. *IEEE Signal Processing Letters*, 21(11), 1389–1393.
- Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7, 91.

> Trước khi nộp bài, hãy đối chiếu lại số trang/tập của từng tài liệu với bản gốc (DOI) — danh mục trên được soạn
> để định hướng, một số chi tiết thư mục cần kiểm tra lại theo định dạng của tạp chí/hội nghị.
