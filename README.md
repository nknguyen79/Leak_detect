# Leak_detect — nhận dạng rò rỉ ống PVC áp lực thấp bằng âm thanh + học sâu

Pipeline nghiên cứu chạy trên Kaggle (GPU T4 x2), phục vụ bài báo *"Acoustic Leak Identification in
Low-Pressure PVC Water Distribution Pipes Using Deep Convolutional Neural Networks"*.

| Tệp | Vai trò |
|---|---|
| `notebooks/leak-cnn-pvc-kaggle-v7.ipynb` | **Phiên bản hiện hành.** Chạy trên Kaggle: `Add Data` → đặt `cfg.data_root` → `Run All`. |
| `notebooks/leak-cnn-pvc-kaggle-v6.ipynb` | Bản v6 gốc (kèm output của lần chạy trên 80 bản ghi) — nguồn để dựng v7. |
| `tools/build_v7.py`, `tools/v7_cells.py` | Dựng v7 **tái lập được** từ v6: mỗi chỉnh sửa là một phép thay thế có kiểm tra. |
| `tools/run_smoke.py` | Chạy end-to-end ở chế độ smoke (3 epoch, 2 fold, dữ liệu mô phỏng) để kiểm tra toàn tuyến. |
| `notebooks/leak-da-pvc-hdpe-dann-coral-kaggle.ipynb` | **Thích nghi miền PVC → HDPE**: 1D/2D-CNN + GRL (DANN) + Deep CORAL, kèm AdaBN, few-shot, Oracle. |
| `tools/build_da.py`, `tools/da_cells.py` | Dựng notebook thích nghi miền từ mã nguồn các cell (kiểm tra cú pháp từng cell). |

## Chạy

**Kaggle:** bật GPU T4 x2 và Internet, gắn dataset có hai thư mục `leak/` và `noleak/`, sửa
`cfg.data_root` ở mục 1.2, rồi `Run All`. Cấu hình mặc định (5 mô hình × 5 fold × 3 lần lặp CV
+ 7 biến thể ablation) mất khoảng 45–60 phút.

**Kiểm tra nhanh ngoài Kaggle (CPU):**

```bash
pip install torch torchaudio torchvision librosa soundfile scikit-learn pandas matplotlib nbclient ipykernel
LEAK_OUT=/tmp/leak_smoke python tools/run_smoke.py notebooks/leak-cnn-pvc-kaggle-v7.ipynb
```

Biến môi trường: `LEAK_SMOKE=1` (chế độ thử nhanh), `LEAK_DATA=<thư mục dữ liệu>` (rỗng → dữ liệu
mô phỏng), `LEAK_OUT=<thư mục đầu ra>`.

**Dựng lại v7 từ v6** (sau khi sửa `tools/`):

```bash
python tools/build_v7.py notebooks/leak-cnn-pvc-kaggle-v6.ipynb notebooks/leak-cnn-pvc-kaggle-v7.ipynb
```

## Notebook thích nghi miền PVC → HDPE (DANN + Deep CORAL)

Hiện thực tài liệu *domain_shift* và sơ đồ 4 giai đoạn (lệch miền → VMD + CWT + CNN/CBAM → GRL/CORAL →
không gian bất biến). Miền nguồn = PVC có nhãn; miền đích = HDPE, nhãn **chỉ** dùng để đánh giá.

| Thành phần | Nội dung |
|---|---|
| Tiền xử lý | 2 kHz, lọc dải 15–350 Hz, notch 50 Hz, VMD theo lô trên GPU, CWT Morlet tần số log (hoặc log-Mel), CMVN theo miền |
| $G_f$ | `CNN2D` (CWT + 4 khối Conv2d + CBAM) và `CNN1D` (sóng thô, bộ lọc dải học được + CBAM-1D) |
| Căn chỉnh | GRL với $\lambda_p = 2/(1+e^{-10p})-1$, bộ phân biệt miền $G_d$, Deep CORAL $\frac{1}{4d^2}\lVert C_S-C_T\rVert_F^2$ (tuỳ chọn Deep MMD) |
| So sánh | Source-only, AdaBN, CORAL, DANN, **DANN+CORAL**, DANN+CORAL+PhysAug, Few-shot FT (LR 1e-4/1e-5/1e-3), Oracle |
| Đánh giá | K-fold theo **bản ghi** HDPE (thích nghi không nhãn / kiểm thử tách rời), epoch cuối + ngưỡng 0,5 đăng ký trước, CI bootstrap theo bản ghi, ΔAUC ghép cặp + Holm, proxy A-distance / CORAL / MMD / t-SNE |

Ba chế độ tự nhận diện: `real` (có `PVC/` và `HDPE/`), `physics_sim` (chỉ có PVC thật — ví dụ bộ v7 — miền
đích giả lập bằng dịch tần Korteweg + suy hao nhớt đàn hồi trên các bản ghi PVC **tách rời**), `synthetic`.
HDPE chưa có nhãn (`unlabeled/` hoặc `DA_HIDE_TARGET_LABELS=1`) → bỏ đánh giá, xuất mô hình + bảng xếp hạng bản ghi nghi rò.

```bash
python tools/build_da.py notebooks/leak-da-pvc-hdpe-dann-coral-kaggle.ipynb          # dựng lại sau khi sửa tools/da_cells.py
LEAK_OUT=/tmp/da_smoke python tools/run_smoke.py notebooks/leak-da-pvc-hdpe-dann-coral-kaggle.ipynb
DA_SOURCE=<thư mục PVC> LEAK_OUT=/tmp/da_sim python tools/run_smoke.py notebooks/leak-da-pvc-hdpe-dann-coral-kaggle.ipynb
```

Biến môi trường thêm: `DA_SOURCE`, `DA_TARGET` (thư mục từng miền), `DA_HIDE_TARGET_LABELS=1`.

## Thay đổi chính v6 → v7

| Mã | Nội dung |
|---|---|
| B1 | Sửa lỗi Grad-CAM của CRNN làm v6 **dừng giữa chừng** (cuDNN RNN backward ở chế độ eval) — mục 10–16 của v6 chưa từng chạy. |
| B2 | Thực thi "chỉ chọn checkpoint sau đỉnh lịch LR" (v6 chỉ khai báo; LeakCNN1D bị chọn ở epoch ~0). |
| B3 | Không tự hạ `low_band_hz` 2800 → 1000 Hz dựa trên một kiểm định không còn ý nghĩa sau hiệu chỉnh đa so sánh. |
| B4–B8 | SpecAugment theo từng mẫu và từng nhánh; `DEPLOY_MODEL` dùng trước khi định nghĩa; ghi đè `sel_df`; lọc sai baseline RMS; ResNet18 nhận ảnh quá nhỏ; LeakCNN1D dùng được "độ to". |
| S1 | CV lặp (3 phân hoạch) + AUC của OOF trung bình qua các lần lặp. |
| S2 | Quy tắc gộp mức bản ghi đăng ký trước (`p_mean`, xác suất thô). |
| S3 | Baseline thống kê log-Mel + so sánh **ghép cặp** CNN vs baseline (Holm). |
| S4–S5 | Holm cho mọi kiểm định theo dải; giả thuyết dải thấp kiểm định bằng CI; tỉ lệ thể hiện dương kết luận 3 mức. |
| A1 | Ablation bật mặc định (bỏ FIR, loss cân bằng bản ghi, 16 kHz, frame 1/4 s, bỏ notch/high-pass), chọn theo validation-trong. |
| C1 | Tự sinh `site_map_template.csv` để bổ sung thông tin điểm khảo sát (leave-one-site-out). |
| D1–D2 | Gán nhãn theo từ; dò thư mục dữ liệu an toàn; không còn số liệu cũ viết cứng trong văn bản. |

Chi tiết: mục 16 của notebook v7.
