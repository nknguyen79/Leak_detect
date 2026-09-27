# Leak_detect — nhận dạng rò rỉ ống PVC áp lực thấp bằng âm thanh + học sâu

Pipeline nghiên cứu chạy trên Kaggle (GPU T4 x2), phục vụ bài báo *"Acoustic Leak Identification in
Low-Pressure PVC Water Distribution Pipes Using Deep Convolutional Neural Networks"*.

| Tệp | Vai trò |
|---|---|
| `notebooks/leak-cnn-pvc-kaggle-v7.ipynb` | **Phiên bản hiện hành.** Chạy trên Kaggle: `Add Data` → đặt `cfg.data_root` → `Run All`. |
| `notebooks/leak-cnn-pvc-kaggle-v6.ipynb` | Bản v6 gốc (kèm output của lần chạy trên 80 bản ghi) — nguồn để dựng v7. |
| `tools/build_v7.py`, `tools/v7_cells.py` | Dựng v7 **tái lập được** từ v6: mỗi chỉnh sửa là một phép thay thế có kiểm tra. |
| `tools/run_smoke.py` | Chạy end-to-end ở chế độ smoke (3 epoch, 2 fold, dữ liệu mô phỏng) để kiểm tra toàn tuyến. |

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
