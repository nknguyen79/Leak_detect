# Leak_detect — nhận dạng rò rỉ ống PVC áp lực thấp bằng âm thanh

Hai pipeline nghiên cứu chạy trên Kaggle cho cùng tập dữ liệu (gậy nghe + micro, ống nhánh PVC Φ27, 0,5–2 bar,
hai thư mục `leak/` và `noleak/`):

| Tệp | Vai trò |
|---|---|
| `notebooks/leak-ml-handcrafted-kaggle-v1.ipynb` | **Đặc trưng thủ công + học máy** (SVM, RF, XGBoost, k-NN): xử lý tín hiệu 7 điều kiện, ~168 đặc trưng, dải tần (RQ1), xử lý vs thô (RQ2), tập đặc trưng tối thiểu (RQ3), độ bền/thời gian nghe (RQ4). Chỉ cần CPU. |
| `docs/phan-tich-quy-trinh-ML-dac-trung-thu-cong.md` | **Bài phân tích phương pháp**: vật lý bài toán, từng bước quy trình, thiết kế thí nghiệm, thống kê, ánh xạ hình/bảng vào bài báo, tài liệu tham khảo. |
| `tools/ml_nb/*.py`, `tools/build_ml.py` | Nguồn của notebook ML (định dạng "percent"); `build_ml.py` dựng lại `.ipynb`. |
| `notebooks/leak-cnn-pvc-kaggle-v7.ipynb` | Pipeline **học sâu** (5 kiến trúc CNN) — phiên bản hiện hành. |
| `notebooks/leak-cnn-pvc-kaggle-v6.ipynb` | Bản v6 gốc (kèm output của lần chạy trên 80 bản ghi) — nguồn để dựng v7. |
| `tools/build_v7.py`, `tools/v7_cells.py` | Dựng v7 **tái lập được** từ v6: mỗi chỉnh sửa là một phép thay thế có kiểm tra. |
| `tools/run_smoke.py` | Chạy một notebook end-to-end ở chế độ smoke trên dữ liệu mô phỏng để kiểm tra toàn tuyến. |

## Notebook đặc trưng thủ công + ML

**Kaggle:** `File → Import Notebook` → `notebooks/leak-ml-handcrafted-kaggle-v1.ipynb`; `Add Data` → dataset có
`leak/` và `noleak/`; sửa `cfg.data_root` ở mục 1.2 (sai thì tự dò trong `/kaggle/input`); `Accelerator = None`;
`Run All`. Cấu hình mặc định (7 điều kiện tín hiệu, CV 3×5 fold, CV lồng, tập kiểm tra khoá 20 %) mất khoảng
40–80 phút trên CPU Kaggle. Kết quả nằm trong `/kaggle/working/leak_ml/` và `leak_ml_outputs.zip`.

**Kiểm tra nhanh ngoài Kaggle (CPU):**

```bash
pip install numpy scipy pandas scikit-learn librosa soundfile PyWavelets xgboost shap matplotlib nbclient ipykernel
python tools/build_ml.py                     # sau khi sửa tools/ml_nb/*.py
LEAK_OUT=/tmp/leak_ml_smoke python tools/run_smoke.py notebooks/leak-ml-handcrafted-kaggle-v1.ipynb
```

`LEAK_SYNTH_N=<số bản ghi mỗi lớp>` đổi cỡ dữ liệu mô phỏng; `LEAK_SMOKE=0` chạy cấu hình đầy đủ.

## Notebook CNN (v7)

### Chạy

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

### Thay đổi chính v6 → v7

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
