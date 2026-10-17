# Leak_detect — nhận dạng rò rỉ ống PVC áp lực thấp bằng âm thanh + học sâu

Pipeline nghiên cứu chạy trên Kaggle (GPU T4 x2), phục vụ bài báo *"Acoustic Leak Identification in
Low-Pressure PVC Water Distribution Pipes Using Deep Convolutional Neural Networks"*.

| Tệp | Vai trò |
|---|---|
| `notebooks/leak-cnn-pvc-kaggle-v7.ipynb` | **Phiên bản hiện hành.** Chạy trên Kaggle: `Add Data` → đặt `cfg.data_root` → `Run All`. |
| `notebooks/leak-cnn-pvc-kaggle-v6.ipynb` | Bản v6 gốc (kèm output của lần chạy trên 80 bản ghi) — nguồn để dựng v7. |
| `tools/build_v7.py`, `tools/v7_cells.py` | Dựng v7 **tái lập được** từ v6: mỗi chỉnh sửa là một phép thay thế có kiểm tra. |
| `tools/run_smoke.py` | Chạy end-to-end ở chế độ smoke (3 epoch, 2 fold, dữ liệu mô phỏng) để kiểm tra toàn tuyến. |
| `notebooks/leak-da-sim2real-sx-rx-kaggle.ipynb` | **Thích nghi miền mô phỏng → thực địa (Sx → Rx)**: 1D/2D-CNN + GRL (DANN, CDAN+E) + Deep CORAL, chọn dải tự động, DAug, few-shot, Oracle. |
| `notebooks/leak-da-pvc-hdpe-dann-coral-kaggle.ipynb` | Cùng mã nguồn, kịch bản **PVC → HDPE** (dải 15–350 Hz cố định, dịch tần Korteweg). |
| `tools/build_da.py`, `tools/da_cells.py` | Dựng hai notebook thích nghi miền từ một mã nguồn chung (`SCENARIO` + `PRESETS`; kiểm tra cú pháp từng cell). |

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

## Notebook thích nghi miền (Sx → Rx và PVC → HDPE)

Một mã nguồn (`tools/da_cells.py`), hai kịch bản chọn bằng `SCENARIO` ở mục 1.2:

| | `sim2real` (mặc định) | `pvc2hdpe` |
|---|---|---|
| Miền nguồn → đích | Sx: âm thanh leak/noleak **mô phỏng** → Rx: thu **thực địa** | ống PVC → ống HDPE |
| Cơ chế lệch chính | nhiễu nền, kênh đo, SNR, tính không dừng, nhãn yếu, tỉ lệ lớp, sai số mô phỏng | dịch tần theo vận tốc sóng (Korteweg) |
| Dải tần | **chọn tự động** (không dùng nhãn Rx): điểm = rò làm tăng năng lượng dải trên Sx × (1 − tách miền, min theo lớp) | cố định 15–350 Hz |
| Tần số lấy mẫu | suy ra từ dải, ≤ sr gốc nhỏ nhất của hai miền | 2 kHz |
| DAug (lô nguồn) | nhiễu theo **phổ nền Rx**, xung va đập, EQ kênh; dịch tần chỉ khi đo được r ≠ 1 | dịch tần 0,5–0,65 + suy hao + nhiễu nền HDPE |
| Chỉ có một miền | chỉ Rx → Sx sinh bằng **bộ mô phỏng tích hợp**; chỉ Sx → Rx giả lập "hiện trường hoá" | chỉ PVC → HDPE giả lập (vật lý) |

Chung cho cả hai: VMD theo lô trên GPU, CWT Morlet tần số log, CMVN theo miền, `CNN2D` + CBAM và `CNN1D`; phương pháp
Source-only, AdaBN, CORAL, DANN, **DANN+CORAL**, **CDAN+E+CORAL**, DANN+CORAL+DAug, **Few-shot chỉ miền đích**, Few-shot FT
(LR 1e-4/1e-5/1e-3), Oracle; lô nguồn cân bằng lớp; chia fold đích theo **bản ghi** hoặc **điểm khảo sát**
(`target_group_csv`); nhãn đích chỉ để đánh giá; epoch cuối + ngưỡng 0,5 đăng ký trước; CI bootstrap theo bản ghi, ΔAUC
ghép cặp + Holm; **giá trị của dữ liệu nguồn** (few-shot từ nguồn vs chỉ đích); **BBSE** ước lượng tỉ lệ rò của đích;
A-distance / CORAL / MMD / t-SNE; xuất TorchScript + manifest + bảng xếp hạng bản ghi nghi rò.

```bash
python tools/build_da.py                                                             # dựng cả hai notebook
LEAK_OUT=/tmp/s2r python tools/run_smoke.py notebooks/leak-da-sim2real-sx-rx-kaggle.ipynb          # dữ liệu mô phỏng hai miền
DA_TARGET=<thư mục Rx> LEAK_OUT=/tmp/s2r_rx python tools/run_smoke.py notebooks/leak-da-sim2real-sx-rx-kaggle.ipynb
LEAK_OUT=/tmp/p2h python tools/run_smoke.py notebooks/leak-da-pvc-hdpe-dann-coral-kaggle.ipynb
```

Biến môi trường: `DA_SOURCE`, `DA_TARGET` (thư mục từng miền), `DA_HIDE_TARGET_LABELS=1` (coi miền đích là không nhãn),
`DA_METHODS=source_only,dann_coral,...`, `DA_BACKBONES=CNN1D` (chạy nhanh), cùng `LEAK_SMOKE`, `LEAK_DATA`, `LEAK_OUT`.

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
