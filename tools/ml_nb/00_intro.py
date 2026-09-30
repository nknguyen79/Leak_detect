# %% [markdown]
# # Nhận dạng rò rỉ nước trên ống nhánh PVC áp lực thấp bằng đặc trưng âm thanh thủ công và học máy
#
# **Handcrafted Acoustic Features and Machine Learning for Leak Detection on Low-Pressure PVC Service Pipes:
# Frequency Band, Pre-processing and Minimal Feature Subset**
#
# ---
#
# ### Bối cảnh dữ liệu
#
# Âm thanh thu bằng **gậy nghe + micro độ nhạy cao** trên **ống nhánh PVC Φ27** (từ ống phân phối tới đồng hồ
# khách hàng), **áp suất 0,5–2 bar**; hai thư mục `leak/` và `noleak/`; mỗi mã `Rxxx` là một lần thu độc lập
# (có thể gồm nhiều tệp).
#
# ### Câu hỏi nghiên cứu / Research questions
#
# | | Câu hỏi | Thí nghiệm | Mục |
# |---|---|---|---|
# | **RQ1** | Dải tần nào chứa tín hiệu rò rỉ? | E1: LTAS, AUC từng dải 1/6 & 1/3 octave, quét dải bằng mô hình | 5 |
# | **RQ2** | Tín hiệu đã xử lý (lọc, loại bất thường, loại âm không liên quan, khử nhiễu) hay tín hiệu thô cho nhận dạng tốt hơn? | E2: 7 điều kiện tín hiệu × 4 mô hình, ghép cặp; E4c khẳng định trên TEST | 7, 9.4 |
# | **RQ3** | Tập đặc trưng nhỏ nhất nào giữ được hiệu năng? | E3: xếp hạng đồng thuận trong fold, đường cong AUC(k), độ ổn định | 8 |
# | **RQ4** | Mô hình có bền, diễn giải được và dùng được ngoài hiện trường không? | E4–E5: CV lồng, TEST khoá, SHAP, confound mức, rò rỉ dữ liệu, thời gian nghe tối thiểu | 9–10 |
#
# ### Quy trình / Pipeline
#
# ```
# WAV ─► QC tệp ─► C1 band-pass + notch ─► C2 loại khung bất thường ─► C3 loại âm không liên quan ─► C4/C5/C6 khử nhiễu
#                                          (va chạm, xung, bão hoà)      (tiếng nói, xe, sự kiện)       (so sánh 3 cách)
#   ─► khung 2 s ─► ~170 đặc trưng thủ công ─► [trong từng fold] chuẩn hoá → xếp hạng đồng thuận → top-k → SVM | RF | XGBoost | k-NN
#   ─► gộp xác suất về mức BẢN GHI ─► CV lặp nhóm theo bản ghi (DEV) + tập KIỂM TRA KHOÁ (TEST)
# ```
#
# ### Cách chạy trên Kaggle
#
# 1. `Add Data` → gắn dataset có hai thư mục `leak/` và `noleak/`; sửa `cfg.data_root` ở mục 1.2 (để sai thì
#    notebook tự dò trong `/kaggle/input`; không thấy dữ liệu → **chế độ mô phỏng**, số liệu không có giá trị).
# 2. Không cần GPU. `Settings → Accelerator = None` (CPU 4 lõi) là đủ; bật Internet nếu thiếu `xgboost`/`PyWavelets`.
# 3. `Run All`. Cấu hình mặc định (7 điều kiện, 3×5-fold, CV lồng) mất khoảng **40–80 phút** trên CPU Kaggle.
#    Đặc trưng được cache trong `leak_ml/cache/` nên chạy lại chỉ mất phần học máy.
# 4. Kết quả: `/kaggle/working/leak_ml/` (hình 300 dpi PNG+PDF, bảng CSV+LaTeX, `RESULTS_summary.md`,
#    `key_results.json`, mô hình `.joblib`) và bản nén `leak_ml_outputs.zip`.
#
# > Tài liệu phân tích phương pháp đi kèm: `docs/phan-tich-quy-trinh-ML-dac-trung-thu-cong.md` trong kho mã.

# %% [markdown]
# ## 0. Cơ sở tài liệu / Literature basis
#
# | Công trình | Đối tượng | Đặc trưng | Mô hình | Ghi chú dùng cho bài này |
# |---|---|---|---|---|
# | Hunaidi & Chu (1999), *Applied Acoustics* 58 | ống nhựa cấp nước | phổ tín hiệu rò rỉ | — | năng lượng rò rỉ ống nhựa tập trung ở tần số thấp, suy giảm mạnh theo khoảng cách |
# | Gao et al. (2005), *J. Sound Vib.* 283 | ống nhựa chôn | — | — | ống nhựa hoạt động như bộ lọc thông thấp; lựa chọn cảm biến |
# | Almeida et al. (2014), *Sensors* 14 | ống nhựa chôn | — | — | ảnh hưởng lọc của ống và cảm biến lên dải tần hữu ích |
# | Martini et al. (2015), *Shock Vib.*; (2017), *Appl. Sci.* 7 | ống nhựa/PE đường kính nhỏ | đặc trưng rung/AE theo dải | ngưỡng/thống kê | gần nhất với ống nhánh Φ27 |
# | Kang et al. (2018), *IEEE Trans. Ind. Electron.* 65 | mạng cấp nước | FFT/đặc trưng phổ | CNN + **SVM** | SVM là bộ phân loại chuẩn |
# | El-Zahab et al. (2018), *MSSP* 108 | mạng cấp nước (gia tốc kế) | thống kê thời gian/tần số | NB, DT, **k-NN**, SVM… | tổ hợp đặc trưng thủ công + ML cổ điển |
# | Cody, Dey & Narasimhan (2020), *J. Pipeline Syst. Eng. Pract.* 11 | ống cấp nước | **dự báo tuyến tính (LPC)** | phát hiện bất thường | cơ sở cho nhóm đặc trưng LPC |
# | Ning et al. (2021), *Applied Acoustics* 182 | ống khí | MFCC + đặc trưng phổ/thời gian | **Random Forest** | quy trình trích + chọn đặc trưng + RF |
# | Xu et al. (2021), *Process Saf. Environ. Prot.* 153 | ống | VMD + năng lượng/entropy | **SVM** | đặc trưng phân rã đa phân giải |
# | Fan, Tariq & Zayed (2022), *Automation in Construction* 138 | tổng quan | — | — | SVM, ANN, RF, k-NN là các mô hình ML dùng nhiều nhất |
# | Tijani et al. (2022), *Sci. Total Environ.* 821; Fares et al. (2023), *Environ. Technol.* 44 | mạng thực (noise logger/AE) | đặc trưng thời gian + tần số | SVM, RF, **boosting**, k-NN… | so sánh nhiều bộ phân loại, chọn đặc trưng |
# | Cody, Tolson & Orchard (2020), *J. Comput. Civ. Eng.* 34 | mạng thực | phổ đồ thuỷ âm | autoencoder | mốc so sánh học sâu |
#
# Khoảng trống: rất ít công trình về **ống nhánh PVC đường kính nhỏ, áp lực 0,5–2 bar, thu bằng gậy nghe**;
# phần lớn chia train/test ngẫu nhiên theo đoạn (rò rỉ dữ liệu) và không kiểm định thống kê ở mức bản ghi;
# hầu như không công trình nào **đo trực tiếp** tác động của từng bước tiền xử lý hay tìm **dải tần** và **tập
# đặc trưng tối thiểu** bằng giao thức không thiên lệch. Danh mục tài liệu đầy đủ: xem tài liệu phân tích đi kèm.
