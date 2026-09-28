# Tổng quan tài liệu: Quy trình nhận dạng – phân loại rò rỉ nước bằng âm thanh/rung động và khoảng trống nghiên cứu cho ống nhựa PVC/HDPE áp lực thấp (0,5–2 bar)

> **Phạm vi.** Tổng hợp các công trình trên tạp chí uy tín (Elsevier, ASCE, IEEE, SAGE, MDPI, IWA) và nguồn trong nước về chuỗi xử lý
> *thu dữ liệu `leak`/`noleak` → tiền xử lý tín hiệu → trích xuất/chọn đặc trưng → mô hình ML/DL → kiểm chứng, so sánh*,
> từ đó xác định khoảng trống học thuật cho đề tài phát hiện rò rỉ trên ống PVC, HDPE ở áp lực 0,5–2 bar.
>
> **Độ tin cậy của trích dẫn.** Mục có ký hiệu ✔ đã được đối chiếu thư mục (tác giả, năm, tạp chí, tập/số bài) qua tra cứu trực tuyến
> ngày 28/09/2026. Mục có ký hiệu ◐ là tài liệu kinh điển được trích từ hiểu biết chung, **cần tra lại số trang/tập trước khi đưa vào bài báo**.
> Số liệu định lượng (độ chính xác, dải áp lực…) chỉ được nêu khi đã kiểm chứng được.

---

## 1. Bức tranh chung

Phát hiện rò rỉ bằng âm học (Acoustic Leak Detection – ALD) dựa trên tiếng ồn sinh ra khi dòng nước phụt qua lỗ/vết nứt: dòng rối, va đập tia nước vào môi trường xung quanh
và dao động thành ống. Tín hiệu này truyền dọc ống chủ yếu bằng sóng chất lỏng (fluid-borne wave) và được ghi bằng **gia tốc kế** (trên thành ống), **hydrophone** (trong nước),
cảm biến áp suất động, cảm biến phát xạ âm (AE) hoặc micro địa âm.

Ba nhận định nền tảng từ tài liệu:

1. **Ống nhựa khó hơn ống kim loại rất nhiều.** Thành ống nhựa có mô-đun đàn hồi thấp, tính nhớt đàn hồi cao nên sóng bị suy giảm mạnh và năng lượng dồn về tần số thấp
   (thường < 1 kHz, nhiều trường hợp < 250 Hz) — điều đã được chỉ ra từ Hunaidi & Chu (1999) và được mô hình hóa bởi nhóm Southampton (Gao, Brennan, Muggleton, Joseph).
2. **Áp lực quyết định năng lượng và băng tần của tiếng rò.** Áp lực cao → vận tốc tia lớn → biên độ và thành phần tần số cao tăng. Hầu hết thí nghiệm công bố
   dùng áp lực ≥ 2,5–3 bar; vùng 0,5–2 bar — điển hình cho mạng phân phối cuối nguồn, khu dân cư, cấp nước nông thôn ở Việt Nam — gần như chưa được mô tả có hệ thống.
3. **Làn sóng ML/DL (2018–2026)** đưa độ chính xác báo cáo lên 90–99 %, nhưng các tổng quan mới (Liu, Zayed & Xiao 2025; Wu và cs. 2024) cảnh báo về
   dữ liệu nhỏ, phân chia train/test rò rỉ thông tin, thiếu kiểm chứng chéo miền (cross-domain) và thiếu tính giải thích.

---

## 2. Danh mục và phân tích các công trình tiêu biểu

### 2.1. Nền tảng vật lý – đặc tính tiếng rò trong ống nhựa

| # | Công trình | Nội dung chính | Đánh giá / liên hệ đề tài |
|---|---|---|---|
| 1 | ✔ **Hunaidi O., Chu W.T. (1999).** *Acoustical characteristics of leak signals in plastic water distribution pipes.* Applied Acoustics 58: 235–254. doi:10.1016/S0003-682X(99)00013-4 | Khảo sát phổ tiếng rò theo loại rò, lưu lượng rò, **áp lực ống**, mùa; đo hệ số suy giảm và vận tốc truyền theo tần số trong ống nhựa. Năng lượng tập trung ở tần số thấp; áp lực càng cao thì thành phần tần số cao càng mạnh. | Tài liệu gốc bắt buộc trích dẫn. Là cơ sở vật lý cho giả thuyết "dải thấp" của đề tài. Tuy nhiên chỉ dùng phân tích phổ – tương quan, không có ML; dải áp lực khảo sát không tập trung vào < 2 bar. |
| 2 | ◐ **Gao Y., Brennan M.J., Joseph P.F., Muggleton J.M., Hunaidi O. (2004).** *A model of the correlation function of leak noise in buried plastic pipes.* J. Sound Vib. 277: 133–148. | Mô hình giải tích hàm tương quan chéo tiếng rò trong ống nhựa chôn, tính tới suy giảm và lọc của ống. | Cơ sở để giải thích vì sao phổ tín hiệu thu được là "phổ rò × hàm truyền ống × đáp ứng cảm biến". |
| 3 | ◐ **Gao Y., Brennan M.J., Joseph P.F. và cs. (2005).** *On the selection of acoustic/vibration sensors for leak detection in plastic water pipes.* J. Sound Vib. 283: 927–941. | So sánh hydrophone – gia tốc kế – cảm biến vận tốc trong ống nhựa; hydrophone cho băng thông/SNR tốt hơn ở tần số thấp. | Liên quan trực tiếp đến việc chọn cảm biến ở áp lực thấp, khi SNR vốn đã kém. |
| 4 | ✔ **Scussel O., Brennan M.J., de Almeida F.C.L., Iwanaga M.K., Muggleton J.M., Joseph P.F., Gao Y. (2023).** *Key factors that influence the frequency range of measured leak noise in buried plastic water pipes: theory and experiment.* Acoustics 5(2), doi:10.3390/acoustics5020029 | Mô hình ống–nước–đất + đáp ứng cảm biến để **dự báo băng thông** tiếng rò đo được; băng thông phụ thuộc khoảng cách cảm biến, vận tốc sóng chất lỏng, loại cảm biến. | Cho phép chọn dải lọc/đặc trưng **có căn cứ vật lý** thay vì cảm tính — có thể kết hợp với chọn dải bằng dữ liệu trong đề tài. |
| 5 | ✔ **Shekofteh M.R., Horoshenkov K.V., John E., Gowdy C., Blenkharn A., Boxall J.B. (2026).** *Calibrated acoustic leak signatures in pressurised plastic water pipes: a laboratory analysis.* Sensors 26(14): 4325, doi:10.3390/s26144325 | Dữ liệu gia tốc thành ống **đã hiệu chuẩn** cho nhiều cấu hình rò ở áp lực tĩnh **2,8–4,2 bar**. Phổ công suất biến thiên tới 5 bậc độ lớn; **rò qua van/đầu phun (thường dùng trong phòng thí nghiệm) sinh năng lượng lớn hơn đáng kể so với lỗ khoan trực tiếp hoặc vết nứt**. | Rất quan trọng: (i) khẳng định vẫn *thiếu* dữ liệu chữ ký rò hiệu chuẩn; (ii) dải áp lực vẫn > 2,8 bar; (iii) cảnh báo rằng dữ liệu `leak` tạo bằng van có thể "dễ" hơn thực tế → nguy cơ đánh giá lạc quan. |
| 6 | ✔ **Hamamed N., Mechri C., Mhammedi T., Yaakoubi N., El Guerjouma R., Bouaziz S., Haddar M. (2023).** *Comparative study of leak detection in PVC water pipes using ceramic, polymer, and SAW sensors.* Sensors 23(18): 7717, doi:10.3390/s23187717 | So sánh PZT, PVDF, SAW trên ống **PVC**; với lỗ rò nhỏ (1–2 mm) PVDF nhạy nhất, rồi SAW, rồi PZT. | Gợi ý cảm biến giá rẻ cho ống PVC; chưa có tầng ML và chưa khảo sát áp lực thấp có hệ thống. |
| 7 | ✔ **Butterfield J.D., Krynkin A., Collins R.P., Beck S.B.M. (2017).** *Experimental investigation into vibro-acoustic emission signal processing techniques to quantify leak flow rate in plastic water distribution pipes.* Applied Acoustics 119: 146–155. | Trên ống MDPE: RMS tín hiệu tương quan mạnh với lưu lượng rò; môi trường bao quanh (đất lấp) ảnh hưởng mạnh tới tín hiệu. | Cơ sở cho đặc trưng "độ to" (RMS/band-energy) và cho bài toán mở rộng: **ước lượng mức độ rò**, không chỉ nhị phân. |
| 8 | ✔ **Butterfield J.D., Meruane V., Collins R.P., Meyers G., Beck S.B.M. (2018).** *Prediction of leak flow rate in plastic water distribution pipes using vibro-acoustic measurements.* Structural Health Monitoring 17(4), doi:10.1177/1475921717723881 | Dùng học máy (mạng nơ-ron/SVM) dự báo lưu lượng rò trong ống nhựa từ tín hiệu rung-âm. | Một trong số ít công trình ML trên **ống nhựa** có kiểm soát áp lực/lưu lượng. |

### 2.2. Xử lý tín hiệu và trích xuất đặc trưng (ML "cổ điển")

| # | Công trình | Chuỗi xử lý | Đánh giá |
|---|---|---|---|
| 9 | ✔ **Shukla H., Piratla K. (2020).** *Leakage detection in water pipelines using supervised classification of acceleration signals.* Automation in Construction 117: 103256. | Gia tốc kế → biến đổi wavelet liên tục (scalogram) → CNN; độ chính xác ~**86 %**. | Thẳng thắn về giới hạn (chưa phân tích độ nhạy). Minh chứng rằng bài toán thực nghiệm không "dễ" như nhiều báo cáo 99 %. |
| 10 | ✔ **Bùi Quý Thắng (Quy T.B.), Kim J.-M. (2020).** *Leak detection in a gas pipeline using spectral portrait of acoustic emission signals.* Measurement 152: 107403. | Tín hiệu AE → "chân dung phổ" (spectral portrait) → khoảng cách Kullback–Leibler → SVM đa lớp nhận dạng kích thước rò. | Tác giả người Việt (ĐH Ulsan). Ý tưởng dùng **độ lệch phân bố phổ so với trạng thái không rò** rất hợp với ống nhựa áp thấp, nhưng thực hiện trên ống khí/thép. |
| 11 | ◐ **Cody R., Harmouche J., Narasimhan S. (2018).** *Leak detection in water distribution pipes using singular spectrum analysis.* Mechanical Systems and Signal Processing 111: 676–. | Hydrophone → SSA tách thành phần → phát hiện bất thường. | Tiền xử lý khử nhiễu hiệu quả với SNR thấp; đáng thử với tín hiệu áp thấp. |
| 12 | ✔ **Aghashahi M., Sela L., Banks M.K. (2023).** *Benchmarking dataset for leak detection and localization in water distribution systems.* Data in Brief 48: 109148. | **Bộ dữ liệu mở**: 280 bản ghi 30 s, ống **PVC** Ø152,4 mm, 47 m; gia tốc kế & áp suất động 51,2 kHz, hydrophone 8 kHz; 4 kiểu rò (lỗ, nứt dọc, nứt vòng, gioăng) + không rò; mạng vòng/nhánh; 6 điều kiện nền. | Nguồn **kiểm chứng ngoài** (external validation) rất giá trị cho đề tài. Hạn chế: không phải HDPE, không quét áp lực 0,5–2 bar. |

### 2.3. Học sâu (Deep Learning) cho nhận dạng rò rỉ bằng âm thanh

| # | Công trình | Mô hình / đặc trưng | Đánh giá |
|---|---|---|---|
| 13 | ✔ **Kang J., Park Y.-J., Lee J., Wang S.-H., Eom D.-S. (2018).** *Novel leakage detection by ensemble CNN-SVM and graph-based localization in water distribution systems.* IEEE Trans. Industrial Electronics 65(5): 4279–4289. | 1D-CNN học đặc trưng từ tín hiệu rung + SVM; định vị bằng đồ thị. | Công trình khởi đầu cho 1D-CNN trong ALD; dữ liệu mạng thực (Hàn Quốc), chủ yếu ống kim loại. |
| 14 | ✔ **Cody R., Tolson B., Orchard J. (2020).** *Detecting leaks in water distribution pipes using a deep autoencoder and hydroacoustic spectrograms.* J. Computing in Civil Engineering 34(2): 04020001. | Phổ đồ hydrophone → autoencoder sâu (học không giám sát/phát hiện bất thường); ~**97,2 %**. | Hướng **học một lớp (chỉ dữ liệu `noleak`)** rất thực tế vì dữ liệu `leak` hiếm. |
| 15 | ✔ **Ahmad S., Ahmad Z., Kim C.-H., Kim J.-M. (2022).** *A method for pipeline leak detection based on acoustic imaging and deep learning.* Sensors 22(4): 1562. | AE → CWT (ảnh âm học) → CNN. | Đại diện dòng "tín hiệu → ảnh thời–tần → CNN". Nhóm Ulsan có nhiều đồng tác giả Việt Nam trong chuỗi bài tiếp theo (1D-DenseNet 2025, scalogram tăng cường 2023). |
| 16 | ✔ **Wu Y., Ma X., Guo G., Jia T., Huang Y., Liu S., Fan J., Wu X. (2024).** *Advancing deep learning-based acoustic leak detection methods towards application for WDS from a data-centric perspective.* Water Research 261: 121999. | Tiếp cận **data-centric**: chất lượng nhãn, cân bằng, tăng cường dữ liệu quyết định hiệu năng hơn là kiến trúc. | Hỗ trợ trực tiếp cho thiết kế tập `leak/noleak` của đề tài. |
| 17 | ✔ **Xu Z., Liu H., Fu G., Zheng R., Zayed T., Liu S. (2025).** *Interpretable deep learning for acoustic leak detection in water distribution systems.* Water Research 273: 123076. | Mô hình DL có **giải thích** (chỉ ra dải tần/đoạn thời gian mô hình dựa vào). | Cho phép kiểm tra mô hình "nghe" đúng tiếng rò hay chỉ học nhiễu nền — cần thiết khi SNR thấp. |
| 18 | ✔ **Wang C., Liu Z., Fei J., Long Z., Wang P., Yu T. (2025).** *Evaluating the generalizability and transferability of acoustic leak detection models for water distribution networks.* Water Research 286: 124273. | Đánh giá **chéo miền**: đa vùng, chéo vật liệu, chéo đường kính; fine-tuning đạt 96,8 % (chéo vật liệu) và 96,2 % (chéo đường kính), giảm 50 % dữ liệu miền đích. Chuyển giao **bất đối xứng**: kim loại → phi kim tốt hơn chiều ngược lại do miền nguồn có dải tần rộng hơn. | Chứng cứ trực tiếp rằng **vật liệu ống là một "miền"** — nhưng chưa xét **áp lực** như một trục dịch chuyển miền, chưa tách PVC ↔ HDPE. |
| 19 | ✔ **Zhang C., Alexander B.J., Stephens M.L., Lambert M.F., Gong J. (2023).** *A convolutional neural network for pipe crack and leak detection in smart water network.* Structural Health Monitoring. doi:10.1177/14759217221080198 | CNN phân biệt nứt/rò từ dữ liệu giám sát thủy lực – âm học mạng thông minh (Adelaide). | Chỉ ra nhu cầu phân loại **mức độ/kiểu hư hỏng**, không chỉ có/không. |
| 20 | Các công trình 2024–2026 (✔ tiêu đề/tạp chí; tác giả cần tra thêm): *Leakage detection in WDS based on logarithmic spectrogram CNN for continuous monitoring* (J. Water Resour. Plann. Manage. 150(6), 2024); *Enhancing automated ALD using ensemble ML* (JWRPM 151(3), 2025); *Acoustic identification of water supply pipe leakage based on bispectrum analysis* (J. Pipeline Syst. Eng. Pract. 16(3), 2025); *Leveraging 1-D deep learning for robust leak detection using acceleration data* (JPSEP 16(4), 2025); *Acoustic water pipe leak detection using transformer-based VAE* (NDT&E Int., 2026). | Log-spectrogram CNN, ensemble, bispectrum (đặc trưng phi tuyến bậc cao), 1D-CNN trên gia tốc, Transformer-VAE học trên dữ liệu bình thường. | Xu hướng: (a) đặc trưng bậc cao/phi tuyến, (b) học không giám sát, (c) Transformer. Hầu hết trên mạng đô thị Trung Quốc, áp lực vận hành thông thường. |

### 2.4. Các tổng quan (review) nên dùng làm khung

| # | Công trình | Ghi chú |
|---|---|---|
| 21 | ✔ **Fan H., Tariq S., Zayed T. (2022).** *Acoustic leak detection approaches for water pipelines.* Automation in Construction 138: 104226. | Phân tích khoa học lượng (Scopus, 30 năm) + phân loại thiết bị, xử lý tín hiệu, phương pháp phát hiện thời gian thực/không thời gian thực. |
| 22 | ✔ **Hu Z., Tariq S., Zayed T. (2021).** *A comprehensive review of acoustic based leak localization method in pressurized pipelines.* Mech. Syst. Signal Process. 161: 107994. | Tập trung định vị; hữu ích cho phần mở rộng. |
| 23 | ✔ **Liu R., Zayed T., Xiao R. (2025).** *A critical systematic review of ML-based models for water leak detection using vibroacoustic technology.* Eng. Appl. Artif. Intell. | Tổng quan phê phán mới nhất về ML trên dữ liệu rung-âm: rủi ro đánh giá, thiếu dữ liệu chuẩn, thiếu kiểm chứng hiện trường. |

### 2.5. Nghiên cứu trong nước

Tra cứu trên các nguồn trong nước (Tạp chí KH&CN – ĐH Đà Nẵng, Tạp chí Khoa học Thủy lợi, VJOL, báo chí KH&CN) cho thấy:

- **Chưa tìm thấy** công bố bình duyệt trong nước nào xây dựng trọn quy trình *âm thanh → đặc trưng → ML* để phân loại `leak/noleak` trên ống PVC/HDPE.
- Các công trình trong nước hiện có chủ yếu là: (i) hệ thống **đo lưu lượng – áp lực + truyền 4G** để chẩn đoán rò rỉ đa điểm (đề tài NCKH sinh viên ĐH Sư phạm Kỹ thuật – ĐH Đà Nẵng, giải nhất cấp ĐH Đà Nẵng 2023–2024 — kết quả truyền thông, chưa thấy bài báo bình duyệt); (ii) quản lý thất thoát nước (NRW) theo DMA; (iii) thiết bị dò rò thương mại (nhập khẩu).
- **"Mạng Hà Nội" (Hanoi network)** xuất hiện dày đặc trong tài liệu quốc tế (ví dụ các công trình định vị rò bằng CNN + suy luận Bayes trên bản đồ áp lực, UPC Barcelona) **chỉ là mạng chuẩn thủy lực mô phỏng** (Fujiwara & Khang, 1990), *không phải* dữ liệu âm thanh thực đo ở Việt Nam — cần tránh nhầm lẫn khi viết phần tổng quan.
- Nhà nghiên cứu người Việt có công bố quốc tế liên quan chủ yếu thuộc nhóm ĐH Ulsan (Hàn Quốc), trên **ống khí/thép với cảm biến AE**, không phải ống cấp nước nhựa áp thấp.
- Bối cảnh chuẩn: **TCVN 13606:2023** quy định áp lực tự do tối thiểu tại điểm lấy nước vào nhà ≥ 10 m cột nước (~1 bar) — tức dải 0,5–2 bar của đề tài *chính là* vùng vận hành thực tế ở cuối mạng, nơi ống uPVC/HDPE chiếm ưu thế.

➡ Đây là một **khoảng trống rõ ràng ở cấp quốc gia**: thiếu dữ liệu, thiếu quy trình được kiểm chứng và thiếu đánh giá cho điều kiện Việt Nam.

---

## 3. Tổng hợp quy trình chuẩn (từ tài liệu) và những điểm yếu phổ biến

| Khâu | Thực hành phổ biến trong tài liệu | Điểm yếu thường gặp |
|---|---|---|
| **1. Thu dữ liệu** | Gia tốc kế/hydrophone/AE; fs 8–51,2 kHz; bản ghi 10–60 s; rò tạo bằng van, đầu phun hoặc lỗ khoan; `noleak` = ống vận hành bình thường. | Rò bằng van "ồn" hơn rò thật (Shekofteh 2026); `noleak` ít đa dạng (thiếu tiếng bơm, giao thông, vòi mở); không ghi áp lực/lưu lượng kèm theo; số bản ghi độc lập nhỏ (vài chục – vài trăm). |
| **2. Tiền xử lý** | Bỏ DC, lọc thông cao/thông dải, notch điện lưới 50/60 Hz, chuẩn hóa, khử nhiễu wavelet/SSA/phổ trừ, cắt khung 0,25–2 s, tăng cường dữ liệu (SpecAugment, trộn nhiễu). | Chọn dải lọc tùy ý; chuẩn hóa biên độ có thể **xóa mất thông tin "độ to"** vốn tương quan với lưu lượng rò (Butterfield 2017). |
| **3. Đặc trưng** | Miền thời gian (RMS, kurtosis, crest factor, ZCR, entropy); miền tần (PSD, năng lượng dải, centroid, rolloff, flatness); cepstral (MFCC, GFCC); thời–tần (STFT, log-Mel, CWT scalogram, HHT/EMD); bậc cao (bispectrum). | Đặc trưng thiết kế cho giọng nói (Mel, MFCC) nén mạnh vùng < 1 kHz theo cách **không tối ưu cho tiếng rò ống nhựa**; ít công trình liên hệ đặc trưng với vật lý (băng thông dự báo). |
| **4. Chọn đặc trưng** | ANOVA/t-test, mRMR, tầm quan trọng RF, SHAP, PCA. | Hiếm khi hiệu chỉnh so sánh bội (Holm/Bonferroni) khi kiểm định nhiều dải. |
| **5. Mô hình** | SVM, kNN, RF, XGBoost; 1D-CNN, 2D-CNN (log-Mel/CWT), CRNN, autoencoder, Transformer/VAE; ensemble. | So sánh không công bằng (khác tiền xử lý, khác ngân sách tinh chỉnh); ít baseline thống kê mạnh. |
| **6. Đánh giá** | Accuracy, precision/recall, F1, AUC; k-fold CV; đôi khi hold-out theo vị trí. | **Rò rỉ dữ liệu**: các khung cắt từ *cùng một bản ghi* nằm ở cả train và test → độ chính xác 97–99 % bị thổi phồng; hiếm có CV lặp, khoảng tin cậy, kiểm định ghép cặp, đánh giá ở mức **bản ghi** và **leave-one-site/condition-out**. |

---

## 4. Khoảng trống học thuật áp dụng cho đề tài (PVC, HDPE, 0,5–2 bar)

### G1. Thiếu đặc tả chữ ký âm học của rò rỉ ống nhựa ở áp lực thấp
Các nghiên cứu hiệu chuẩn mới nhất vẫn ở 2,8–4,2 bar (Shekofteh 2026); Hunaidi & Chu (1999) chỉ ra phụ thuộc áp lực nhưng không quét dày vùng < 2 bar.
**Đóng góp khả dĩ:** đo có hệ thống phổ công suất, SNR, băng thông hiệu dụng theo lưới (áp lực 0,5; 1,0; 1,5; 2,0 bar) × (PVC, HDPE) × (kiểu rò) × (lưu lượng rò);
xây **đường cong giới hạn phát hiện** (lưu lượng rò nhỏ nhất phát hiện được với AUC ≥ ngưỡng) theo áp lực. Đây là kết quả vật lý – thống kê có giá trị công bố độc lập.

### G2. Áp lực như một trục dịch chuyển miền (domain shift) chưa được nghiên cứu
Wang và cs. (2025) chỉ xét vùng, vật liệu, đường kính. **Chưa có** công trình nào huấn luyện ở một mức áp lực và đánh giá ở mức khác.
**Đóng góp khả dĩ:** giao thức *leave-one-pressure-out*; so sánh (a) đặc trưng chuẩn hóa bất biến áp lực (hình dạng phổ, tỉ lệ năng lượng dải),
(b) mô hình có điều kiện áp lực (pressure-conditioned, FiLM), (c) fine-tuning ít mẫu. Trả lời câu hỏi: *mô hình học ở 2 bar có còn dùng được ở 0,5 bar?*

### G3. Chuyển giao PVC ↔ HDPE
PVC cứng hơn (vận tốc sóng chất lỏng cao hơn, suy giảm thấp hơn) so với HDPE (nhớt đàn hồi mạnh, suy giảm lớn, dải tần hẹp hơn). Wang (2025) cho thấy chuyển giao bất đối xứng
giữa kim loại và phi kim do độ rộng dải tần miền nguồn; **hiện tượng tương tự giữa hai loại nhựa chưa được kiểm chứng**.
**Đóng góp khả dĩ:** ma trận chuyển giao PVC→HDPE, HDPE→PVC ở từng mức áp lực; giải thích bằng dải tần hiệu dụng (liên hệ G1 và mô hình Scussel 2023).

### G4. Lựa chọn dải tần/đặc trưng có căn cứ vật lý cho tín hiệu áp thấp
Log-Mel/MFCC được mượn từ nhận dạng tiếng nói. Với ống nhựa áp thấp, thông tin nằm ở vài chục Hz đến ~1–3 kHz.
**Đóng góp khả dĩ:** so sánh log-Mel vs. **thang tần tuyến tính/log dải thấp**, CQT, GFCC, CWT, bispectrum; ablation theo dải; đối chiếu dải mô hình
"chú ý" (Grad-CAM/occlusion, SHAP) với băng thông dự báo lý thuyết (Scussel 2023) → *physics-informed band selection*. Kiểm định giả thuyết "dải thấp"
có hiệu chỉnh so sánh bội (repo v7 đã có khung Holm + CI — có thể trình bày như đóng góp phương pháp).

### G5. Tính hiện thực của rò rỉ và tính đa dạng của lớp `noleak`
Rò tạo bằng van cho năng lượng cao hơn rò thật (Shekofteh 2026) → nguy cơ mô hình "học" van.
**Đóng góp khả dĩ:** tập dữ liệu có các kiểu rò điển hình ở Việt Nam: nứt dọc ống uPVC, hở mối dán keo, hở ren/khớp nối nhanh, hở mối hàn/khớp nối cơ khí HDPE, lỗ nhỏ;
lớp `noleak` gồm nhiễu thực tế: máy bơm tăng áp hộ gia đình, bồn nước mái, vòi đang mở (tiêu thụ hợp pháp — "kẻ giả mạo" khó nhất), giao thông, bọt khí/cấp nước gián đoạn.
Đánh giá theo **từng loại nhiễu gây nhầm** (confusion theo điều kiện).

### G6. Giao thức đánh giá chặt chẽ – chống rò rỉ dữ liệu
Nhiều bài báo 97–99 % dùng chia khung ngẫu nhiên. Tổng quan phê phán của Liu, Zayed & Xiao (2025) nêu rõ vấn đề này.
**Đóng góp khả dĩ:** đề xuất và công bố giao thức: chia theo **bản ghi**, CV lặp phân tầng theo nhóm, gộp xác suất mức bản ghi đăng ký trước,
**leave-one-condition-out** (áp lực/vật liệu/kiểu rò/vị trí), báo cáo AUC + MCC + khoảng tin cậy bootstrap, kiểm định ghép cặp giữa mô hình (DeLong/McNemar/Wilcoxon + Holm),
và **so với baseline thống kê mạnh** (log-Mel + LR/RF). Cho thấy mức "thổi phồng" khi chia khung ngẫu nhiên — một kết quả phương pháp luận dễ được chấp nhận ở tạp chí tốt.

### G7. Dữ liệu nhỏ: học tự giám sát, mô hình âm thanh tiền huấn luyện, học một lớp
Chưa có đánh giá các mô hình âm thanh tiền huấn luyện (PANNs, YAMNet/VGGish, AST, BEATs, wav2vec-style) cho rò rỉ ống nhựa áp thấp; học một lớp (Cody 2020; Transformer-VAE 2026) chưa được thử ở áp thấp.
**Đóng góp khả dĩ:** so sánh *CNN huấn luyện từ đầu* vs *embedding tiền huấn luyện + bộ phân loại nhẹ* vs *phát hiện bất thường chỉ dùng `noleak`* theo số lượng bản ghi `leak` (đường cong học).

### G8. Cảm biến giá rẻ và triển khai biên
Hamamed (2023) cho thấy PVDF/SAW khả thi trên PVC nhưng chưa ghép ML; gia tốc kế MEMS và micro tiếp xúc giá rẻ chưa được đánh giá ở áp thấp.
**Đóng góp khả dĩ:** so sánh cảm biến chuẩn đo lường vs. MEMS/piezo giá rẻ; lượng tử hóa mô hình, chạy trên vi điều khiển (TinyML) — phù hợp năng lực đầu tư của công ty cấp nước trong nước.

### G9. Từ phát hiện sang mức độ rò
Butterfield (2017, 2018) liên hệ RMS với lưu lượng rò trên MDPE ở áp lực thông thường.
**Đóng góp khả dĩ:** hồi quy/phân loại thứ bậc lưu lượng rò ở 0,5–2 bar; nghiên cứu tương tác áp lực × kích thước lỗ trên đặc trưng năng lượng dải.

### G10. Khoảng trống trong nước
Chưa có bộ dữ liệu âm thanh rò rỉ mở cho ống nhựa ở Việt Nam, chưa có quy trình được kiểm chứng.
**Đóng góp khả dĩ:** công bố bộ dữ liệu (kèm siêu dữ liệu: áp lực, vật liệu, DN, kiểu rò, lưu lượng rò, cảm biến, vị trí) + mã nguồn tái lập;
kiểm chứng ngoài trên bộ dữ liệu PVC của Aghashahi và cs. (2023).

---

## 5. Gợi ý định vị đề tài và câu hỏi nghiên cứu

**Tên hướng định vị:** *"Nhận dạng rò rỉ bằng âm học trên ống PVC/HDPE áp lực thấp (0,5–2 bar): đặc tả chữ ký, lựa chọn đặc trưng có căn cứ vật lý và đánh giá tổng quát hóa chéo áp lực – chéo vật liệu."*

- **RQ1** (G1, G4): Băng thông hiệu dụng và SNR của tiếng rò thay đổi thế nào theo áp lực 0,5–2 bar trên PVC và HDPE; dải nào mang thông tin phân biệt?
- **RQ2** (G4, G7): Với dữ liệu nhỏ, biểu diễn nào (đặc trưng thủ công dải thấp, log-Mel, CWT, embedding tiền huấn luyện) và mô hình nào (RF/SVM, 1D/2D-CNN, CRNN) cho hiệu năng tốt nhất **ở mức bản ghi**?
- **RQ3** (G2, G3): Mô hình tổng quát hóa ra sao khi đổi áp lực và đổi vật liệu; cần bao nhiêu dữ liệu miền đích để fine-tune?
- **RQ4** (G5, G6): Hiệu năng giảm bao nhiêu khi (a) chuyển từ chia khung sang chia bản ghi/điều kiện, (b) thêm nhiễu thực tế Việt Nam vào lớp `noleak`, (c) thay rò bằng van bằng rò kiểu nứt/mối nối?

**Thiết kế thí nghiệm tối thiểu đề xuất:** 2 vật liệu × 4 mức áp lực × ≥ 3 kiểu rò × ≥ 2 kích thước × ≥ 5 lần lặp độc lập, cộng các điều kiện `noleak` có nhiễu;
ghi đồng thời áp lực, lưu lượng rò, nhiệt độ; ít nhất 2 loại cảm biến. Lưu **bản ghi thô** (không cắt sẵn) để có thể chia theo nhóm.

**Liên hệ mã nguồn hiện có (`notebooks/leak-cnn-pvc-kaggle-v7.ipynb`):** các thành phần CV lặp, gộp mức bản ghi, baseline log-Mel,
kiểm định Holm theo dải, ablation (bỏ FIR, 16 kHz, khung 1/4 s…) và `site_map_template.csv` cho leave-one-site-out đã đáp ứng một phần G4 và G6.
Cần bổ sung: cột **áp lực** và **vật liệu** trong siêu dữ liệu để chạy leave-one-pressure-out / leave-one-material-out (G2, G3), và bộ dữ liệu HDPE.

---

## 6. Nguồn đã tra cứu

- Hunaidi & Chu 1999 — https://www.sciencedirect.com/science/article/abs/pii/S0003682X99000134 ; bản chấp nhận NRC: https://nrc-publications.canada.ca/eng/view/accepted/?id=80c56d3d-3500-455a-83a7-9eb9a94963a9
- Scussel và cs. 2023 — https://www.mdpi.com/2624-599X/5/2/29
- Shekofteh và cs. 2026 — https://pmc.ncbi.nlm.nih.gov/articles/PMC13419278/
- Hamamed và cs. 2023 — https://pmc.ncbi.nlm.nih.gov/articles/PMC10537180/
- Butterfield và cs. 2017 — https://www.sciencedirect.com/science/article/pii/S0003682X17300099
- Butterfield và cs. 2018 — https://journals.sagepub.com/doi/full/10.1177/1475921717723881
- Shukla & Piratla 2020 — https://www.sciencedirect.com/science/article/abs/pii/S0926580519310301
- Quy & Kim 2020 — https://www.sciencedirect.com/science/article/abs/pii/S0263224119312709
- Aghashahi, Sela & Banks 2023 — https://www.sciencedirect.com/science/article/pii/S2352340923002676 ; dữ liệu: https://data.mendeley.com/datasets/tbrnp6vrnj/1
- Kang và cs. 2018 — https://ieeexplore.ieee.org/document/8074786/
- Cody, Tolson & Orchard 2020 — https://ascelibrary.org/doi/abs/10.1061/(ASCE)CP.1943-5487.0000881
- Ahmad và cs. 2022 — https://www.mdpi.com/1424-8220/22/4/1562
- Wu và cs. 2024 — https://pubmed.ncbi.nlm.nih.gov/38941677/
- Xu và cs. 2025 — https://pubmed.ncbi.nlm.nih.gov/39756226/
- Wang và cs. 2025 — https://pubmed.ncbi.nlm.nih.gov/40716211/
- Zhang và cs. 2023 — https://journals.sagepub.com/doi/abs/10.1177/14759217221080198
- Fan, Tariq & Zayed 2022 — https://www.sciencedirect.com/science/article/abs/pii/S0926580522000991
- Hu, Tariq & Zayed 2021 — https://www.sciencedirect.com/science/article/abs/pii/S0888327021003897
- Liu, Zayed & Xiao 2025 — https://www.sciencedirect.com/science/article/abs/pii/S0952197625014344
- Log-spectrogram CNN (JWRPM 2024) — https://ascelibrary.org/doi/10.1061/JWRMD5.WRENG-6276
- Ensemble ML (JWRPM 2025) — https://ascelibrary.org/doi/abs/10.1061/JWRMD5.WRENG-6578
- Bispectrum (JPSEP 2025) — https://ascelibrary.org/doi/abs/10.1061/JPSEA2.PSENG-1825
- Transformer-VAE (NDT&E 2026) — https://www.sciencedirect.com/science/article/pii/S0963869526000551
- Định vị rò trên mạng Hanoi (CNN + Bayes) — https://upcommons.upc.edu/bitstreams/977c7a58-b135-4fd2-af1b-07b2afcb7f5d/download
- Đề tài sinh viên ĐH Đà Nẵng (4G, lưu lượng – áp lực) — https://vnexpress.net/sinh-vien-thiet-ke-he-thong-chan-doan-ro-ri-ong-nuoc-4798274.html
- TCVN 13606:2023 — https://tieuchuan.vsqi.gov.vn/tieuchuan/view?sohieu=TCVN+13606:2023
