# Hướng thực hiện đồ án LLIE bằng Lightweight CNN

> Phạm vi: đồ án môn học nhỏ (CS2203), **không** nhắm tới paper công bố, **không** reproduce SOTA, **không** phát minh phương pháp mới.
> Ràng buộc: laptop không có GPU. Repo chỉ viết code; train/eval chạy trên Colab (hoặc máy có GPU), sau đó mang kết quả về repo.
> 
> 📚 **Tài liệu nghiên cứu chi tiết & bộ hướng dẫn thực nghiệm hoàn chỉnh đã được hợp nhất tại thư mục:**  
> 👉 [`docs/research/research_approach/README.md`](file:///E:/MyProject/low-light-image-enhancement-cnn/docs/research/research_approach/README.md)

---

## 1. Nguyên tắc chọn skill

Research-hub có 8 giai đoạn dành cho nghiên cứu đầy đủ. Đồ án này **chỉ cần phiên bản rút gọn**:

- Đề tài đã khóa (xem `topic_analysis.md`): Lightweight U-Net, L1 loss, LOL-v1/v2, PSNR/SSIM, baseline Gamma/CLAHE.
- Ba paper tham khảo chính đã chọn: DarkIR, HVI, NTIRE 2024 LLIE.
- Vì vậy bỏ qua các bước tìm novelty và ideation, cũng như phần phát hành poster, blog, video.

## 2. Skill NÊN dùng

| Giai đoạn | Skill | Dùng để làm gì (mức tối thiểu) |
| :-- | :-- | :-- |
| Đọc tài liệu | `paper-lookup` (hoặc `rs-paper-search`) | Lấy metadata/PDF của 3 paper chính + 2-3 paper phụ (vd. "Learning Optimized LLIE for Edge", LOL gốc). Không cần review hệ thống. |
| Đọc tài liệu | `citation-management` | Tạo BibTeX chuẩn cho mục Tài liệu tham khảo. |
| Chuẩn bị | `get-available-resources` | Kiểm tra nhanh tài nguyên để chọn batch size, patch size cho Colab (và xác nhận laptop chỉ dùng để code). |
| Dữ liệu | `exploratory-data-analysis` | Kiểm tra nhanh dataset LOL: số cặp, kích thước, độ sáng trung bình, đảm bảo không rò rỉ train/test. |
| Mô hình | `pytorch-lightning` *(tùy chọn)* | Chỉ dùng nếu muốn gọn code train/log. Nếu muốn dễ giải thích với giảng viên thì dùng PyTorch thuần vòng lặp train đơn giản. |
| Đánh giá | `statistical-analysis` *(nhẹ)* | Chỉ để báo cáo trung bình ± độ lệch chuẩn PSNR/SSIM trên tập test, và chọn seed. Không cần power analysis. |
| Hình ảnh | `scientific-visualization` | Bảng kết quả, đường cong loss, ảnh so sánh Input / Gamma / CLAHE / CNN / U-Net / GT. |
| Hình ảnh | `scientific-schematics` | Vẽ sơ đồ pipeline và kiến trúc U-Net cho báo cáo và slide. |
| Báo cáo | `scientific-writing` | Viết báo cáo dạng paper: Abstract, Intro, Related Work, Problem, Method, Experiments, Discussion, References. |
| Báo cáo | `venue-templates` *(tùy chọn)* | Chỉ để mượn template LaTeX (vd. CVPR/IEEE) nếu giảng viên yêu cầu định dạng paper. |
| Trình bày | `scientific-slides` | Làm slide theo đúng danh sách nội dung giảng viên yêu cầu. |
| Kiểm tra cuối | `peer-review` *(tùy chọn, nhẹ)* | Cho đọc thử bản nháp báo cáo để phát hiện chỗ lỏng, chỗ tuyên bố quá mức (đặc biệt không được claim SOTA). |

## 3. Skill KHÔNG nên dùng (vượt phạm vi)

| Nhóm | Skill | Lý do bỏ |
| :-- | :-- | :-- |
| Ideation / novelty | `rs-idea-spark`, `rs-bottleneck-diagnosis`, `rs-pattern-ideator`, `rs-scoop-check`, `rs-idea-quality` | Đề tài không cần tính mới; giảng viên nói rõ không cần phát minh phương pháp. |
| Review văn liệu lớn | `literature-review` | Quá nặng; chỉ cần Related Work ngắn dựa trên 3-5 paper. |
| Dữ liệu lớn | `polars`, `vaex`, `dask`, `zarr-python` | LOL chỉ vài trăm ảnh, không cần out-of-core. |
| Mô hình ngoài phạm vi | `transformers`, `torch-geometric`, `stable-baselines3`, `pymc`, `shap`, `umap-learn`, ... | Đề tài loại trừ Transformer/Diffusion/pretrained; không liên quan GNN/RL. |
| Hạ tầng | `modal`, `optimize-for-gpu` | Colab miễn phí là đủ cho U-Net nhỏ; tránh tốn công thiết lập. |
| Phát hành | `rs-paper2assets/poster/blog/video/reel`, `latex-posters` | Không phải yêu cầu của môn. |
| Thiết kế thực nghiệm | `experimental-design`, `statistical-power` | Không phải nghiên cứu có người tham gia; thiết kế đã cố định (so sánh 3-4 mô hình). |

## 4. Kế hoạch thực hiện (rút gọn, ~4 tuần)

| Tuần | Công việc | Skill | Sản phẩm |
| :-- | :-- | :-- | :-- |
| 1 | Tải và đọc 3 paper chính, chốt Related Work; kiểm tra dataset LOL-v1/v2; viết code dataset loader, metric PSNR/SSIM, baseline Gamma/CLAHE | `paper-lookup`, `citation-management`, `exploratory-data-analysis` | `references.bib`, bảng thống kê dataset, kết quả baseline truyền thống |
| 2 | Viết Plain CNN và Lightweight U-Net (L1 loss), notebook train cho Colab | `get-available-resources`, `pytorch-lightning` (tùy chọn) | Code train/eval, notebook Colab |
| 3 | Chạy trên Colab, mang kết quả về; thêm Residual U-Net hoặc L1+SSIM nếu còn thời gian; thử ExDark/FiveK như phần Limitations (tùy chọn) | `statistical-analysis` (nhẹ) | Bảng PSNR/SSIM/Params/Inference time, checkpoint |
| 4 | Vẽ hình, viết báo cáo, làm slide, tự review | `scientific-visualization`, `scientific-schematics`, `scientific-writing`, `scientific-slides`, `peer-review` | Báo cáo, slide, source code |

## 5. Thiết kế thực nghiệm tối thiểu

| Mô hình | Vai trò |
| :-- | :-- |
| Gamma, CLAHE | Baseline truyền thống |
| Plain CNN | Baseline học sâu đơn giản |
| Lightweight U-Net | Mô hình chính |
| U-Net + Residual *(tùy chọn)* | Mở rộng nếu còn thời gian |

- **Dữ liệu:** LOL-v1 + LOL-v2 Real (train/test theo split chuẩn có sẵn). Dữ liệu mở rộng chỉ để minh họa limitation.
- **Độ đo:** PSNR, SSIM (LPIPS bỏ qua trừ khi còn dư thời gian). Ghi thêm số tham số và thời gian inference.
- **Huấn luyện:** random crop (vd. 128-256 px), flip, Adam, L1 loss, một seed chính (tối đa 3 seed nếu Colab cho phép).
- **Kết quả:** một bảng so sánh, một hình định tính, một đường cong loss.
- **Tuyên bố:** chỉ nói "CNN cơ bản giải quyết bài toán đến mức nào và độ phức tạp kiến trúc ảnh hưởng ra sao", **không** so sánh ngang hàng hay claim hơn DarkIR/HVI.

## 6. Ánh xạ với yêu cầu giảng viên

| Yêu cầu | Đáp ứng bằng |
| :-- | :-- |
| Bài toán thực tế, phát biểu input/output/ví dụ | Mục Problem Statement + hình minh họa low-light → enhanced |
| Paper hiện đại (CVF) | DarkIR, HVI, NTIRE 2024 (Related Work) |
| Dataset, độ đo, môi trường | LOL, PSNR/SSIM, mô tả Colab/GPU |
| CNN mức cơ bản, hiểu rõ chi tiết | Plain CNN + Lightweight U-Net, giải thích từng thành phần (conv, downsample, upsample, skip, loss) |
| Không claim SOTA | Nêu rõ trong Intro và Discussion |
| Báo cáo, code, slide | Xem mục 4 (tuần 4) |

## 7. Nguyên tắc giữ đồ án nhỏ

1. Tối đa 3 mô hình học sâu; mở rộng chỉ làm khi baseline đã xong.
2. Ưu tiên hiểu và giải thích được mọi dòng code hơn là điểm số cao.
3. Mỗi skill chỉ dùng ở mức tối thiểu cần thiết; không chạy cả pipeline 8 giai đoạn.
4. Nếu trễ tiến độ: bỏ Residual U-Net, bỏ dataset mở rộng, bỏ `peer-review`; giữ lõi Gamma/CLAHE, Plain CNN, U-Net trên LOL.
