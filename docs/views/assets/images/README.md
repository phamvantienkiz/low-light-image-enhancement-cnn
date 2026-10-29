# 🖼️ Assets Directory - Images & Media
## CS2203 Low-Light Image Enhancement (Compact U-Net)

Thư mục này được sử dụng để lưu trữ các tài nguyên đồ họa, hình ảnh thực nghiệm và sơ đồ kiến trúc phục vụ hiển thị trên trang web `docs/views/`:

### Cấu trúc khuyến nghị:
```
assets/images/
├── datasets/             # Ảnh mẫu từ các tập LOL-v1, LOL-v2 Real, ExDark
│   ├── lol_v1_dark.png
│   ├── lol_v1_gt.png
│   └── exdark_sample.png
├── models/               # Sơ đồ kiến trúc mô hình xuất từ công cụ vẽ
│   ├── unet_architecture.svg
│   └── skip_gating_block.png
├── results/              # Ảnh so sánh kết quả thực nghiệm sau khi huấn luyện
│   ├── comparison_he.png
│   ├── comparison_clahe.png
│   ├── comparison_retinex.png
│   └── comparison_compact_unet.png
└── papers/               # Hình ảnh trích xuất từ 4 bài báo tham khảo CVF
    ├── darkir_metaformer.png
    └── hvi_color_space.png
```
