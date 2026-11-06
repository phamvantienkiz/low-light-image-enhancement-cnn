# Phân Tích & Đánh Giá Các Thư Viện Python Phục Vụ Viết LaTeX và Export Trực Tiếp (PDF & Word)

> **Môn học:** Chuyên đề Cao học / Đồ án M.Sc Computer Science (CS2203 - UIT)  
> **Đề tài:** Low-Light Image Enhancement Using a Lightweight Convolutional Neural Network  
> **Vị trí tài liệu:** `presentation/plan/giai_phap_thu_vien_latex_va_export.md`  
> **Mục đích:** Tài liệu kỹ thuật chi tiết phân tích hiện trạng môi trường, các thư viện Python, cơ chế biên dịch LaTeX cục bộ, tính toán dung lượng chiếm dụng trong `.venv` (quản lý bằng `uv`), và các phương án xuất trực tiếp ra định dạng PDF / Word (.docx) mà không cần sử dụng dịch vụ web (Overleaf).

---

## 1. Bản Chất Kỹ Thuật: Vì Sao Cần Phân Biệt Giữa "Viết LaTeX" và "Biên Dịch Ra PDF/Word"?

Khi làm việc với LaTeX trong hệ sinh thái Python, cần phân định rõ 3 tầng công nghệ:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ TẦNG 1: TẠO NỘI DUNG (Content Generation)                                       │
│ Sinh mã nguồn text LaTeX (.tex) từ dữ liệu thực nghiệm, bảng số liệu, công thức   │
│ Công cụ: Python thuần, Jinja2, PyLaTeX, Pandas (df.to_latex())                  │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
┌──────────────────────────────────────┐   ┌──────────────────────────────────────┐
│ TẦNG 2: BIÊN DỊCH RA PDF             │   │ TẦNG 3: CHUYỂN ĐỔI RA WORD (.DOCX)   │
│ Yêu cầu một TeX Engine độc lập       │   │ Yêu cầu công cụ Document Converter   │
│ để tính toán layout, ngắt dòng,      │   │ để dịch công thức LaTeX sang Word    │
│ căn lề, nạp font và xuất PDF vector. │   │ Equation chuẩn (OMML) native.        │
│ Công cụ: Tectonic Engine             │   │ Công cụ: Pandoc / pypandoc-binary    │
└──────────────────────────────────────┘   └──────────────────────────────────────┘
```

1. **LaTeX không phải định dạng hiển thị tĩnh (như HTML hay Markdown):** LaTeX là một ngôn ngữ lập trình macro biên dịch. Bản thân Python không thể tự "vẽ" ra một file PDF chuẩn IEEE 2 cột phức tạp nếu không có một **TeX Engine** đứng sau làm nhiệm vụ typeset (tính toán khoảng cách vi mô giữa các chữ, ngắt dòng, cân bằng 2 cột, giải quyết trích dẫn BibTeX).
2. **Hạn chế của giải pháp truyền thống:** Cài đặt các bộ TeX phân phối cổ điển như **TeX Live (~8 GB)** hoặc **MiKTeX (~4 GB)** là quá nặng, cấu hình biến môi trường phức tạp và vi phạm nguyên tắc cách ly môi trường (`AGENTS.md`).
3. **Mục tiêu đặt ra:** Tìm kiếm giải pháp tối ưu nhất, có thể tích hợp gọn gàng trong môi trường ảo `.venv`, được quản lý bởi `uv`, dung lượng tối thiểu (dưới 200 MB), và hoạt động hoàn toàn offline trên Windows.

---

## 2. Rà Soát Hiện Trạng Môi Trường Ảo `.venv` Hiện Tại

Kiểm tra danh mục các package hiện có trong `.venv` của dự án (thông qua `uv pip list`):
- `pandas` (3.1.0) & `numpy` (2.5.2): **Đã có sẵn**. Hỗ trợ đọc các file CSV thực nghiệm (`table_main.csv`, `paired_tests.csv`, `efficiency.csv`) và xuất thẳng sang cú pháp bảng LaTeX thông qua hàm `df.to_latex()`.
- `jinja2` (3.1.6): **Đã có sẵn**. Hỗ trợ tạo template bài báo và điền dữ liệu động vào file `.tex`.
- `matplotlib` (3.11.2): **Đã có sẵn**. Hỗ trợ vẽ biểu đồ Forest plot, biểu đồ sai số, xuất ra file vector PDF chèn trực tiếp vào bài báo với chất lượng in ấn cao nhất.

> [!NOTE]
> Toàn bộ các công cụ phục vụ **Tầng 1 (Tạo nội dung, trích xuất bảng biểu, vẽ đồ thị)** hiện đã có đầy đủ trong `.venv`.  
> Chi phí dung lượng tốn thêm cho Tầng 1: **`0 MB`**.

---

## 3. Đánh Giá Chi Tiết Từng Thư Viện / Công Cụ Đề Xuất

### 3.1. `tectonic` — Giải Pháp Xuất PDF Cục Bộ Hiện Đại Nhất (Khuyên Dùng)

- **Bản chất kỹ thuật:** `tectonic` là một trình biên dịch TeX/LaTeX độc lập được viết lại hoàn toàn bằng ngôn ngữ Rust, đóng gói kèm theo engine XeTeX, fontconfig và thư viện HarfBuzz.
- **Cơ chế hoạt động:**
  - Thay vì phải tải hàng nghìn package không dùng đến như TeX Live, `tectonic` sử dụng kiến trúc **VFS (Virtual File System) và tải package theo nhu cầu (on-demand streaming)** từ CTAN.
  - Khi biên dịch file `.tex` sử dụng các gói chuẩn (như `IEEEtran.cls`, `amsmath`, `graphicx`, `booktabs`), `tectonic` tự động tải đúng các file cần thiết trong lần chạy đầu tiên và lưu vào cache cục bộ của máy tính.
  - **Tự động lặp nhiều lượt (Automatic multiple passes):** `tectonic` tự động chạy 2–3 lượt biên dịch để giải quyết triệt để các liên kết chéo (`\ref`, `\label`) và danh mục tài liệu tham khảo BibTeX mà không cần người dùng phải gõ thủ công `pdflatex -> bibtex -> pdflatex`.
- **Dung lượng tính toán:**
  - File bánh xe tải về (`.whl`): **~18 MB**.
  - Dung lượng giải nén cài đặt trong `.venv`: **~45 – 50 MB** (chủ yếu là 1 file binary `tectonic.exe` và Python C-extension binding).
  - Cache macro/font lần đầu tiên (tải về `%LOCALAPPDATA%\Tectonic\cache`): **~20 – 30 MB**.
- **Cách sử dụng trực tiếp trong Python:**
  ```python
  import tectonic

  # Biên dịch trực tiếp file .tex sang PDF chỉ bằng 1 dòng lệnh
  pdf_data = tectonic.sync_compile("presentation/paper/main.tex")

  # Lưu kết quả ra file PDF
  with open("presentation/paper/main.pdf", "wb") as f:
    f.write(pdf_data)
```

---

### 3.2. `pypandoc-binary` — Giải Pháp Xuất Sang Microsoft Word (`.docx`) (Khuyên Dùng)

- **Bản chất kỹ thuật:** `pandoc` là bộ chuyển đổi văn bản phổ thông (Universal Document Converter) mạnh nhất hiện nay. Package `pypandoc-binary` trên PyPI là gói Python đóng gói sẵn binary thực thi của Pandoc cho Windows x64.
- **Vì sao chọn `pypandoc-binary` thay vì `pypandoc` thường?**
  - Gói `pypandoc` thông thường chỉ là wrapper, đòi hỏi người dùng phải tự cài phần mềm Pandoc lên toàn hệ điều hành Windows.
  - Gói `pypandoc-binary` tích hợp sẵn file thực thi `pandoc.exe` ngay bên trong thư mục `site-packages` của `.venv`, bảo đảm tính **cách ly môi trường 100%**, không tác động tới hệ thống máy chủ/OS host.
- **Ưu điểm khi chuyển đổi LaTeX sang Word:**
  - Chuyển đổi toàn bộ các khối toán học LaTeX (`\begin{equation}`, inline math `$...$`) thành **công thức chuẩn của Word (OMML - Office Math Markup Language)**. Khi mở file `.docx` trong Microsoft Word, người đọc hoặc giảng viên có thể click đúp chuột vào công thức để chỉnh sửa bình thường (không bị biến thành ảnh bitmap vỡ hạt).
  - Tự động chuyển đổi bảng LaTeX (`table`, `tabular`) thành bảng biểu native của Word.
  - Hỗ trợ xử lý trích dẫn BibTeX thông qua flag `--citeproc`.
- **Dung lượng tính toán:**
  - File bánh xe tải về (`.whl`): **~35 MB**.
  - Dung lượng giải nén cài đặt trong `.venv`: **~105 – 115 MB** (gồm binary `pandoc.exe` ~100 MB).
- **Cách sử dụng trực tiếp trong Python:**
  ```python
  import pypandoc

  # Chuyển đổi file main.tex sang main.docx kèm trích dẫn BibTeX
  pypandoc.convert_file(
      "presentation/paper/main.tex",
      "docx",
      outputfile="presentation/paper/main.docx",
      extra_args=[
          "--citeproc",
          "--bibliography=presentation/paper/references.bib",
      ],
  )
```

---

### 3.3. `pylatex` — Thư Viện Sinh Mã LaTeX Bằng Lập Trình Hướng Đối Tượng (Tùy Chọn)

- **Bản chất kỹ thuật:** `pylatex` cung cấp một lớp trừu tượng (abstraction layer) bằng Python để người lập trình có thể định nghĩa văn bản LaTeX mà không cần gõ cú pháp dấu gạch chéo `\`:
  ```python
  from pylatex import Document, Section, Subsection, Table

  doc = Document()
  with doc.create(Section("Introduction")):
    doc.append("Noi dung phan mo dau...")
```
- **Dung lượng tính toán:**
  - File bánh xe tải về: **~200 KB**.
  - Dung lượng giải nén trong `.venv`: **~1.5 MB** (kèm phụ thuộc `ordered-set`).
- **Đánh giá ưu nhược điểm & Nhận định:**
  - *Nhược điểm chí mạng:* Bản thân `pylatex` **không thể tự sinh ra file PDF**. Khi gọi `doc.generate_pdf()`, nó vẫn phải tìm một TeX compiler trên máy tính (như `pdflatex` hoặc `tectonic`).
  - *Tính thực tiễn:* Với một bài báo khoa học chuẩn IEEE 2 cột phức tạp (cần căn chỉnh tỉ mỉ từng ngắt dòng, cân bằng cột `\balance`, cấu hình gói hình ảnh, trích dẫn nhiều tác giả), việc viết bằng code Python OOP rất cồng kềnh, khó kiểm soát định dạng hơn so với việc viết trực tiếp vào file `.tex` hoặc dùng template Jinja2.
  - *Khuyến nghị:* **Không cần thiết cài đặt**. Sử dụng template `.tex` chuẩn kết hợp `jinja2` (đã có sẵn) là phương án sạch và chuẩn mực hơn.

---

### 3.4. `jinja2` + Template LaTeX — Giải Pháp Tự Động Hóa Chuẩn Nghiên Cứu (Khuyên Dùng)

- **Cơ chế hoạt động:**
  - Tạo một file khung mẫu `paper_template.tex` chuẩn IEEEtran.
  - Dùng cú pháp Jinja2 (`{{ psnr_unet }}`, `{{ ssim_darkir }}`) để làm vị trí chờ dữ liệu.
  - Script Python đọc file CSV kết quả, nạp số liệu vào template và xuất ra file `main.tex`.
- **Dung lượng tốn thêm:** **`0 MB`** (vì `jinja2` đã có sẵn trong `.venv`).

---

## 4. Bảng Tính Toán & So Sánh Các Kịch Bản Lựa Chọn

Dưới đây là bảng tổng hợp các kịch bản để cân nhắc trước khi ra quyết định:

| Tiêu chí | Kịch bản 1: Chỉ xuất PDF (Khuyên dùng) | Kịch bản 2: Xuất cả PDF và Word (.docx) | Kịch bản 3: Không cài vào `.venv` (Dùng Portable Tool) |
|:---|:---:|:---:|:---:|
| **Mục tiêu đáp ứng** | Xuất bài báo chuẩn IEEE 2 cột dạng PDF | Vừa có PDF chuẩn nộp hội nghị/hội đồng, vừa có file Word nộp trường chấm điểm | Vừa có PDF vừa có Word nhưng giữ nguyên vẹn 100% dung lượng `.venv` |
| **Các package cần cài trong `.venv`** | `tectonic` | `tectonic` và `pypandoc-binary` | *Không cài bất kỳ package nào* |
| **Dung lượng download (Wheel)** | ~18 MB | ~53 MB (18 MB + 35 MB) | 0 MB qua `uv` (tải file zip ngoài) |
| **Dung lượng giải nén trong `.venv`** | **~45 – 50 MB** | **~155 – 165 MB** | **0 MB** |
| **Cache phụ trợ (lưu ngoài `.venv`)** | ~20 – 30 MB (cache CTAN macro) | ~20 – 30 MB (cache CTAN macro) | ~20 – 30 MB (cache CTAN macro) |
| **Mức độ phức tạp khi thao tác** | Rất đơn giản (1 dòng lệnh Python hoặc CLI) | Đơn giản (1 script Python xuất cả 2 định dạng) | Cần tải thủ công 2 file `.exe` đặt vào thư mục công cụ ngoài repo |
| **Mức độ phụ thuộc hệ thống** | 0% (gói gọn hoàn toàn trong `.venv`) | 0% (gói gọn hoàn toàn trong `.venv`) | Cần quản lý đường dẫn file binary trong code |

---

## 5. Quy Trình Vận Hành Hoàn Chỉnh Khi Triển Khai (Pipeline Architecture)

Khi bạn đưa ra quyết định lựa chọn kịch bản, toàn bộ quy trình từ số liệu thực nghiệm đến bài báo hoàn chỉnh sẽ vận hành tự động theo sơ đồ sau:

```
[Kết quả thực nghiệm]
 ├─ table_main.csv (v3 & v4)
 ├─ paired_tests.csv (v3 & v4)
 └─ efficiency.csv (v3 & v4)
            │
            ▼ (Chạy script Python nội bộ)
[docs/ai/export_paper_tables.py]
 ├─ Tự động trích xuất bảng LaTeX vào: presentation/paper/tables/
 └─ Tự động vẽ biểu đồ vector PDF vào: presentation/paper/figures/forest_plot.pdf
            │
            ▼
[presentation/paper/main.tex] (Khung bài báo khoa học chuẩn IEEEtran 2 cột)
            │
            ├─────────────────────────────────────────┐
            ▼ (Qua Tectonic)                          ▼ (Qua Pandoc)
[presentation/paper/main.pdf]             [presentation/paper/main.docx]
(Bản in chuẩn bài báo IEEE 2 cột)         (Bản Word công thức toán OMML native)
```

---

## 6. Hướng Dẫn Các Lệnh Tương Ứng (Tham Khảo Khi Ra Quyết Định)

> [!IMPORTANT]
> Đây là các lệnh mẫu để tham khảo. **Chưa có lệnh nào được chạy thực tế**. Mọi thao tác cài đặt sẽ chỉ thực hiện khi có xác nhận chính thức từ bạn.

### Nếu chọn Kịch bản 1 (Chỉ PDF — tốn ~50 MB trong `.venv`):
```powershell
uv pip install tectonic
```
Sau đó xuất PDF bằng script Python:
```python
import tectonic

pdf = tectonic.sync_compile("presentation/paper/main.tex")
open("presentation/paper/main.pdf", "wb").write(pdf)
```

### Nếu chọn Kịch bản 2 (Cả PDF và Word — tốn ~160 MB trong `.venv`):
```powershell
uv pip install tectonic pypandoc-binary
```
Sau đó xuất cả PDF và Word bằng script Python:
```python
import pypandoc
import tectonic

# 1. Xuất PDF
pdf = tectonic.sync_compile("presentation/paper/main.tex")
open("presentation/paper/main.pdf", "wb").write(pdf)

# 2. Xuất Word (.docx)
pypandoc.convert_file(
    "presentation/paper/main.tex",
    "docx",
    outputfile="presentation/paper/main.docx",
    extra_args=[
        "--citeproc",
        "--bibliography=presentation/paper/references.bib",
    ],
)
```

### Nếu chọn Kịch bản 3 (Giữ `.venv` 0 MB):
Không chạy bất kỳ lệnh `uv pip install` nào. Chỉ tải các file `tectonic.exe` và `pandoc.exe` lưu vào một thư mục công cụ độc lập và dùng `subprocess.run()` để thực thi.
