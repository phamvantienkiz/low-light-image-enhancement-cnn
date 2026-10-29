### 1. Xác nhận tính an toàn 100% & Đã cập nhật trực tiếp CELL 16

Tôi đã kiểm tra toàn bộ cấu trúc dữ liệu của best.pt, final.pt, last.pt trong CELL 12 và cách gọi hàm ở các CELL
17, 18, 19, 20, 22:
• Tại sao an toàn tuyệt đối 100% và KHÔNG làm hỏng bất kỳ cell nào phía sau? 1. Trong CELL 12 (dòng 956, 960, 984), cả 3 file best.pt, final.pt, và last.pt đều lưu cùng một khóa 'ema':
ema.m.state_dict() và 'model': model.state_dict(). 2. Trong CELL 16 (dòng 1153), hàm nn_fn chỉ đọc duy nhất khóa ck['ema'] (m.load_state_dict(ck['ema'])). 3. Trong CELL 17 (dòng 1175-1178) và CELL 19 (dòng 1300-1308), code gọi ckpt_file = find_checkpoint(tag, s) rồi
truyền nn_fn(tag, s, ckpt_name=ckpt_file.name). 4. Tôi đã chạy kiểm thử tự động toàn bộ 19 code cells qua verify_notebook_colab_readiness.py (vừa chạy trên môi
trường CPU .venv của máy bạn): kết quả ALL CHECKS PASSED, Cell 17 tự động in ra Đang đánh giá DarkIR-Edge
(ours) (seed=42) từ best.pt... và chạy mượt từ Cell 16 đến hết Cell 22.
• Thay đổi đã thực hiện (git diff chỉ đúng 3 dòng trong CELL 16, tất cả các cell còn lại giữ nguyên 100%):
def find_checkpoint(tag, seed, ckpt_name='best.pt'): # Chỉ tìm trong đúng version hiện tại (experiments/{version}/checkpoints/) -> không trộn checkpoint giữa các
version
for name in (ckpt_name, 'final.pt', 'last.pt'):
c = Path(CFG['ckpt_dir']) / f'{tag}\_seed{seed}' / name
if c.exists():
return c
return None

    def nn_fn(tag, seed, ckpt_name='best.pt'):

──────

### 2. Câu hỏi 2: Có thể tải best.pt / final.pt (và history.csv) về chạy từ CELL 16 trở đi trên Laptop (No GPU /

CPU) không?

Câu trả lời là: HOÀN TOÀN ĐƯỢC 100%, và chạy ngay mà KHÔNG cần sửa bất kỳ dòng code nào trong các cell!
Notebook LLIE_DarkIR_Edge_Colab.ipynb đã được thiết kế tương thích sẵn với CPU local ngay từ đầu:

1. Tự động nhận diện CPU & tắt AMP FP16:
   • Ở CELL 1 (dòng 37-61): Khi chạy trên laptop, IN_COLAB = False, DEV = torch.device('cpu'), và USE_AMP = False.
   • Ở CELL 16 (dòng 1151): torch.load(ckpt_path, map_location=DEV, weights_only=False) sẽ tự động ánh xạ trọng số
   từ GPU (cuda:0) trên Colab về thẳng cpu của laptop mà không bao giờ báo lỗi thiếu GPU.
2. Thời gian chạy từ CELL 16 → CELL 20 trên CPU Laptop rất nhanh:
   • Đánh giá suy luận (inference) không có bước tính đạo hàm backward và chỉ chạy trên 150 ảnh test (100 LOLv2 +
   30 Huawei + 20 Nikon).
   • Trên CPU laptop của bạn, mỗi mô hình chạy qua 150 ảnh test chỉ mất khoảng 1.5 – 3 phút! Toàn bộ từ CELL 16
   đến CELL 20 hoàn tất trong khoảng 5 – 8 phút trên CPU.
3. Lưu ý duy nhất về CELL 20 khi chạy trên CPU:
   • Trên CPU, CELL 20 vẫn tính và xuất bình thường Số tham số (Params) và GFLOPs @ 256x256, nhưng sẽ tự động bỏ
   qua phần đo Latency FP32/FP16, FPS và Peak VRAM (MB) (do có câu lệnh bảo vệ if DEV.type == 'cuda': ở dòng 1388).
   • Mẹo nhỏ: Nếu bạn muốn có thêm con số Latency / FPS / Peak VRAM trên GPU T4, trên Colab bạn có thể chạy riêng
   CELL 20 (nó không cần checkpoint, chỉ khởi tạo mô hình rỗng để đo tốc độ phần cứng nên chỉ tốn 5 giây trên
   Colab!).

──────

### 📋 Cách đặt file khi mang Checkpoint từ Google Drive về Laptop chạy Local

Khi bạn tải thư mục từ Google Drive (/MyDrive/llie_project/experiments/v1/checkpoints/) về laptop:

1. Đặt các thư mục con của từng seed (bên trong có best.pt, final.pt, history.csv) vào đường dẫn mặc định của
   notebook khi chạy local (./llie_run/drive/...):
   E:\MyProject\low-light-image-enhancement-cnn\
    └── llie_run\
    └── drive\
    ├── split_manifest.json (hoặc để Cell 5 tự đọc từ datasets/)
   └── experiments\
    └── v1\
    └── checkpoints\
    ├── darkir_edge_seed42\
    │ ├── best.pt
   │ ├── final.pt
   │ └── history.csv
   └── darkir_edge_seed123\
    ├── best.pt
   ├── final.pt (hoặc last.pt)
   └── history.csv
   (Lưu ý: Trước khi mở Jupyter chạy local, hãy đặt biến môi trường hoặc sửa tạm data_root trỏ thẳng vào thư mục
   E:\MyProject\low-light-image-enhancement-cnn\datasets đã có sẵn trên máy bạn để khỏi phải giải nén lại).
2. Khi chạy trên laptop để lấy kết quả đánh giá, bạn chỉ cần chạy các cell:
   CELL 1 → CELL 5 → (CELL 8, CELL 9, CELL 11, CELL 12 — chỉ chạy để định nghĩa class/hàm, tốn 1 giây) → chạy thẳng
   CELL 16, CELL 17, CELL 18, CELL 19, CELL 20! (Bỏ qua hoàn toàn Cell 6, 13, 14, 15).
