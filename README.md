# XMD-XSS Demo Website

Website demo cơ bản cho KLTN **Explainable Multimodal Deep Learning for XSS Attack Detection**.

## 1. Chức năng chính

Website cho phép giảng viên nhập một payload và xem:

- Kết luận: `XSS` hoặc `Benign`
- Xác suất mô hình dự đoán payload là XSS
- Mức cảnh báo: `An toàn`, `Cần kiểm tra thêm`, `Cảnh báo`, `Nguy hiểm`
- Thời gian suy luận
- Token sau khi tiền xử lý
- Structural features của payload
- Giải thích nhanh dựa trên:
  - dấu hiệu bảo mật trong payload
  - thống kê XAI toàn cục từ Integrated Gradients / SHAP đã xuất từ notebook
- Kết quả thực nghiệm:
  - Accuracy, Precision, Recall, F1-score
  - so sánh mô hình single-modal / multimodal
  - confusion matrix
  - XAI summary
  - thông tin dataset

## 2. Cấu trúc thư mục

```text
xss_demo_website/
├── app.py
├── model_utils.py
├── requirements.txt
├── Dockerfile
├── .streamlit/config.toml
└── artifacts/
    ├── deepxss/
    │   ├── Text + Structural + Char_best.pt
    │   ├── xss_preprocessing_objects.pkl
    │   └── các file kết quả .csv
    └── fmereani/
        ├── Text + Structural + Char_best.pt
        ├── xss_preprocessing_objects.pkl
        └── các file kết quả .csv
```

## 3. Chạy local trên máy cá nhân

### Bước 1: Giải nén project

Giải nén file `xss_demo_website.zip`.

### Bước 2: Tạo môi trường ảo

```bash
cd xss_demo_website
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### Bước 3: Cài thư viện

```bash
pip install -r requirements.txt
```

### Bước 4: Chạy website

```bash
streamlit run app.py
```

Sau đó mở địa chỉ Streamlit hiển thị trên terminal, thường là:

```text
http://localhost:8501
```

## 4. Deploy bằng Streamlit Community Cloud

Cách này dễ nhất để giảng viên mở link test.

### Bước 1: Đưa project lên GitHub

Tạo repository mới, ví dụ:

```text
xmd-xss-demo
```

Upload toàn bộ thư mục này lên GitHub, gồm:

```text
app.py
model_utils.py
requirements.txt
artifacts/
.streamlit/
```

Lưu ý: thư mục `artifacts/` có chứa model `.pt`, cần commit lên GitHub. Nếu GitHub báo file quá lớn thì dùng Git LFS.

### Bước 2: Tạo app trên Streamlit Cloud

1. Vào Streamlit Community Cloud.
2. Chọn `New app`.
3. Chọn repository `xmd-xss-demo`.
4. Main file path: `app.py`.
5. Deploy.

Sau khi deploy xong, gửi link cho giảng viên.

## 5. Deploy bằng Render

Render phù hợp khi muốn deploy bằng Docker.

### Bước 1: Đưa project lên GitHub

Tương tự phần Streamlit Cloud.

### Bước 2: Tạo Web Service trên Render

1. Chọn `New +` → `Web Service`.
2. Kết nối GitHub repository.
3. Environment: `Docker`.
4. Render sẽ tự đọc `Dockerfile`.
5. Deploy.

Sau khi deploy xong, Render sẽ cấp một URL public.

## 6. Chạy bằng Docker local

```bash
cd xss_demo_website
docker build -t xmd-xss-demo .
docker run -p 8501:8501 xmd-xss-demo
```

Mở:

```text
http://localhost:8501
```

## 7. Gợi ý trình bày khi bảo vệ

Khi demo, nên đi theo kịch bản:

1. Mở website.
2. Chọn dataset/model.
3. Nhập payload benign, ví dụ:
   ```text
   Hello, this is a normal search keyword
   ```
4. Nhập payload XSS cơ bản:
   ```text
   <script>alert(1)</script>
   ```
5. Nhập payload có event handler:
   ```text
   <img src=x onerror=alert(document.cookie)>
   ```
6. Giải thích rằng mô hình không chỉ nhìn chuỗi văn bản mà dùng 3 modal:
   - Text token sequence
   - Structural handcrafted features
   - Character-level sequence
7. Mở tab kết quả nghiên cứu để cho thấy:
   - mô hình đa phương thức
   - chỉ số đánh giá
   - confusion matrix
   - XAI summary

## 8. Lưu ý học thuật

Website này là demo phục vụ KLTN. Mục tiêu là thể hiện pipeline nghiên cứu gồm tiền xử lý, mô hình đa phương thức, dự đoán nhị phân XSS/Benign và giải thích XAI. Không nên trình bày đây là hệ thống bảo mật sản xuất thay thế WAF/IDS.
