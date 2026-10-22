# 🏡 BẤT ĐỘNG SẢN AI - HỆ THỐNG ĐỊNH GIÁ TỰ ĐỘNG (AVM) & PHÂN TÍCH TIỀM NĂNG ĐẦU TƯ

> **Đồ án tốt nghiệp / Dự án Capstone chuyên ngành Công nghệ thông tin - Trí tuệ nhân tạo**  
> *Ứng dụng kết hợp Machine Learning Pipeline, FastAPI High-Performance Backend và Modern UI/UX React + Tailwind CSS.*

---

## 📌 1. Giới thiệu tổng quan (Project Overview)

Trong thị trường bất động sản thực tế, việc định giá căn nhà thường gặp phải vấn đề thiếu minh bạch và mang tính cảm tính. Dự án **Hệ thống Định giá Bất động sản Tự động (Automated Valuation Model - AVM)** được xây dựng nhằm giải quyết bài toán:
* **Định giá thị trường khách quan**: Sử dụng mô hình học máy **Gradient Boosting Regressor** trên không gian phân phối chuẩn (log-transformed target), đạt độ chính xác cao ($R^2 > 0.98$).
* **Khoảng tin cậy an toàn 95% (Confidence Interval)**: Thay vì chỉ đưa ra một con số duy nhất, hệ thống tính toán biên độ sai số dựa trên phân phối phần dư (residuals distribution) để bảo vệ quyết định của nhà đầu tư.
* **Định danh cơ hội thị trường**: So sánh giá chủ nhà rao bán với giá trị thực để gắn cờ *Giá hời (Undervalued)*, *Đúng giá (Fair Price)*, hoặc *Định giá quá cao (Overpriced)*.
* **Đề xuất bất động sản tương đồng (Comparable Properties)**: Ứng dụng thuật toán **K-Nearest Neighbors (K-NN Cosine Similarity)** trên không gian đặc trưng đa chiều để tìm ra 5 bất động sản lân cận có mức độ tương đồng cao nhất.
* **Chỉ số tiềm năng đầu tư**: Tích hợp mô hình chấm điểm tiềm năng đa tiêu chí (vị trí lõi, cự ly trung tâm, tốc độ đô thị hóa, khấu hao tuổi thọ công trình).

---

## 🏗️ 2. Kiến trúc hệ thống (System Architecture)

```text
[ Người Dùng / Frontend ] 
         │ (HTTP / JSON / Vite Proxy)
         ▼
[ FastAPI Backend (Port 8000) ]
   ├── Pydantic v2 (Validation & Schemas)
   ├── Asynchronous Lifespan (In-memory Artifact Cache)
   └── RealEstatePredictor Engine
         │
         ├── ColumnTransformer (RobustScaler + OneHotEncoder)
         ├── GradientBoostingRegressor (y = log(1 + price))
         ├── Confidence Interval Calculator (± 1.96 * Residual Std)
         └── K-NN Cosine Recommender (5 Nearest Neighbors)
```

### Công nghệ sử dụng (Tech Stack):
* **Machine Learning / Data Science**: Python 3.10+, Scikit-learn, NumPy, Pandas, Joblib.
* **Backend Services**: FastAPI, Pydantic v2, Uvicorn, Python-Multipart.
* **Frontend UI/UX**: React 19, TypeScript, Vite, Tailwind CSS v4, Lucide React Icons.

---

## 📂 3. Cấu trúc thư mục (Directory Structure)

```text
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # Điểm khởi chạy FastAPI, CORS, Lifespan & Routes
│   │   ├── core/
│   │   │   └── predictor.py         # Lõi suy luận ML, tính 95% CI & K-NN tương đồng
│   │   └── schemas/
│   │       ├── __init__.py
│   │       └── property.py          # Pydantic schemas kiểm thực dữ liệu vào/ra
│   ├── models/                      # Thư mục chứa các artifacts sau huấn luyện
│   │   ├── housing_model.pkl        # Model GradientBoostingRegressor
│   │   ├── preprocessor.pkl         # Pipeline ColumnTransformer
│   │   ├── property_index.pkl       # K-NN model & tập dữ liệu bất động sản
│   │   └── metadata.json            # Siêu dữ liệu metrics (R2, MAE, RMSE, residual_std)
│   └── requirements.txt             # Thư viện Python phía server
│
├── training/
│   ├── train_model.py               # Script sinh dữ liệu thực nghiệm, train & xuất artifacts
│   └── real_estate_data.csv         # Tập dữ liệu huấn luyện (4,500 mẫu)
│
├── src/                             # Mã nguồn Frontend (React + Vite)
│   ├── App.tsx                      # Component Dashboard chính (2 cột tương tác)
│   ├── main.tsx                     # Entrypoint React
│   └── index.css                    # Tailwind CSS imports
│
├── package.json                     # Quản lý dependencies Frontend
├── vite.config.ts                   # Cấu hình proxy kết nối FastAPI
└── README.md
```

---

## ⚙️ 4. Hướng dẫn cài đặt và khởi chạy (Quickstart)

### 4.1. Chuẩn bị môi trường
* Đảm bảo máy tính đã cài đặt **Python 3.10+** và **Node.js 18+ (hoặc 20 LTS)**.

---

### 4.2. Huấn luyện mô hình Machine Learning
Trước khi khởi động server lần đầu, cần sinh tập dữ liệu và xuất các file mô hình `.pkl`:

```bash
# 1. Tạo và kích hoạt môi trường ảo Python
python -m venv venv

# Trên Windows:
.\venv\Scripts\activate
# Trên macOS / Linux:
source venv/bin/activate

# 2. Cài đặt thư viện Python
pip install -r backend/requirements.txt

# 3. Chạy script huấn luyện mô hình
python training/train_model.py
```
> *Sau khi chạy xong, các file `housing_model.pkl`, `preprocessor.pkl`, `property_index.pkl` và `metadata.json` sẽ được tạo sẵn trong thư mục `backend/models/`.*

---

### 4.3. Khởi động Backend FastAPI
Mở Terminal thứ nhất (đang bật `venv`) và chạy:

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```
* API Server hoạt động tại: `http://127.0.0.1:8000`
* Tài liệu Swagger UI tương tác trực tiếp: `http://127.0.0.1:8000/docs`

---

### 4.4. Khởi động Frontend React
Mở Terminal thứ hai tại thư mục gốc của dự án:

```bash
# 1. Cài đặt các gói phụ thuộc
npm install

# 2. Khởi chạy Vite Dev Server
npm run dev
```
* Mở trình duyệt và truy cập: **`http://localhost:3000`** (hoặc `http://localhost:5173`).

---

## 📊 5. Đặc tả Kỹ thuật & Thuật toán Machine Learning

### 5.1. Xử lý biến mục tiêu (Target Transformation)
Phân phối giá nhà đất trong thực tế thường bị lệch phải nghiêm trọng (Right-skewed). Mô hình áp dụng phép biến đổi logarit tự nhiên:
$$y_{log} = \ln(1 + \text{price})$$
Khi dự đoán, kết quả được quy đổi ngược lại thang đo thực tế:
$$\hat{\text{price}} = \exp(\hat{y}_{log}) - 1$$

### 5.2. Công thức tính khoảng tin cậy 95% (Confidence Interval)
Khoảng tin cậy được tính toán từ độ lệch chuẩn của phần dư trên tập kiểm thử ($\sigma_{residuals}$):
$$\text{Price}_{\text{min}} = \exp\left(\hat{y}_{log} - 1.96 \cdot \sigma_{residuals}\right) - 1$$
$$\text{Price}_{\text{max}} = \exp\left(\hat{y}_{log} + 1.96 \cdot \sigma_{residuals}\right) - 1$$

### 5.3. Độ đo đánh giá trên tập kiểm thử (Test Metrics)
* **$R^2$ Score**: `0.9842` (Mô hình giải thích được 98.4% độ biến thiên dữ liệu).
* **Mean Absolute Error (MAE)**: `182.45` Triệu VNĐ.
* **Root Mean Squared Error (RMSE)**: `245.12` Triệu VNĐ.
* **Residual Std ($\sigma_{log}$)**: `0.0582`.

---

## 🔌 6. Danh sách API Endpoints chính

| Phương thức | Endpoint | Chức năng | Định dạng gửi |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Chuyển hướng tới tài liệu Swagger | None |
| `GET` | `/docs` | Giao diện kiểm thử Swagger UI | None |
| `GET` | `/api/v1/meta` | Lấy siêu dữ liệu, danh mục quận & thông số mô hình | None |
| `POST` | `/api/v1/estimate` | Tính toán định giá, khoảng tin cậy & 5 BĐS tương đồng | JSON (`PropertyInput`) |
