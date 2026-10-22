import os
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Định nghĩa đường dẫn lưu trữ artifacts
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "backend" / "models"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

LOCATIONS = [
    "Quận Hải Châu", "Quận Thanh Khê", "Quận Sơn Trà", 
    "Quận Ngũ Hành Sơn", "Quận Cẩm Lệ", "Quận Liên Chiểu"
]
PROPERTY_TYPES = ["Apartment", "Townhouse", "Villa"]
FURNISHING_STATUSES = ["Unfurnished", "Semi", "Furnished"]

LOCATION_BASE_PRICE_M2 = {
    "Quận Hải Châu": 55.0,      # Triệu VNĐ / m2
    "Quận Sơn Trà": 48.0,
    "Quận Ngũ Hành Sơn": 42.0,
    "Quận Thanh Khê": 38.0,
    "Quận Cẩm Lệ": 30.0,
    "Quận Liên Chiểu": 25.0
}

TYPE_MULTIPLIER = {
    "Apartment": 0.9,
    "Townhouse": 1.25,
    "Villa": 1.8
}

FURNISHING_BONUS = {
    "Unfurnished": 0.0,
    "Semi": 150.0,             # Triệu VNĐ
    "Furnished": 400.0          # Triệu VNĐ
}

def generate_synthetic_data(n_samples: int = 4000, random_state: int = 42) -> pd.DataFrame:
    """Sinh dữ liệu bất động sản giả lập có tương quan kinh tế thực tế."""
    np.random.seed(random_state)
    
    locations = np.random.choice(LOCATIONS, size=n_samples, p=[0.25, 0.2, 0.2, 0.15, 0.1, 0.1])
    prop_types = np.random.choice(PROPERTY_TYPES, size=n_samples, p=[0.45, 0.45, 0.10])
    furnishings = np.random.choice(FURNISHING_STATUSES, size=n_samples, p=[0.3, 0.4, 0.3])
    
    # Sinh đặc trưng số học
    areas = []
    for pt in prop_types:
        if pt == "Apartment":
            areas.append(np.random.normal(65, 20))
        elif pt == "Townhouse":
            areas.append(np.random.normal(110, 35))
        else: # Villa
            areas.append(np.random.normal(260, 60))
    areas = np.clip(np.array(areas), 25.0, 600.0)
    
    bedrooms = np.clip(np.round(areas / 35 + np.random.normal(0, 0.6)), 1, 8).astype(int)
    bathrooms = np.clip(np.round(bedrooms * 0.8 + np.random.normal(0, 0.4)), 1, 6).astype(int)
    
    stories = []
    for pt in prop_types:
        if pt == "Apartment":
            stories.append(1)
        elif pt == "Townhouse":
            stories.append(int(np.clip(np.random.choice([2, 3, 4, 5], p=[0.2, 0.5, 0.25, 0.05]), 1, 5)))
        else:
            stories.append(int(np.clip(np.random.choice([2, 3], p=[0.6, 0.4]), 1, 4)))
    stories = np.array(stories)
    
    years_built = np.random.randint(1995, 2026, size=n_samples)
    dist_to_center = []
    for loc in locations:
        if loc == "Quận Hải Châu":
            dist_to_center.append(np.random.uniform(0.5, 3.5))
        elif loc in ["Quận Thanh Khê", "Quận Sơn Trà"]:
            dist_to_center.append(np.random.uniform(3.0, 7.0))
        else:
            dist_to_center.append(np.random.uniform(6.0, 15.0))
    dist_to_center = np.array(dist_to_center)

    # Tính giá nhà thực tế theo logic kinh tế (Đơn vị: Triệu VNĐ)
    prices = []
    for i in range(n_samples):
        loc = locations[i]
        pt = prop_types[i]
        fn = furnishings[i]
        area = areas[i]
        yr = years_built[i]
        dist = dist_to_center[i]
        st = stories[i]
        
        base_m2 = LOCATION_BASE_PRICE_M2[loc] * TYPE_MULTIPLIER[pt]
        
        # Yếu tố khấu hao theo tuổi nhà (-0.7%/năm, tối đa giảm 20%)
        age = max(0, 2026 - yr)
        age_factor = max(0.80, 1.0 - (age * 0.007))
        
        # Yếu tố khoảng cách tới trung tâm
        dist_factor = max(0.85, 1.05 - (dist * 0.015))
        
        # Giá đất/kết cấu
        total_val = (base_m2 * area * age_factor * dist_factor)
        if pt in ["Townhouse", "Villa"] and st > 1:
            total_val += (st - 1) * (area * 0.25 * 5.5)  # Cộng thêm chi phí sàn xây dựng
            
        total_val += FURNISHING_BONUS[fn]
        
        # Biến thiên thị trường ngẫu nhiên (+- 6%)
        noise = np.random.normal(1.0, 0.06)
        prices.append(round(total_val * noise, 2))
        
    df = pd.DataFrame({
        "id": range(1, n_samples + 1),
        "area": np.round(areas, 1),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "year_built": years_built,
        "distance_to_center": np.round(dist_to_center, 2),
        "location": locations,
        "property_type": prop_types,
        "furnishing_status": furnishings,
        "price": prices
    })
    return df

def train_and_export():
    print("--- 1. Bắt đầu sinh tập dữ liệu giả lập ---")
    df = generate_synthetic_data(n_samples=4500)
    
    # Lưu file CSV dataset
    dataset_csv_path = Path(__file__).resolve().parent / "real_estate_data.csv"
    df.to_csv(dataset_csv_path, index=False, encoding="utf-8-sig")
    print(f"Đã lưu dataset tại: {dataset_csv_path}")

    # Tách features và target
    X = df.drop(columns=["id", "price"])
    y = df["price"]
    
    # Log-transform target
    y_log = np.log1p(y)

    numeric_features = ["area", "bedrooms", "bathrooms", "stories", "year_built", "distance_to_center"]
    categorical_features = ["location", "property_type", "furnishing_status"]

    # Preprocessing pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, numeric_features),
        ("cat", cat_pipeline, categorical_features)
    ])

    X_train, X_test, y_log_train, y_log_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    print("--- 2. Fit Preprocessor và huấn luyện Regressor ---")
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Dùng GradientBoostingRegressor (tương thích scikit-learn thuần, dễ thay thế bằng LightGBM)
    model = GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.85,
        random_state=42
    )
    model.fit(X_train_trans, y_log_train)

    # Đánh giá trên tập test
    y_log_pred = model.predict(X_test_trans)
    y_pred = np.expm1(y_log_pred)
    y_true = np.expm1(y_log_test)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Tính độ lệch chuẩn phần dư log (log residuals std) để làm khoảng tin cậy 95% (+- 1.96 * std)
    residuals_log = y_log_test - y_log_pred
    residual_std = float(np.std(residuals_log))

    print(f"Kết quả Test - MAE: {mae:.2f} Triệu VNĐ | RMSE: {rmse:.2f} Triệu VNĐ | R2 Score: {r2:.4f}")
    print(f"Residual Std (log scale): {residual_std:.4f}")

    print("--- 3. Xây dựng K-NN Recommender Index ---")
    # Fit K-NN trên toàn bộ không gian đặc trưng để phục vụ tìm kiếm nhà tương đồng
    X_all_trans = preprocessor.transform(X)
    knn = NearestNeighbors(n_neighbors=6, metric="cosine")
    knn.fit(X_all_trans)

    property_index = {
        "knn_model": knn,
        "properties_df": df[["id", "location", "property_type", "area", "bedrooms", "bathrooms", "price"]].copy(),
        "feature_matrix": X_all_trans
    }

    print("--- 4. Xuất các file artifacts ---")
    joblib.dump(model, ARTIFACTS_DIR / "housing_model.pkl")
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.pkl")
    joblib.dump(property_index, ARTIFACTS_DIR / "property_index.pkl")

    metadata = {
        "model_name": "RealEstate-GradientBoostingRegressor",
        "r2_score": round(r2, 4),
        "mae_million_vnd": round(mae, 2),
        "rmse_million_vnd": round(rmse, 2),
        "residual_std": round(residual_std, 4),
        "n_samples": len(df),
        "locations": LOCATIONS,
        "property_types": PROPERTY_TYPES,
        "furnishing_statuses": FURNISHING_STATUSES,
        "numeric_features": numeric_features
    }
    
    with open(ARTIFACTS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Hoàn thành Bước 1! Tất cả artifacts đã được lưu trữ sẵn sàng cho FastAPI.")

if __name__ == "__main__":
    train_and_export()
