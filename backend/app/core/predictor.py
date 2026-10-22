import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Thư mục chứa artifacts đã xuất từ Bước 1
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

class RealEstatePredictor:
    """
    Class quản lý suy luận định giá bất động sản, tính khoảng tin cậy 95%,
    đánh giá tiềm năng đầu tư và tìm kiếm bất động sản tương đồng bằng K-NN.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or MODELS_DIR
        self.model = None
        self.preprocessor = None
        self.property_index = None
        self.knn_model = None
        self.properties_df = None
        self.feature_matrix = None
        self.metadata: Dict[str, Any] = {}
        self.residual_std: float = 0.0582
        self.is_loaded: bool = False
        self.load_artifacts()

    def load_artifacts(self) -> None:
        """Nạp các file .pkl và metadata.json từ thư mục models."""
        metadata_path = self.models_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                self.residual_std = float(self.metadata.get("residual_std", 0.0582))
                logger.info(f"Loaded metadata successfully from {metadata_path}")
            except Exception as e:
                logger.warning(f"Không thể đọc metadata.json: {e}")

        # Thử nạp joblib models nếu có
        try:
            import joblib
            model_path = self.models_dir / "housing_model.pkl"
            preprocessor_path = self.models_dir / "preprocessor.pkl"
            property_index_path = self.models_dir / "property_index.pkl"

            if model_path.exists() and preprocessor_path.exists() and property_index_path.exists():
                self.model = joblib.load(model_path)
                self.preprocessor = joblib.load(preprocessor_path)
                self.property_index = joblib.load(property_index_path)
                
                self.knn_model = self.property_index["knn_model"]
                self.properties_df = self.property_index["properties_df"]
                self.feature_matrix = self.property_index["feature_matrix"]
                self.is_loaded = True
                logger.info("Nạp đầy đủ các artifacts scikit-learn (.pkl) thành công!")
                return
        except Exception as e:
            logger.warning(f"Chưa có hoặc không thể nạp artifacts .pkl ({e}). Kích hoạt engine dự phòng tối ưu.")

        # Dự phòng engine nội suy toán học chính xác theo training script
        self._init_fallback_engine()

    def _init_fallback_engine(self) -> None:
        """Khởi tạo engine dự phòng giả lập tương quan dữ liệu 4,500 mẫu nếu pkl chưa train."""
        from training.train_model import generate_synthetic_data, LOCATIONS, PROPERTY_TYPES, FURNISHING_STATUSES
        try:
            sample_df = generate_synthetic_data(n_samples=500, random_state=42)
            self.properties_df = sample_df[["id", "location", "property_type", "area", "bedrooms", "bathrooms", "price"]].copy()
        except Exception:
            # Dữ liệu mẫu khởi tạo an toàn
            records = [
                {"id": 101, "location": "Quận Hải Châu", "property_type": "Apartment", "area": 85.0, "bedrooms": 2, "bathrooms": 2, "price": 4250.0},
                {"id": 102, "location": "Quận Sơn Trà", "property_type": "Apartment", "area": 75.0, "bedrooms": 2, "bathrooms": 2, "price": 3400.0},
                {"id": 103, "location": "Quận Ngũ Hành Sơn", "property_type": "Villa", "area": 250.0, "bedrooms": 4, "bathrooms": 4, "price": 18500.0},
                {"id": 104, "location": "Quận Thanh Khê", "property_type": "Townhouse", "area": 110.0, "bedrooms": 3, "bathrooms": 3, "price": 5300.0},
                {"id": 105, "location": "Quận Cẩm Lệ", "property_type": "Townhouse", "area": 100.0, "bedrooms": 3, "bathrooms": 2, "price": 3800.0},
                {"id": 106, "location": "Quận Liên Chiểu", "property_type": "Apartment", "area": 60.0, "bedrooms": 2, "bathrooms": 1, "price": 1600.0},
            ]
            self.properties_df = pd.DataFrame(records)

    def calculate_potential_score(
        self,
        distance_to_center: float,
        year_built: int,
        location: str,
        property_type: str,
        price_diff_pct: Optional[float]
    ) -> int:
        """
        Tính điểm tiềm năng bất động sản (0 - 100):
        - Vị trí trung tâm & cự ly di chuyển (max 35 điểm)
        - Năm xây dựng / khấu hao công trình (max 25 điểm)
        - Vị trí chiến lược / du lịch / hạ tầng (max 20 điểm)
        - Loại hình bất động sản & thanh khoản (max 10 điểm)
        - Giá mua hời so với định giá (max 10 điểm thưởng)
        """
        score = 0.0

        # 1. Cự ly tới trung tâm
        if distance_to_center <= 2.0:
            score += 35.0
        elif distance_to_center <= 5.0:
            score += 28.0
        elif distance_to_center <= 8.0:
            score += 20.0
        elif distance_to_center <= 12.0:
            score += 14.0
        else:
            score += 8.0

        # 2. Năm xây dựng
        age = max(0, 2026 - year_built)
        if age <= 2:
            score += 25.0
        elif age <= 5:
            score += 21.0
        elif age <= 10:
            score += 16.0
        elif age <= 18:
            score += 10.0
        else:
            score += 5.0

        # 3. Khu vực phát triển
        location_scores = {
            "Quận Hải Châu": 20.0,      # Trung tâm lõi tài chính & hành chính
            "Quận Sơn Trà": 19.0,       # Đô thị ven biển & du lịch cao cấp
            "Quận Ngũ Hành Sơn": 18.0,  # Khu resort, trường quốc tế, công nghệ
            "Quận Thanh Khê": 15.0,     # Trung tâm thương mại dân sinh
            "Quận Cẩm Lệ": 13.0,        # Đô thị mở rộng về phía Nam
            "Quận Liên Chiểu": 12.0     # Cảng biển Liên Chiểu & khu công nghiệp
        }
        score += location_scores.get(location, 14.0)

        # 4. Loại hình bất động sản
        type_scores = {
            "Apartment": 9.0,
            "Townhouse": 10.0,
            "Villa": 8.0
        }
        score += type_scores.get(property_type, 8.0)

        # 5. Điểm thưởng định giá hời (nếu có giá rao bán)
        if price_diff_pct is not None:
            if price_diff_pct <= -15.0:
                score += 10.0
            elif price_diff_pct <= -5.0:
                score += 6.0
            elif price_diff_pct >= 15.0:
                score -= 6.0

        return int(np.clip(round(score), 10, 99))

    def _fallback_predict_price(self, data: Dict[str, Any]) -> float:
        """Tính giá nội suy chuẩn xác theo công thức kinh tế học của bài toán."""
        base_prices = {
            "Quận Hải Châu": 55.0,
            "Quận Sơn Trà": 48.0,
            "Quận Ngũ Hành Sơn": 42.0,
            "Quận Thanh Khê": 38.0,
            "Quận Cẩm Lệ": 30.0,
            "Quận Liên Chiểu": 25.0
        }
        type_multipliers = {
            "Apartment": 0.9,
            "Townhouse": 1.25,
            "Villa": 1.8
        }
        furnishing_bonuses = {
            "Unfurnished": 0.0,
            "Semi": 150.0,
            "Furnished": 400.0
        }

        loc = data["location"]
        pt = data["property_type"]
        fn = data["furnishing_status"]
        area = float(data["area"])
        yr = int(data["year_built"])
        dist = float(data["distance_to_center"])
        st = int(data["stories"])

        base_m2 = base_prices.get(loc, 35.0) * type_multipliers.get(pt, 1.0)
        age = max(0, 2026 - yr)
        age_factor = max(0.80, 1.0 - (age * 0.007))
        dist_factor = max(0.85, 1.05 - (dist * 0.015))

        total_val = base_m2 * area * age_factor * dist_factor
        if pt in ["Townhouse", "Villa"] and st > 1:
            total_val += (st - 1) * (area * 0.25 * 5.5)

        total_val += furnishing_bonuses.get(fn, 0.0)
        return float(round(total_val, 2))

    def _find_similar_fallback(self, data: Dict[str, Any], estimated_price: float) -> List[Dict[str, Any]]:
        """Thuật toán tìm kiếm 5 bất động sản tương đồng dựa trên khoảng cách đặc trưng chuẩn hóa."""
        if self.properties_df is None or len(self.properties_df) == 0:
            return []

        df = self.properties_df.copy()
        
        # Tính trọng số sai khác giữa các đặc trưng
        area_diff = np.abs(df["area"] - data["area"]) / max(1.0, data["area"])
        loc_diff = (df["location"] != data["location"]).astype(float)
        type_diff = (df["property_type"] != data["property_type"]).astype(float)
        price_diff = np.abs(df["price"] - estimated_price) / max(1.0, estimated_price)

        # Khoảng cách tổng hợp
        total_dist = (0.35 * area_diff) + (0.30 * loc_diff) + (0.20 * type_diff) + (0.15 * price_diff)
        df["comp_dist"] = total_dist
        
        # Sắp xếp lấy top 5
        top_similar = df.sort_values(by="comp_dist").head(5)
        results = []
        for _, row in top_similar.iterrows():
            sim_pct = round(float(np.clip(100.0 * (1.0 - row["comp_dist"] * 0.6), 65.0, 99.4)), 1)
            results.append({
                "id": int(row["id"]),
                "location": str(row["location"]),
                "property_type": str(row["property_type"]),
                "area": float(row["area"]),
                "bedrooms": int(row["bedrooms"]),
                "bathrooms": int(row["bathrooms"]),
                "price": float(row["price"]),
                "similarity_pct": sim_pct
            })
        return results

    def predict_valuation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Nhận thông số đầu vào và trả về kết quả định giá toàn diện.
        """
        area = float(data["area"])
        asking_price = data.get("asking_price")

        # 1. Dự đoán giá trị ước tính
        if self.is_loaded and self.model is not None and self.preprocessor is not None:
            try:
                input_df = pd.DataFrame([{
                    "area": float(data["area"]),
                    "bedrooms": int(data["bedrooms"]),
                    "bathrooms": int(data["bathrooms"]),
                    "stories": int(data["stories"]),
                    "year_built": int(data["year_built"]),
                    "distance_to_center": float(data["distance_to_center"]),
                    "location": str(data["location"]),
                    "property_type": str(data["property_type"]),
                    "furnishing_status": str(data["furnishing_status"])
                }])

                X_trans = self.preprocessor.transform(input_df)
                log_pred = float(self.model.predict(X_trans)[0])
                estimated_price = float(np.expm1(log_pred))

                # 2. Khoảng tin cậy 95%
                price_min = float(np.expm1(log_pred - 1.96 * self.residual_std))
                price_max = float(np.expm1(log_pred + 1.96 * self.residual_std))

                # 3. K-NN tương đồng
                similar_properties = []
                if self.knn_model is not None and self.properties_df is not None:
                    distances, indices = self.knn_model.kneighbors(X_trans, n_neighbors=6)
                    for d, idx in zip(distances[0], indices[0]):
                        row = self.properties_df.iloc[idx]
                        sim_pct = round(float(np.clip((1.0 - d) * 100.0, 50.0, 99.8)), 1)
                        similar_properties.append({
                            "id": int(row["id"]),
                            "location": str(row["location"]),
                            "property_type": str(row["property_type"]),
                            "area": float(row["area"]),
                            "bedrooms": int(row["bedrooms"]),
                            "bathrooms": int(row["bathrooms"]),
                            "price": float(row["price"]),
                            "similarity_pct": sim_pct
                        })
                    similar_properties = similar_properties[:5]
            except Exception as ex:
                logger.error(f"Lỗi khi chạy model scikit-learn, dùng fallback: {ex}")
                estimated_price = self._fallback_predict_price(data)
                log_p = np.log1p(estimated_price)
                price_min = float(np.expm1(log_p - 1.96 * self.residual_std))
                price_max = float(np.expm1(log_p + 1.96 * self.residual_std))
                similar_properties = self._find_similar_fallback(data, estimated_price)
        else:
            estimated_price = self._fallback_predict_price(data)
            log_p = np.log1p(estimated_price)
            price_min = float(np.expm1(log_p - 1.96 * self.residual_std))
            price_max = float(np.expm1(log_p + 1.96 * self.residual_std))
            similar_properties = self._find_similar_fallback(data, estimated_price)

        # 4. Đơn giá trên mỗi m2
        unit_price_per_m2 = round(estimated_price / max(1.0, area), 2)

        # 5. Đánh giá trạng thái định giá (Asking Price Evaluation)
        price_diff_pct = None
        if asking_price is not None and asking_price > 0:
            price_diff_pct = round(((asking_price - estimated_price) / estimated_price) * 100.0, 2)
            if price_diff_pct < -5.0:
                valuation_status = "UNDERVALUED"
            elif price_diff_pct > 5.0:
                valuation_status = "OVERPRICED"
            else:
                valuation_status = "FAIR_PRICE"
        else:
            valuation_status = "NO_ASKING_PRICE"

        # 6. Tính điểm tiềm năng khu vực
        potential_score = self.calculate_potential_score(
            distance_to_center=float(data["distance_to_center"]),
            year_built=int(data["year_built"]),
            location=str(data["location"]),
            property_type=str(data["property_type"]),
            price_diff_pct=price_diff_pct
        )

        return {
            "estimated_price": round(estimated_price, 2),
            "price_min": round(price_min, 2),
            "price_max": round(price_max, 2),
            "unit_price_per_m2": unit_price_per_m2,
            "valuation_status": valuation_status,
            "price_diff_pct": price_diff_pct,
            "potential_score": potential_score,
            "similar_properties": similar_properties
        }
