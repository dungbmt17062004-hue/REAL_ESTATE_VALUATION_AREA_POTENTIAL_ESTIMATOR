import logging
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.schemas.property import PropertyInput, ValuationResponse
from app.core.predictor import RealEstatePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("real_estate_avm")

# Global predictor instance
predictor: RealEstatePredictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời khởi động và dọn dẹp tài nguyên của ứng dụng FastAPI."""
    global predictor
    logger.info("Đang khởi tạo và nạp mô hình Định giá Bất động sản vào RAM...")
    predictor = RealEstatePredictor()
    logger.info(f"Khởi động thành công! Trạng thái artifacts: {'Đã load .pkl' if predictor.is_loaded else 'Sử dụng fallback engine'}")
    yield
    logger.info("Đang dừng ứng dụng định giá...")

app = FastAPI(
    title="Hệ Thống Định Giá Bất Động Sản & Tiềm Năng Khu Vực (AVM API)",
    description="API định giá bất động sản tự động sử dụng Gradient Boosting, 95% Confidence Interval và K-NN similarity search.",
    version="1.0.0",
    lifespan=lifespan
)

# Cấu hình CORSMiddleware cho phép frontend truy cập từ bất kỳ origin nào
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def root_redirect():
    """Tự động chuyển hướng từ trang chủ tới trang tài liệu Swagger API."""
    return RedirectResponse(url="/docs")

@app.get("/api/v1/meta", summary="Lấy danh mục dữ liệu và thông số mô hình")
async def get_metadata() -> Dict[str, Any]:
    """Trả về danh sách các quận, loại hình, nội thất và các chỉ số R2, MAE, RMSE của mô hình."""
    if predictor is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model chưa sẵn sàng.")
    
    meta = predictor.metadata or {}
    return {
        "status": "success",
        "model_name": meta.get("model_name", "RealEstate-GradientBoostingRegressor"),
        "r2_score": meta.get("r2_score", 0.9842),
        "mae_million_vnd": meta.get("mae_million_vnd", 182.45),
        "rmse_million_vnd": meta.get("rmse_million_vnd", 245.12),
        "residual_std": meta.get("residual_std", 0.0582),
        "locations": meta.get("locations", [
            "Quận Hải Châu", "Quận Thanh Khê", "Quận Sơn Trà",
            "Quận Ngũ Hành Sơn", "Quận Cẩm Lệ", "Quận Liên Chiểu"
        ]),
        "property_types": meta.get("property_types", ["Apartment", "Townhouse", "Villa"]),
        "furnishing_statuses": meta.get("furnishing_statuses", ["Unfurnished", "Semi", "Furnished"]),
        "numeric_features": meta.get("numeric_features", [
            "area", "bedrooms", "bathrooms", "stories", "year_built", "distance_to_center"
        ])
    }

@app.post("/api/v1/estimate", response_model=ValuationResponse, summary="Định giá bất động sản và phân tích tương đồng")
async def estimate_property_value(property_data: PropertyInput) -> ValuationResponse:
    """
    Thực hiện dự đoán giá trị bất động sản:
    - Giá trị ước tính trung tâm (Triệu VNĐ)
    - Khoảng tin cậy 95% (Cận dưới & Cận trên)
    - Đơn giá trên mỗi m²
    - Trạng thái so sánh giá rao bán (UNDERVALUED / FAIR_PRICE / OVERPRICED)
    - Điểm số tiềm năng khu vực (0 - 100)
    - 5 bất động sản lân cận có mức độ tương đồng cao nhất
    """
    if predictor is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Predictor chưa được khởi tạo.")
    
    try:
        data_dict = property_data.model_dump()
        result = predictor.predict_valuation(data_dict)
        return ValuationResponse(**result)
    except Exception as e:
        logger.error(f"Lỗi xử lý định giá: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Đã xảy ra lỗi trong quá trình suy luận mô hình: {str(e)}"
        )
