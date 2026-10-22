from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class PropertyInput(BaseModel):
    """Schema dữ liệu đầu vào cho một bất động sản cần định giá."""
    area: float = Field(..., gt=0, le=2000, description="Diện tích sử dụng (m²)")
    bedrooms: int = Field(..., ge=1, le=20, description="Số lượng phòng ngủ")
    bathrooms: int = Field(..., ge=1, le=15, description="Số lượng phòng vệ sinh / phòng tắm")
    stories: int = Field(..., ge=1, le=10, description="Số tầng nhà (1 đối với căn hộ)")
    year_built: int = Field(..., ge=1970, le=2030, description="Năm xây dựng")
    distance_to_center: float = Field(..., ge=0.0, le=50.0, description="Khoảng cách tới trung tâm thành phố (km)")
    location: str = Field(..., description="Quận / Vị trí bất động sản tại Đà Nẵng")
    property_type: str = Field(..., description="Loại hình (Apartment, Townhouse, Villa)")
    furnishing_status: str = Field(..., description="Tình trạng nội thất (Unfurnished, Semi, Furnished)")
    asking_price: Optional[float] = Field(None, ge=0, description="Giá rao bán tham khảo của chủ nhà (Triệu VNĐ)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "area": 85.5,
                "bedrooms": 2,
                "bathrooms": 2,
                "stories": 1,
                "year_built": 2021,
                "distance_to_center": 2.5,
                "location": "Quận Hải Châu",
                "property_type": "Apartment",
                "furnishing_status": "Furnished",
                "asking_price": 4200.0
            }
        }
    }

class SimilarProperty(BaseModel):
    """Schema thông tin cho 1 bất động sản tương đồng tìm thấy qua K-NN."""
    id: int
    location: str
    property_type: str
    area: float
    bedrooms: int
    bathrooms: int
    price: float
    similarity_pct: float

class ValuationResponse(BaseModel):
    """Schema phản hồi kết quả định giá toàn diện."""
    estimated_price: float = Field(..., description="Giá trị ước tính trung tâm (Triệu VNĐ)")
    price_min: float = Field(..., description="Cận dưới khoảng tin cậy 95% (Triệu VNĐ)")
    price_max: float = Field(..., description="Cận trên khoảng tin cậy 95% (Triệu VNĐ)")
    unit_price_per_m2: float = Field(..., description="Đơn giá ước tính trên mỗi m² (Triệu VNĐ/m²)")
    valuation_status: Literal["UNDERVALUED", "FAIR_PRICE", "OVERPRICED", "NO_ASKING_PRICE"] = Field(
        ..., description="Trạng thái so sánh giá rao bán với giá ước tính"
    )
    price_diff_pct: Optional[float] = Field(
        None, description="Tỷ lệ chênh lệch giá rao so với định giá (%)"
    )
    potential_score: int = Field(..., ge=0, le=100, description="Điểm tiềm năng khu vực & khai thác (0 - 100)")
    similar_properties: List[SimilarProperty] = Field(
        default_factory=list, description="Danh sách 5 bất động sản tương đồng lân cận"
    )
