from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class AnalyzeResponse(BaseModel):
    """
    API 分析結果的標準回應結構 (Response Schema)
    透過 Pydantic 定義，除了確保型別安全，還能自動在 Swagger UI 生成精美的文件！
    """
    status: str = Field(..., description="API 處理狀態，通常為 'success' 或 'warning'")
    filename: Optional[str] = Field(None, description="上傳處理的原始圖片檔名")
    message: Optional[str] = Field(None, description="系統訊息，通常在沒有偵測到物體或出錯時出現")
    
    # 這裡因為 animal_1, animal_2 等是動態擴充的 Key，我們用 Dict 來保留彈性
    # 如果未來業務擴充把結構固定死了，可以直接換成另一組嵌套的 BaseModel
    measurements: Optional[Dict[str, Any]] = Field(
        None, 
        description="距離量測結果，包含單一目標的眼距與跨目標的右眼距離",
        json_schema_extra={
            "example": {
                "animal_1": {"class": "person", "eye_distance_pixels": 45.2},
                "animal_2": {"class": "cat", "eye_distance_pixels": 32.1},
                "inter_animal_right_eye_distance_pixels": 150.5
            }
        }
    )
    
    output_image_path: Optional[str] = Field(None, description="伺服器上儲存的視覺化標註圖片路徑")
    detected_animals_count: int = Field(0, description="畫面上成功辨識出的目標(動物/人)總數")
