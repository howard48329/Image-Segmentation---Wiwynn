from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import logging

# 匯入我們寫好的所有模組
from app.model import SegmentationEngine
from app.utils.geometry import analyze_measurements
from app.utils.data_proc import bytes_to_cv2, draw_and_save_results, append_to_csv
from app.schemas import AnalyzeResponse

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Image Segmentation Metrology API",
    description="結合影像分割模型於測量學 (Metrology) 之應用：計算動物眼睛間距",
    version="1.0.0"
)

# 全域初始化雙引擎模型變數
ai_model = SegmentationEngine()

@app.on_event("startup")
async def startup_event():
    """
    API 啟動時執行的動作：事先將雙模型的權重 (.pt 檔) 載入記憶體 (Warm-up)，
    避免每次使用者打 API 時都要重新讀取檔案。
    """
    logger.info("啟動 API：正在準備載入模型權重...")
    # 這裡直接呼叫預設的 yolov8n-seg.pt 與 yolov8n-pose.pt
    ai_model.load()
    logger.info("模型載入完成成功。")

@app.get("/")
def read_root():
    """
    健康檢查或專案介紹入口。
    """
    return {
        "status": "online",
        "message": "歡迎來到 Image Segmentation Metrology API。請前往 /docs 查看詳細 API 文件。"
    }

@app.post("/api/analyze", response_model=AnalyzeResponse, summary="分析影像並計算特徵點幾何距離")
async def analyze_image(file: UploadFile = File(...)):
    """
    核心 API (Metrology End-to-End Pipeline)：
    1. 接收動物圖片 (Bytes 轉 OpenCV)
    2. 雙引擎推論 (取得 Mask 重疊特徵點)
    3. 幾何量測 (計算眼距)
    4. 輸出視覺化結果與 CSV
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="上傳的檔案必須是圖片格式 (如 jpg, png)")
    
    try:
        # 1. 讀取圖片 Bytes 轉為 OpenCV 格式
        image_bytes = await file.read()
        img_cv2 = bytes_to_cv2(image_bytes)
        
        # 2. 呼叫高精度雙引擎模型進行預測
        predictions = ai_model.predict(img_cv2)
        
        # 防呆機制：如果沒有偵測到動物
        if len(predictions) == 0:
            return AnalyzeResponse(
                status="warning",
                filename=file.filename,
                message="在圖片中未偵測到任何符合的目標。"
            )
            
        # 3. 計算測量學距離 (尤拉公式)
        measurements = analyze_measurements(predictions)
        
        # 4. 產生視覺化的標註圖檔 (畫線、畫框、畫 Mask)
        output_img_path = draw_and_save_results(img_cv2, predictions, file.filename)
        
        # 5. 將量測數據紀錄到 CSV 報表中
        append_to_csv(file.filename, measurements)
        
        # 6. 回傳強型別驗證過的 Pydantic Schema
        return AnalyzeResponse(
            status="success",
            filename=file.filename,
            measurements=measurements,
            output_image_path=output_img_path,
            detected_animals_count=len(predictions)
        )
        
    except Exception as e:
        logger.error(f"分析影像時發生錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail="伺服器處理影像時遭遇內部錯誤。")
