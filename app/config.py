from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    透過 Pydantic Settings 來管理專案內的環境變數。
    它會自動尋找專案根目錄下的 .env 檔案並載入值。
    """
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # 取得 .env 中的權重檔設定，若 .env 沒定義則使用右側的字串作為預設防呆
    weights_dir: str = "model_weights"
    seg_model_name: str = "yolov8n-seg.pt"
    pose_model_name: str = "yolov8n-pose.pt"

    class Config:
        env_file = ".env"

# 將 Settings 實例化為一個全域物件，方便各模組引入
settings = Settings()
