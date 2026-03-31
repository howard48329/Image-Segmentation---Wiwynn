import os
import urllib.request
from pathlib import Path

def get_model_url(model_name: str) -> str:
    """根據模型檔名回傳 Ultralytics 的 Github Release 下載網址"""
    # 這裡可以根據您的客製化模型，將網址換成您公司的 S3 或 Google Drive 連結
    return f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{model_name}"

def download_model(model_name: str, dest_dir: str) -> str:
    """下載指定的 YOLO 模型權重到目標資料夾，並回傳完整的儲存路徑。"""
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    file_path = dest_path / model_name
    
    # 【模型權重檢查機制】
    if not file_path.exists():
        print(f"🕵️ 系統檢查：找不到模型權重 {model_name}，正在自動從網路下載...")
        url = get_model_url(model_name)
        try:
            # 加入簡單的反爬蟲假裝機制或直接使用 urlretrieve
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                # 這裡為了簡單，不加上次寫的進度條，直接串流寫入
                out_file.write(response.read())
            print(f"✅ 成功下載並儲存至: {file_path}")
        except Exception as e:
            print(f"❌ 下載 {model_name} 失敗，請確認網路連線: {e}")
            raise
    else:
        print(f"✅ 系統檢查：已存在模型權重 {file_path}，跳過下載直接使用。")
        
    return str(file_path)

if __name__ == "__main__":
    # 如果獨立執行此腳本，則預設下載這兩個檔案到 model_weights 目錄下
    download_model("yolov8n-seg.pt", "model_weights")
    download_model("yolov8n-pose.pt", "model_weights")
