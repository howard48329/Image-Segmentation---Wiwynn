import os
import sys
import json
import urllib.request
import zipfile
from pathlib import Path

def download_progress_hook(block_num, block_size, total_size):
    """終端機下載進度條回呼函數"""
    if total_size > 0:
        downloaded = block_num * block_size
        percent = min(100, int(downloaded * 100 / total_size))
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        
        # 繪製由 = 組成的進度條
        bar_length = 30
        filled_length = int(bar_length * percent // 100)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        
        # 使用 \r 回到行首，覆寫目前行的輸出
        sys.stdout.write(f"\r下載進度: [{bar}] {percent}% ({downloaded_mb:.1f}MB / {total_mb:.1f}MB)")
        sys.stdout.flush()

# 定義常數
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
CACHE_DIR = Path("data/.cache")
INPUT_DIR = Path("data/input")

def download_and_extract_annotations():
    """下載並解壓縮 COCO 2017 validation annotations 加入快取"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / "annotations_trainval2017.zip"
    extract_path = CACHE_DIR / "annotations"
    
    # 檢查是否已經解壓縮過
    if not extract_path.exists():
        # 檢查是否已經下載過 zip
        if not zip_path.exists():
            print(f"正在下載 COCO 標註檔 (約 241MB)，這可能需要幾分鐘...")
            urllib.request.urlretrieve(COCO_ANNOTATIONS_URL, zip_path, reporthook=download_progress_hook)
            print("\n標註檔下載完成！")
            
        print("正在解壓縮標註檔...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(CACHE_DIR)
            print("解壓縮完成！")
        except zipfile.BadZipFile:
            print("發現損壞的壓縮檔 (可能是先前下載中斷)。正在重新下載...")
            if zip_path.exists():
                zip_path.unlink()  # 刪除壞掉的檔案
            urllib.request.urlretrieve(COCO_ANNOTATIONS_URL, zip_path, reporthook=download_progress_hook)
            print() # 前面的進度條沒有換行，這裡補上一個換行
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(CACHE_DIR)
            print("重新下載並解壓縮完成！")
        
    return extract_path / "instances_val2017.json"

def download_images(num_images=5):
    """篩選並下載含有至少兩隻動物的影像"""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    ann_file = download_and_extract_annotations()
    
    print("正在載入 JSON 標註檔...")
    with open(ann_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
        
    # 指定精確的篩選名單：人、貓、狗、羊、牛、馬
    desired_classes = {'person', 'cat', 'dog', 'sheep', 'cow', 'horse'}
    target_category_ids = {cat['id'] for cat in coco_data['categories'] if cat['name'] in desired_classes}
    
    # 建立 image_id 的快速查詢字典，方便稍後取得圖片網址
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # 計算每張影像中的目標(人與動物)數量
    target_count_per_image = {}
    for ann in coco_data['annotations']:
        if ann['category_id'] in target_category_ids:
            img_id = ann['image_id']
            target_count_per_image[img_id] = target_count_per_image.get(img_id, 0) + 1
            
    # 篩選出有包含最少 2 個目標的 image_ids
    # 這裡我們設定 >= 2，確保有足夠的目標可以進行量測
    target_image_ids = [img_id for img_id, count in target_count_per_image.items() if count >= 2]
    
    print(f"在驗證集中找到 {len(target_image_ids)} 張含有至少兩個人或動物的影像。")
    print(f"準備下載前 {num_images} 張作為測試用圖片...")
    
    # 開始利用圖片網址下載目標圖片
    downloaded = 0
    for img_id in target_image_ids:
        if downloaded >= num_images:
            break
            
        img_info = images_dict[img_id]
        img_url = img_info['coco_url']
        filename = img_info['file_name']
        save_path = INPUT_DIR / filename
        
        if not save_path.exists():
            print(f"下載圖片 [{downloaded+1}/{num_images}]: {filename} ...")
            try:
                urllib.request.urlretrieve(img_url, save_path)
                downloaded += 1
            except Exception as e:
                print(f"下載 {filename} 失敗: {e}")
        else:
            print(f"圖片已存在: {filename}，跳過下載。")
            downloaded += 1
            
    print(f"\n成功準備了 {downloaded} 張測試圖片至 '{INPUT_DIR}' 資料夾。")

if __name__ == "__main__":
    # 執行腳本，預設下載 50 張符合條件的圖片
    download_images(num_images=300)