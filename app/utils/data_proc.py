import cv2
import numpy as np
import csv
from pathlib import Path
from typing import List, Dict, Any

def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """將 FastAPI 接收到的上傳圖片 (bytes) 轉換為 OpenCV 的像素陣列"""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img_cv2

def draw_and_save_results(img_cv2: np.ndarray, predictions: List[Dict[str, Any]], filename: str) -> str:
    """
    將輪廓 (Mask)、邊界框 (BBox)、特徵點及距離連線繪製於原圖上，
    並將視覺化結果儲存到硬碟的輸出資料夾。
    """
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 複製原圖作為繪圖底板
    img_draw = img_cv2.copy()
    valid_right_eyes = []
    
    for pred in predictions:
        # 1. 畫半透明 Mask 輪廓
        mask = pred["mask"]
        color_mask = np.zeros_like(img_draw)
        # 這裡將分割輪廓塗上萊姆綠色 (BGR: 0, 255, 0)
        color_mask[mask == 1] = [0, 255, 0] 
        cv2.addWeighted(color_mask, 0.4, img_draw, 1.0, 0, img_draw)
        
        # 2. 畫 Bounding Box 
        x1, y1, x2, y2 = [int(v) for v in pred["bbox"]]
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (255, 0, 0), 2) # 藍框
        
        # 標註物體類別名稱與文字背景
        # (這裡做一點小優化，讓文字能有更清楚的黑色底框)
        text = pred["class_name"]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_draw, (x1, max(y1-th-10, 0)), (x1+tw+10, max(y1, 10)), (255, 0, 0), -1)
        cv2.putText(img_draw, text, (x1+5, max(y1-5, 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 3. 畫特徵點與單隻動物雙眼連線
        kpts = pred.get("keypoints", {})
        l_eye = kpts.get("left_eye")
        r_eye = kpts.get("right_eye")
        
        if l_eye:
            cv2.circle(img_draw, l_eye, 5, (0, 0, 255), -1) # 紅點代表左眼
        if r_eye:
            cv2.circle(img_draw, r_eye, 5, (0, 165, 255), -1) # 橘紅點代表右眼
            valid_right_eyes.append(r_eye)
            
        if l_eye and r_eye:
            # 畫出單隻動物雙眼的黃色連線
            cv2.line(img_draw, l_eye, r_eye, (0, 255, 255), 2)
            
    # 4. 畫出最為關鍵的「兩隻動物右眼連線」
    if len(valid_right_eyes) >= 2:
        pt1 = valid_right_eyes[0]
        pt2 = valid_right_eyes[1]
        # 粉紅色的一條粗線！
        cv2.line(img_draw, pt1, pt2, (255, 0, 255), 3)

    output_path = output_dir / f"annotated_{filename}"
    cv2.imwrite(str(output_path), img_draw)
    return str(output_path)

def append_to_csv(filename: str, measurements: Dict[str, Any]):
    """將量測出的像素距離紀錄至統一的 CSV 報表"""
    csv_path = Path("data/sample.csv")
    csv_dir = csv_path.parent
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    csv_exists = csv_path.exists()
    
    # 嘗試從結果中取出前兩隻動物的距離資料
    inter_dist = measurements.get("inter_animal_right_eye_distance_pixels", "N/A")
    animal_1_dist = measurements.get("animal_1", {}).get("eye_distance_pixels", "N/A")
    animal_2_dist = measurements.get("animal_2", {}).get("eye_distance_pixels", "N/A")
    
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not csv_exists:
            # 第一行寫入標題 (Header)
            writer.writerow(["Filename", "Animal_1_Eye_Dist_px", "Animal_2_Eye_Dist_px", "Inter_Animal_Right_Eye_Dist_px"])
            
        writer.writerow([filename, animal_1_dist, animal_2_dist, inter_dist])
