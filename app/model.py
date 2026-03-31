import cv2
import numpy as np
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 將專案根目錄加入路徑，以便讓下方可以引入 `app` 與 `scripts` 模組
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from app.config import settings
from scripts.download_weights import download_model

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    logging.warning("尚未安裝 ultralytics。請在終端機執行: pip install ultralytics")

logger = logging.getLogger(__name__)

class SegmentationEngine:
    """
    高精度雙引擎架構 (Top-Down Crop-to-Pose Architecture):
    
    1. Seg Model (第一層推論): 負責精準切割出全圖動物的地形圖 (Mask) 與輪廓矩形 (BBox)。
    2. 局部裁切 (Image Cropping): 透過 BBox 將動物特寫以陣列切片 (Slice) 的方式單獨截取。
                         並加入適當的 Padding (Margin) 以提供神經網路推論所需的上下文。
    3. Pose Model (第二層推論): 針對每一隻單獨的動物特寫，專門放大推論眼睛關鍵點 (Keypoints)，
                         最後透過【座標反向推導機制】(Global Coordinate Mapping) 換回原圖之像素座標。
    """
    def __init__(self):
        self.seg_model = None
        self.pose_model = None
        # COCO 類別對應：0 為人(person)，14 為鳥(bird)，後續為各種爬行/哺乳類動物
        self.target_classes = [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    def load(self, seg_model_name: str = None, pose_model_name: str = None, weights_dir: str = None):
        """
        載入雙引擎模型權重，已整合自動防呆檢查與下載機制。
        優先採用傳入的參數，否則根據 Pydantic 自動讀取環境變數 (.env) 的預設值。
        """
        # 讀取檔名與目錄設定
        seg_model_name = seg_model_name or settings.seg_model_name
        pose_model_name = pose_model_name or settings.pose_model_name
        weights_dir = weights_dir or settings.weights_dir
        
        # 🛡️ 模型權重防呆檢查機制
        # 如果權重檔不在 model_weights 目錄下，它會自動呼叫腳本進行網路下載
        logger.info(f"開始檢查權重檔是否就緒 ({weights_dir} 資料夾)...")
        seg_model_path = download_model(seg_model_name, weights_dir)
        pose_model_path = download_model(pose_model_name, weights_dir)

        logger.info(f"正在載入 Segmentation 模型 ({seg_model_path}) ...")
        self.seg_model = YOLO(seg_model_path)
        
        logger.info(f"正在載入 Pose 模型 ({pose_model_path}) ...")
        self.pose_model = YOLO(pose_model_path)
        logger.info("🔥 高精度雙引擎模型載入成功！")

    def predict(self, img_cv2: np.ndarray) -> List[Dict[str, Any]]:
        """
        輸入單張 OpenCV 影像，回傳精準量測動物所需的預測資料陣列。
        """
        if not self.seg_model or not self.pose_model:
            raise RuntimeError("模型尚未載入，請先呼叫 load()。")

        # --- 1. 第一層推論：抓取全張圖片中所有目標(人/鳥/動物)的 Mask 與 Bbox ---
        seg_results = self.seg_model.predict(source=img_cv2, conf=0.25, classes=self.target_classes, verbose=False)[0]

        predictions = []
        if seg_results.masks is None or seg_results.boxes is None:
            return predictions # 沒抓到動物就提早返回，節省算力

        seg_boxes = seg_results.boxes.xyxy.cpu().numpy()
        class_ids = seg_results.boxes.cls.cpu().numpy().astype(int)
        masks_data = seg_results.masks.data.cpu().numpy()
        
        original_shape = img_cv2.shape[:2] # (H, W)
        img_h, img_w = original_shape

        for i in range(len(seg_boxes)):
            box = seg_boxes[i]
            cls_id = class_ids[i]
            class_name = self.seg_model.names[cls_id]
            
            # 取出原始 Bounding box float 座標並小數轉無號整數
            x1, y1, x2, y2 = [int(v) for v in box]
            
            # --- 2. 局部裁切 (Crop-to-Pose) 的精華：加上 Margin ---
            # 若直接貼著邊緣切會讓卷積神經網絡(CNN)「瞎掉」，喪失判斷臉部的上下文 (Context Feature)。
            # 通常業界實作會給予外圍 5% ~ 10% 的環境 padding。
            margin_x = int((x2 - x1) * 0.05)
            margin_y = int((y2 - y1) * 0.05)
            
            # 利用 max, min 限制裁切座標不會「超出版界 (Out of bounds)」導致陣列崩潰
            crop_x1 = max(0, x1 - margin_x)
            crop_y1 = max(0, y1 - margin_y)
            crop_x2 = min(img_w, x2 + margin_x)
            crop_y2 = min(img_h, y2 + margin_y)
            
            # 拿到被裁切下來的「高解析度」動物影像特寫塊
            crop_img = img_cv2[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # --- 3. 處理 Mask ---
            # 將縮放過的 Mask 復原為全圖大小，並二值化
            mask_resized = cv2.resize(masks_data[i], (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            binary_mask = (mask_resized > 0.5).astype(np.uint8)
            
            # --- 4. 第二層推論：高精度尋找特徵點 ---
            # 把放大特寫的 crop_img 丟進去，且強制不限制類別以便讓官方 pose 發揮盲測潛力
            pose_results = self.pose_model.predict(source=crop_img, conf=0.15, verbose=False)[0]
            
            matched_kpts = {"right_eye": None, "left_eye": None}
            
            # 如果在特寫畫面中有順利抓到任何疑似的特徵點骨架
            if pose_results.keypoints is not None and len(pose_results.keypoints.data) > 0:
                # 只取信心度最高 (第一組) 的預測結果
                kpts = pose_results.keypoints.data[0].cpu().numpy() # shape [17, 3]
                
                # COCO 人體骨架 index: 1 (左眼), 2 (右眼)
                lx_crop, ly_crop, lconf = kpts[1]
                rx_crop, ry_crop, rconf = kpts[2]
                
                # --- 5. 精彩加分點：局部座標反推全域座標 (Global Coordinate Mapping) ---
                # 將特寫圖裡面的 (x, y)，加上當初裁切起點的 (crop_x1, crop_y1)，反算回實際像素座標！
                if rconf > 0.1: 
                    matched_kpts["right_eye"] = (int(rx_crop) + crop_x1, int(ry_crop) + crop_y1)
                if lconf > 0.1: 
                    matched_kpts["left_eye"]  = (int(lx_crop) + crop_x1, int(ly_crop) + crop_y1)

            predictions.append({
                "class_id": cls_id,
                "class_name": class_name,
                "bbox": box.tolist(),
                "mask": binary_mask,
                "keypoints": matched_kpts
            })

        return predictions
