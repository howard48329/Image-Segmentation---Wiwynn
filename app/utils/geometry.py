import math
from typing import List, Dict, Any

def calculate_euclidean_distance(pt1: tuple, pt2: tuple) -> float:
    """計算兩點之間的歐氏距離 (像素為單位)"""
    if not pt1 or not pt2:
        return None
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def analyze_measurements(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    接收神經網路輸出的預測結果，計算特徵點之間的距離。
    包含：單隻動物的雙眼距離、以及不同隻動物的右眼距離。
    """
    measurements = {}
    
    # 紀錄有抓到右眼的動物的特徵點座標
    valid_right_eyes = []
    
    for i, pred in enumerate(predictions):
        kpts = pred.get("keypoints", {})
        left_eye = kpts.get("left_eye")
        right_eye = kpts.get("right_eye")
        
        # 1. 測量單隻動物的雙眼距離
        dist = calculate_euclidean_distance(left_eye, right_eye)
        measurements[f"animal_{i+1}"] = {
            "class": pred["class_name"],
            "eye_distance_pixels": round(dist, 2) if dist is not None else None
        }
        
        if right_eye:
            valid_right_eyes.append((i, right_eye))
            
    # 2. 測量兩隻動物(以上)的右眼距離
    inter_animal_distance = None
    
    # 題目要求測量兩隻動物同側眼睛的距離，我們取最前面兩隻被偵測到的物體來計算
    if len(valid_right_eyes) >= 2:
        # 取出前兩隻動物的右眼座標
        pt1 = valid_right_eyes[0][1]
        pt2 = valid_right_eyes[1][1]
        
        dist = calculate_euclidean_distance(pt1, pt2)
        inter_animal_distance = round(dist, 2)
        
    measurements["inter_animal_right_eye_distance_pixels"] = inter_animal_distance
    return measurements
