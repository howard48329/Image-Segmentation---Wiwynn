import json
from pathlib import Path

# 指向下載好的 COCO 標註檔
ANN_FILE = Path("data/.cache/annotations/instances_val2017.json")

def explore_coco_format():
    if not ANN_FILE.exists():
        print(f"找不到標註檔: {ANN_FILE}")
        print("請先執行 python scripts/download_coco.py 下載標註檔！")
        return

    print("正在載入 JSON 標註檔 (這可能會需要幾秒鐘)...")
    with open(ANN_FILE, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    print("\n" + "="*50)
    print("1. COCO JSON 最上層包含的結構 (Keys):")
    print("="*50)
    # 通常會包含 ['info', 'licenses', 'images', 'annotations', 'categories']
    print(list(coco_data.keys()))
    
    print("\n" + "="*50)
    print("2. 'images' (圖片基本資訊) 陣列的首筆資料範例:")
    print("="*50)
    # 用於對應 image_id 與得知原始圖片長寬、檔名及下載網址
    if coco_data.get('images'):
        first_image = coco_data['images'][0]
        print(json.dumps(first_image, indent=4, ensure_ascii=False))
        
    print("\n" + "="*50)
    print("3. 'categories' (類別清單) 陣列的首筆資料範例:")
    print("="*50)
    # 用於查詢 category_id 對應的是什麼動物，以及父類別 (supercategory)
    if coco_data.get('categories'):
        first_category = coco_data['categories'][0]
        print(json.dumps(first_category, indent=4, ensure_ascii=False))

    print("\n" + "="*50)
    print("4. 'annotations' (標註資料與座標) 陣列的首筆資料範例:")
    print("="*50)
    # 用於取得特定圖片裡物件的 [x, y, 寬, 高] 邊界框以及像素級的分割多邊形 (segmentation)
    if coco_data.get('annotations'):
        first_annotation = coco_data['annotations'][0]
        display_annotation = first_annotation.copy()
        
        # segmentation (影像分割多邊形) 內的座標點陣列通常會超級長，為了讓終端機好印，我們先把它截斷示意
        if 'segmentation' in display_annotation and isinstance(display_annotation['segmentation'], list) and len(display_annotation['segmentation']) > 0:
            poly = display_annotation['segmentation'][0]
            if len(poly) > 6:
                display_annotation['segmentation'] = [poly[:6] + ["... (其餘座標點太長，為畫面美觀已省略)"]]
                
        print(json.dumps(display_annotation, indent=4, ensure_ascii=False))
        
    print("\n" + "="*50)
    print("📊 資料總論:")
    print("="*50)
    print(f"- 圖片總數 (Images)     : {len(coco_data.get('images', []))} 張")
    print(f"- 類別總數 (Categories) : {len(coco_data.get('categories', []))} 類")
    print(f"- 標註總數 (Annotations): {len(coco_data.get('annotations', []))} 個物件")

if __name__ == "__main__":
    explore_coco_format()
