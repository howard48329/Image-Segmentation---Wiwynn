# 🐾 Image Segmentation Metrology API

這是一個專為「影像分割與測量學 (Metrology)」設計的端到端 (End-to-End) 深度學習後端專案。
本專案採用 **雙引擎架構 (Top-Down Crop-to-Pose)**，能切割出影像中的人物或動物輪廓，並量測出特定特徵點（例如雙眼）的幾何像素距離。

---

## 📌 專案介紹
我實作了一個基於 RESTful API 的微服務，當使用者上傳包含「人或動物」的影像時，系統會自動執行以下流程：
1. **第一層推論**：自動標記出物體之 Mask 輪廓地圖與 Bounding Box，並分類。
2. **第二層推論 (Crop-to-Pose)**：將各個 BBox 裁切並加入環境填充(Margin Padding)後放大，針對特寫畫面進行高精度特徵點(Keypoints)捕捉，再透過反向映射(Global Mappping)換算回原圖座標。
3. **幾何量測**：透過尤拉公式 (Euclidean Distance) 精準計算出 **單一目標的雙眼間距** 以及 **跨目標之間的右眼間距像素**。
4. **報表與視覺化**：繪製遮罩、點位與幾何連線於原圖中，並將量測數據自動匯出至 CSV 檔案。

---

## 🛠️ 技術棧說明

* **後端 API 框架**: `FastAPI` (非同步、高效能) + `Uvicorn`
* **AI 深度學習引擎**: `Ultralytics (YOLOv8)` + `PyTorch (CPU-Optimized)`
* **影像處理與數學運算**: `OpenCV-Python-Headless`, `NumPy`
* **組態與環境變數管理**: `Pydantic-Settings` (.env 設定)
* **基礎設施與容器化部署**: `Docker`, `Docker Compose`

---

## 📦 步驟零：建立與自動準備測試資料集

專案預設需要包含多隻目標的影像來進行量測展示。如果沒有合適的測資圖片，專案內建了一支自動化輔助腳本。它會自動連線至 MS COCO Dataset 驗證集，並精準過濾下載「**至少包含兩個目標（人、貓、狗、羊、牛、馬）**」的高品質測試圖集。

請在您的本機終端機運行以下指令：
```bash
python scripts/download_coco.py
```
> 執行完畢後，測試圖片會自動準備於 `data/input/` 目錄下，供您隨後測試 API 介面使用！

---

## 🚀 快速啟動：Docker 部署指令 (推薦模式)

本專案已完全容器化。Dockerfile 只拉取純 CPU 版的 Torch Wheel，避免拉取高達數 GB 的無用 CUDA 驅動元件，讓映像檔保持極致輕量與快速部署。

1. **複製專案並準備環境變數**
   ```bash
   git clone https://github.com/howard48329/Image-Segmentation---Wiwynn
   cd Image-Segmentation---Wiwynn
   cp .env.example .env
   ```

2. **一鍵建置並啟動服務**
   ```bash
   docker compose up --build
   ```
   > `model_weights/` 資料夾內已有預設模型，若無則會自動下載，pose模型為經我自己finetune過之動物眼睛特徵點偵測模型。

---

## 💻 本地運行步驟 (無 Docker 依賴環境)

若不希望使用 Docker，本機需具備 Python 3.10+ 環境：

1. **建立虛擬環境與安裝依賴**：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows 使用者請執行 venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **複製設定檔**：
   ```bash
   cp .env.example .env
   ```
3. **啟動 FastAPI 開發伺服器**：
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## 📚 API 文件連結與使用方法

當伺服器啟動後，請用瀏覽器開啟：

👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

### 測試結果：
   * 網頁下方會立即回傳包含各個動物距離數據的 JSON。
   * 您可以直接到專案的 `data/output/` 資料夾中查看帶有測量連線與精準遮罩的視覺化產出圖。
   * 資料數據也會自動被記錄到 `data/sample.csv` 中。

---

## 🔐 測試帳號資訊

本專案主要用於展示軟體工程架構與 AI Model Fusion 技術，為了讓面試官與測試人員能最快速地體驗核心功能：
* **目前 API 端點設定為完全公開 (Public Access)**。
* **無須準備測試帳號、不需申請 API Key 或 JWT Token**，開啟 Swagger 介面即可直接使用。

*(註：若未來實際上線有資安防護需求，架構層非常容易透過 FastAPI Dependency 疊加 OAuth2 驗證保護)*

---

## 🧰 專案資料夾結構與輔助工具

```text
.
├── app/               # 後端實作 (main: 介面, model: 雙引擎核心, config: 設定檔)
│   └── utils/         # geometry: 幾何像素距離引擎 / data_proc: 視覺化引擎
├── data/              # 本機目錄映射 (input 測資圖/ output 結果圖 / sample.csv 報表)
├── model_weights/     # 本地 AI 模型緩存 (會自動被 .gitignore 排除)
├── scripts/           # 自動化與維運腳本
│   ├── download_coco.py    # 💰 若缺乏圖片，可跑此腳本自動去 COCO 抓包含人/動物的圖
│   └── download_weights.py # 網路權重拉取邏輯
├── docker-compose.yml # 服務編排腳本
└── Dockerfile         # 優化過的微服務封裝檔
```