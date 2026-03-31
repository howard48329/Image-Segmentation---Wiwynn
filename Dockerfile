# 推薦使用 Python 3.10 slim 版本，體積較小且相容性佳
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /workspace

# 安裝 OpenCV 等圖像處理套件所需的基本系統依賴
# 雖然使用了 opencv-python-headless，但部分底層 C 函式庫 (如 libgl1, libxcb1) 仍為必需
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 先複製 requirements.txt 以利用 Docker Cache 加速建置
COPY requirements.txt .

# 🔥 關鍵防呆：預先安裝極小體積的純 CPU 版 PyTorch！
# 在 Linux (Docker) 環境下，如果直接 `pip install ultralytics`，
# pip 預設會幫你把附帶幾 GB 的 NVIDIA CUDA 驅動的 Pytorch 抓下來，導致 Build 出來的 Image 巨大無比。
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 將整個專案的程式碼複製進映像檔
COPY . .

# 曝露 8000 port
EXPOSE 8000

# 預設啟動指令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
