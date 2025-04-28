# 使用官方 Python 映像
FROM python:3.13-slim

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt 並安裝依賴
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 複製所有應用程式檔案
COPY . .

# 啟動應用程式
CMD ["python", "app.py"]