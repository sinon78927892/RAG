# 使用基底映像
FROM python:3.9-slim

# 設定容器內的工作目錄為 /app
WORKDIR /app

# 複製需求檔案到 /app
# COPY requirements.txt .

# 安裝需求
# RUN pip install -r requirements.txt

# 複製應用程式檔案到 /app
COPY . .

# 6. 指定容器啟動指令
CMD ["python", "app.py"]