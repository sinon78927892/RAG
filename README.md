# 基於 LangChain 建立 RAG LINE Chatbot

## 專案介紹

這是一個使用 LangChain 框架與 LangGraph 工具建立 RAG (Retrieval-Augmented Generation) 的 LINE 聊天機器人，實現從PDF資料庫及網路搜尋中檢索資訊並生成對話回應。
本專案的主要學習目標是掌握 RAG 流程的建構與 Docker 的使用，並將其與 LINE 聊天機器人進行整合。

## 專案目標

- **學習和實踐 RAG 技術**：通過實作 RAG 技術，了解如何將外部知識庫的資料與 GPT 模型結合，提升問答系統的準確性和效果。
- **學習 Docker 容器化**：將整個應用程式封裝為 Docker 容器，實現簡單的環境部署，並確保程式能夠在不同環境下順利運行。

## 主要功能

1. **系統流程**  
    - 使用者從 LINE 發送問題。

    - 系統根據問題內容，決定使用：

        - 向量資料庫檢索（若問題與文件相關）

        - 或即時網路搜尋（若問題與文件無關）

    - 檢索並過濾相關文件。

    - 由 LLM 根據文件產生回答。

    - 進行回答虛構與回應性評分。

    - 最終將回答回傳至 LINE 使用者。

2. **資料來源**  
   該系統使用的資料來源為論文《A Survey on Large Language Model (LLM) Security and Privacy》，系統能夠根據這篇論文的內容生成對應的回應。

3. **LINE 聊天機器人**  
   系統將上述功能集成到 LINE 聊天機器人中，使用者可以透過 LINE 提問，並獲得基於論文的回答。

4. **Docker 部署**：應用程式使用 Docker 容器化，簡化了部署過程，並保證在任何環境中都能穩定運行。

## 技術架構
*LangChain
*LangGraph
*Docker
*FAISS (向量資料庫)
*LINE Messaging API
*Flask

## 安裝與使用

### 1.建置 Docker 容器

```
docker build -t rag-line-chatbot .
```

### 2.運行 Docker 容器

```
docker run -d -p 5000:5000 rag-line-chatbot
```

### 3. 設定 LINE 聊天機器人


1. 前往 LINE Developer 註冊並創建一個新的 LINE Messaging API channel。

2. 取得您的 Channel Access Token 和 Channel Secret ，填入.evn 中對應的位置。

3. 安裝並啟動 ngrok，創建一個臨時的公開 URL：
```
ngrok http 5000
```
4. 將 URL 複製到 LINE Developer 裡的 Webhook URL。
```
https://your-ngrok-url.ngrok-free.app/callback
```

## 學習成果

在這個專案中，我學會了如何使用 LangChain 和 LangGraph 工具來建立 RAG 流程，並且將這個流程與 LINE 聊天機器人整合。通過這個專案，我深入理解了如何實現資料檢索和生成模型的結合，同時也學習到了如何使用 Docker 來部署和管理。

## 參考資料
- [利用 Langchain 實作系列 RAG 進階流程：Query Analysis & Self-reflection](https://edge.aif.tw/application-langchain-rag-advanced/)
- [LangChain] (https://www.langchain.com/)
