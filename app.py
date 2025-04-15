import os
from openai import OpenAI

from flask import Flask, request, abort

from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent
)

import json

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import TavilySearchResults
from langchain.schema import Document


app = Flask(__name__)

# 初始化 LINE Bot
channel_access_token = "Wz5hTGSPFsJlWWdhfkd2u9lNJ3laOWT/+HYdLgTYjSj3PBgboejmYEnyI41hF7FdJIlETGbIFi47wA5vUNCkuSGCws7UGNHT3a1lfY3RT8gxDZKM5gEQgFhi9k+UPxcyt7cnETvmluK+cFkqEPiJpQdB04t89/1O/w1cDnyilFU="
channel_secret = "bfc154e4be5c1d9a4b2656addf1e479d"
handler = WebhookHandler(channel_secret)
configuration = Configuration(access_token=channel_access_token)
# messaging_api = MessagingApi(configuration)

api_key = os.environ.get("OPENAI_API_KEY")
os.environ['TAVILY_API_KEY'] = "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"
model = init_chat_model("gpt-4o-mini", model_provider="openai")

file_path = os.path.join("PDF", "A survey on large language model (LLM) security and privacy.pdf")
loader = PyPDFLoader(file_path)
documents = loader.load()

# 文本分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200,
    add_start_index=True,
)
split_documents = text_splitter.split_documents(documents)

# 建立向量資料庫
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_documents, embeddings)
retriever = vectorstore.as_retriever()

# 初始化網路搜尋工具
web_search_tool = TavilySearchResults()

# 定義工具 
class WebSearch(BaseModel):
    """
    網路搜尋工具：若問題與大型語言模型(LLMs)在安全性與隱私領域無關，則用此工具搜尋解答。
    """
    query: str = Field(description="網路搜尋的問題")

class Vectorstore(BaseModel):
    """
    向量資料庫工具：若問題與大型語言模型(LLMs)在安全性與隱私領域有關，則用此工具搜尋解答。
    """
    query: str = Field(description="向量資料庫搜尋的問題")

# 1. 決策
instruction_route = """
你是專家，請根據使用者的問題決定應該用向量資料庫還是網路搜尋工具。
如果問題涉及大型語言模型(LLMs)在安全性與隱私領域，請選擇向量資料庫工具(vectorstore)；其他情況請選擇網路搜尋工具(web_search)。
"""
route_prompt = ChatPromptTemplate.from_messages([
    ("system", instruction_route),
    ("human", "{question}"),
])
llm_router = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm_router = llm_router.bind_tools(tools=[WebSearch, Vectorstore])
question_router = route_prompt | structured_llm_router

# 2. RAG 模式生成答案
instruction_rag = """
你是一位助手，請根據提供的文件內容回答使用者問題。
若答案無法從文件中取得，請回答「我不知道」，禁止虛構答案，務必確保答案準確。
"""
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", instruction_rag),
    ("system", "文件:\n\n{documents}"),
    ("human", "問題: {question}"),
])
llm_rag = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
rag_chain = rag_prompt | llm_rag | StrOutputParser()

# 3. 一般 LLM 問答
instruction_plain = """
你是一位知識豐富的助手，請根據你已有的知識準確回答問題，切勿虛構答案。
"""
plain_prompt = ChatPromptTemplate.from_messages([
    ("system", instruction_plain),
    ("human", "問題: {question}"),
])
llm_plain = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_chain = plain_prompt | llm_plain | StrOutputParser()

# 4. 檢查文件與問題的相關性
class GradeDocuments(BaseModel):
    """
    確認提取文章與問題是否有關(yes/no)
    """

    binary_score: str = Field(description="請問文章與問題是否相關。('yes' or 'no')")

instruction_grade = """
你是一個評分的人員，負責評估文件與使用者問題的關聯性。
如果文件包含與使用者問題相關的關鍵字或語意，則將其評為相關。
輸出 'yes' or 'no' 代表文件與問題的相關與否。
"""
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", instruction_grade),
    ("human", "文件:\n\n{document}\n\n使用者問題: {question}"),
])
llm_grader = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm_grader = llm_grader.with_structured_output(GradeDocuments)
retrieval_grader = grade_prompt | structured_llm_grader

def run(question: str) -> str:
    route_result = question_router.invoke({"question": question})
    tool = None
    if "tool_calls" in route_result.additional_kwargs and route_result.additional_kwargs["tool_calls"]:
        tool = route_result.additional_kwargs["tool_calls"][0]["function"]["name"]
    print("使用工具為:",tool)
    print("---------運行中------------")
    # 根據工具決策採取不同流程
    if tool == "vectorstore":
        # 從向量資料庫取得文件
        docs = retriever.invoke(question)
        # 過濾不相關文件
        filtered_docs = []
        for d in docs:
            score = retrieval_grader.invoke({
                "question": question,
                "document": d.page_content
            })
            if score.binary_score.strip().lower() == "yes":
                filtered_docs.append(d)
        # 如果有相關文件，就用 RAG 模式生成答案
        if filtered_docs:
            print(">>> 使用 RAG 模式回答問題")
            generation = rag_chain.invoke({
                "documents": filtered_docs,
                "question": question
            })
        else:
            print("向量資料庫無相關文件，改用網路搜尋")
            docs_web = web_search_tool.invoke({"query": question})
            web_results = [Document(page_content=d["content"]) for d in docs_web]
            
            generation = rag_chain.invoke({
                "documents": web_results,
                "question": question
            })
    elif tool == "web_search":
        docs_web = web_search_tool.invoke({"query": question})
        web_results = [Document(page_content=d["content"]) for d in docs_web]
        generation = rag_chain.invoke({
            "documents": web_results,
            "question": question
        })
    else:
        # 若決策不明，則直接以 LLM 回答
        generation = llm_chain.invoke({"question": question})
    
    return generation


@app.route("/callback", methods=["POST"])
def linebot():
    body = request.get_data(as_text=True)
    json_data = json.loads(body)
    print(json_data)

    try:
        signature = request.headers["X-Line-Signature"]
        handler.handle(body, signature)

        # 取得使用者輸入的訊息
        tk = json_data["events"][0]["replyToken"]
        msg = json_data["events"][0]["message"]["text"]
        # msg_type = json_data["events"][0]["message"]["type"]
        # print(msg_type)
        # 使用 LangChain 生成回應
        response = run(msg)

        # 取得回應內容並發送
        reply_msg = response.strip()
        text_message = TextMessage(text=reply_msg)
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=tk,
                    messages=[text_message]
                )
            )

        # reply_request = ReplyMessageRequest(
        #     reply_token=tk,
        #     messages=[text_message]
        # )
        # line_bot_api.reply_message(reply_request)
    except Exception as e:
        print(f"發生錯誤: {e}")

    return "OK"

if __name__ == "__main__":
    app.run()
