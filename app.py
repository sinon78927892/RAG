import os
import json

from openai import OpenAI
from dotenv import load_dotenv
from flask import Flask, request, abort

from linebot.v3 import (
    WebhookHandler
)

from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage
)

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
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List

app = Flask(__name__)

# 初始化 LINE Bot
LINE_CHANNEL_ACCESS_TOKEN =  os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
handler = WebhookHandler(LINE_CHANNEL_SECRET)
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ['TAVILY_API_KEY'] = "TAVILY_API_KEY"

file_path = os.path.join("PDF", "A survey on large language model (LLM) security and privacy.pdf")
loader = PyPDFLoader(file_path)
documents = loader.load()

# 文本分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
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

# ==========================決策=======================
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

# 2. =================== 利用提取出來的文件內容來回應問題 ==========================
instruction_rag = """
你是一位負責處理使用者問題的助手，請利用提取出來的文件內容來回應問題。
若答案無法從文件中取得，請回答「我不知道」，禁止虛構答案，務必確保答案準確。
注意：請確保答案的準確性。
"""
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", instruction_rag),
    ("system", "文件:\n\n{documents}"),
    ("human", "問題: {question}"),
])
llm_rag = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
rag_chain = rag_prompt | llm_rag | StrOutputParser()

# ===================== 一般LLM問答 ===========================
instruction_plain = """
你是一位知識豐富的助手，請根據你已有的知識準確回答問題，切勿虛構答案。
"""
plain_prompt = ChatPromptTemplate.from_messages([
    ("system", instruction_plain),
    ("human", "問題: {question}"),
])
llm_plain = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_chain = plain_prompt | llm_plain | StrOutputParser()

# ===================== 檢索結果評分 =========================
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

# ==================== 虛構評分的人員(yes/no)判斷 =================================
class GradeHallucinations(BaseModel):
    """
    確認答案是否為虛構(yes/no)
    """

    binary_score: str = Field(description="答案是否由為虛構。('yes' or 'no')")


# Prompt Template
instruction = """
你是一個評分的人員，負責確認LLM的回應是否為虛構的。
以下會給你一個文件與相對應的LLM回應，請輸出 'yes' or 'no'做為判斷結果。
'Yes' 代表LLM的回答是虛構的，未基於文件內容 'No' 則代表LLM的回答並未虛構，而是基於文件內容得出。
"""
hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", instruction),
    ("human", "文件: \n\n {documents} \n\n LLM 回應: {generation}"),
])

# Grader LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# 使用 LCEL 語法建立 chain
hallucination_grader = hallucination_prompt | structured_llm_grader


# ====================== 回應評分的人員(yes/no) ==========================
class GradeAnswer(BaseModel):
    """
    確認答案是否可回應問題
    """

    binary_score: str = Field(description="答案是否回應問題。('yes' or 'no')")


# Prompt Template
instruction = """
你是一個評分的人員，負責確認答案是否回應了問題。
輸出 'yes' or 'no'。 'Yes' 代表答案確實回應了問題， 'No' 則代表答案並未回應問題。
"""
# Prompt
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", instruction),
    ("human", "使用者問題: \n\n {question} \n\n 答案: {generation}"),
])

# LLM with function call
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# 使用 LCEL 語法建立 chain
answer_grader = answer_prompt | structured_llm_grader


class GraphState(TypedDict):
    """
    State of graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question: str
    generation: str
    documents: List[str]

def retrieve(state):
        """
        根據使用者提出的問題，從內部檢索系統取得相關文件。

        Args:
            state (dict):  The current state graph

        Returns:
            state (dict): New key added to state, documents, that contains list of related documents.
        """
        print("---RETRIEVE---")
        question = state["question"]
        # 從向量資料庫取得文件
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question} 

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = [Document(page_content=d["content"]) for d in docs] # Document()用於儲存一段文字和相關元資料的類別

    documents = documents + web_results

    return {"documents": documents, "question": question}

def retrieval_grade(state):
    """
    檢查與問題的關聯性，只保留相關的文件。

    Args:
        state (dict):  The current state graph

    Returns:
        state (dict): New key added to state, documents, that contains list of related documents.
    """

    # Grade documents
    print("--- 文件相關性檢查 ---")

    documents = state["documents"]
    question = state["question"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        #評分員進行評分
        score = retrieval_grader.invoke({
            "question": question,
            "document": d.page_content
        })
        grade = score.binary_score
        if grade == "yes":
            print("  -GRADE: DOCUMENT RELEVANT-")
            filtered_docs.append(d)
        else:
            print("  -GRADE: DOCUMENT NOT RELEVANT-")
            continue
    return {"documents": filtered_docs, "question": question}

def rag_generate(state):
    """
    Generate answer using  vectorstore / web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    print("---GENERATE IN RAG MODE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({
        "documents": documents,
        "question": question
    })
    return {
        "documents": documents,
        "question": question,
        "generation": generation
    }


def plain_answer(state):
    """
    Generate answer using the LLM without vectorstore.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    print("---GENERATE PLAIN ANSWER---")
    question = state["question"]
    generation = llm_chain.invoke({"question": question})
    return {"question": question, "generation": generation}


### Edges ###
def route_question(state):
    """
    Route question to web search or RAG.
    Args:
        state (dict): The current graph state
    Returns:
        str: Next node to call
    """
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question}) 

    if source is None or not hasattr(source, "additional_kwargs"):
        raise Exception("question_router 回傳無效結果")

    # 如果 LLM 沒有給出任何工具建議，就直接請 LLM 回答
    if "tool_calls" not in source.additional_kwargs:
        print("  -ROUTE TO PLAIN LLM-")
        return "plain_answer"

    # LLM 判斷失敗，路由無法決定要用什麼資料來源
    if len(source.additional_kwargs["tool_calls"]) == 0:
        raise Exception("Router could not decide source")

    # Choose datasource
    datasource = source.additional_kwargs["tool_calls"][0]["function"]["name"]
    if datasource == 'web_search':
        print("  -ROUTE TO WEB SEARCH-")
        return "web_search"
    elif datasource == 'vectorstore':
        print("  -ROUTE TO VECTORSTORE-")
        return "vectorstore"
    else:
        print(f"  -UNKNOWN ROUTE: {datasource}-")
        return "plain_answer"


def route_retrieval(state):
    """
    Determines whether to generate an answer, or use websearch.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ROUTE RETRIEVAL---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        print(
            "  -DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, ROUTE TO WEB SEARCH-"
        )
        return "web_search"
    else:
        # We have relevant documents, so generate answer
        print("  -DECISION: GENERATE WITH RAG LLM-")
        return "rag_generate"


def grade_rag_generation(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({
        "documents": documents,
        "generation": generation
    })
    grade = score.binary_score

    # Check hallucination
    if grade == "no":
        print("  -DECISION: GENERATION IS GROUNDED IN DOCUMENTS-")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({
            "question": question,
            "generation": generation
        })
        grade = score.binary_score
        if grade == "yes":
            print("  -DECISION: GENERATION ADDRESSES QUESTION-")
            return "useful"
        else:
            print("  -DECISION: GENERATION DOES NOT ADDRESS QUESTION-")
            return "not useful"
    else:
        print("  -DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY-")
        return "not supported"


workflow = StateGraph(GraphState) #建立流程有哪些 key 與其型態
# Define the nodes
workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("retrieval_grade", retrieval_grade)  # retrieval grade
workflow.add_node("rag_generate", rag_generate)  # rag
workflow.add_node("plain_answer", plain_answer)  # llm

# Build graph
workflow.set_conditional_entry_point(
    route_question, #函式判斷該走哪一條路
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
        "plain_answer": "plain_answer",
    },
)
workflow.add_edge("retrieve", "retrieval_grade") # retrieve 下一步交給 retrieval_grade 檢查文件是否有用
workflow.add_edge("web_search", "retrieval_grade") # web_search 下一步交給 retrieval_grade
workflow.add_conditional_edges(
    "retrieval_grade", #起點
    route_retrieval, # 根據route_retrieval決定下一步要去哪
    {
        "web_search": "web_search",
        "rag_generate": "rag_generate",
    },
)
workflow.add_conditional_edges(
    "rag_generate",
    grade_rag_generation,
    {
        "not supported": "rag_generate",  # 若內容不支援問題（例如偏題）➜ 重新生成
        "not useful":
        "web_search",  # Fails to answer question: fall-back to web-search
        "useful": END,
    },
)
workflow.add_edge("plain_answer", END)

# Compile
lc_app = workflow.compile()

# LangChain 回應函式
def run(question):
    inputs = {"question": question}
    result_text = None
    for output in lc_app.stream(inputs):
        print("\n")

    # Final generation
    if 'rag_generate' in output:
        result_text = output['rag_generate']['generation']
    elif 'plain_answer' in output:
        result_text = output['plain_answer']['generation']

    return result_text


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
        response = run(msg) or "抱歉，我沒有找到合適的回答。"

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
    except Exception as e:
        print(f"發生錯誤: {e}")

    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)