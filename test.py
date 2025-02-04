from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import TavilySearchResults
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph

# 加載 .env 檔案
load_dotenv()

# 使用環境變數
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ['TAVILY_API_KEY'] = "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"
file_path = R"C:\Users\sinon\Downloads\牙周病診治健康照護手冊.pdf"

# 第一步：加載並解析 PDF

# 使用 PyPDFLoader 加載 PDF 文件
loader = PyPDFLoader(file_path)
documents = loader.load()

# 分割文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # chunk size (characters)
    chunk_overlap=200,  # 區塊之間的重疊
    add_start_index=True,  # track index in original document
)
split_documents = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_documents, embeddings)
retriever = vectorstore.as_retriever()

web_search_tool = TavilySearchResults()


# 定義兩個工具的 DataModel
class web_search(BaseModel):
    """
    網路搜尋工具。若問題與牙周病治療或照護"無關"，請使用web_search工具搜尋解答。
    """
    query: str = Field(description="使用網路搜尋時輸入的問題")


class vectorstore(BaseModel):
    """
    紀錄關於牙周病治療或照護的向量資料庫工具。若問題與牙周病治療或照護有關，請使用此工具搜尋解答。
    """
    query: str = Field(description="搜尋向量資料庫時輸入的問題")


# Prompt Template
instruction = """
你是將使用者問題導向向量資料庫或網路搜尋的專家。
向量資料庫包含有關牙周病治療或照護文件。對於這些主題的問題，請使用向量資料庫工具。其他情況則使用網路搜尋工具。
"""
route_prompt = ChatPromptTemplate.from_messages([
    ("system", instruction),
    ("human", "{question}"),
])

# Route LLM with tools use
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm_router = llm.bind_tools(tools=[web_search, vectorstore])

# 使用 LCEL 語法建立 chain
question_router = route_prompt | structured_llm_router

# Prompt Template
instruction = """
你是一位負責處理使用者問題的助手，請利用提取出來的文件內容來回應問題。
若問題的答案無法從文件內取得，請直接回覆你不知道，禁止虛構答案。
注意：請確保答案的準確性。
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", instruction),
    ("system", "文件: \n\n {documents}"),
    ("human", "問題: {question}"),
])

# LLM & chain
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
rag_chain = prompt | llm | StrOutputParser()

# ===================================================================
# 測試 rag_chain 功能
# question = "牙周病與牙齦炎差在哪?"
# docs = retriever.invoke(question)
# generation = rag_chain.invoke({"documents": docs, "question": question})
# print(generation)
# ===================================================================

# Prompt Teamplate
instruction = """
你是一位負責處理使用者問題的助手，請利用你的知識來回應問題。
回應問題時請確保答案的準確性，勿虛構答案。
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", instruction),
    ("human", "問題: {question}"),
])

# LLM & chain
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_chain = prompt | llm | StrOutputParser()


# ===================================================================
# 測試 llm_chain 功能
# question = "Hi how are you?"
# generation = llm_chain.invoke({"question": question})
# print(generation)
# ===================================================================
class GradeDocuments(BaseModel):
    """
    確認提取文章與問題是否有關(yes/no)
    """

    binary_score: str = Field(description="請問文章與問題是否相關。('yes' or 'no')")


# Prompt Template
instruction = """
你是一個評分的人員，負責評估文件與使用者問題的關聯性。
如果文件包含與使用者問題相關的關鍵字或語意，則將其評為相關。
輸出 'yes' or 'no' 代表文件與問題的相關與否。
"""
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", instruction),
    ("human", "文件: \n\n {document} \n\n 使用者問題: {question}"),
])

# Grader LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 使用 LCEL 語法建立 chain
retrieval_grader = grade_prompt | structured_llm_grader


# ===================================================================
# 測試 grader 功能
# question = "牙周病與牙齦炎差在哪?"
# docs = retriever.invoke(question)
# doc_txt = docs[0].page_content
# response =  retrieval_grader.invoke({"question": question, "document": doc_txt})
# print(response)
# ===================================================================
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


# ===================================================================
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


# ===================================================================
#測試 grader 功能
# question = "牙周病與牙齦炎差在哪?"
# docs = retriever.invoke(question)
# generation = rag_chain.invoke({"documents": docs, "question": question})
# answer_grader.invoke({"question": question,"generation": generation})
# ===================================================================
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
        Retrieve documents related to the question.

        Args:
            state (dict):  The current state graph

        Returns:
            state (dict): New key added to state, documents, that contains list of related documents.
        """

        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
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
    documents = state["documents"] if state["documents"] else []

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = [Document(page_content=d["content"]) for d in docs]

    documents = documents + web_results

    return {"documents": documents, "question": question}


def retrieval_grade(state):
    """
    filter retrieved documents based on question.

    Args:
        state (dict):  The current state graph

    Returns:
        state (dict): New key added to state, documents, that contains list of related documents.
    """

    # Grade documents
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

    documents = state["documents"]
    question = state["question"]

    # Score each doc
    filtered_docs = []
    for d in documents:
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

    # Fallback to plain LLM or raise error if no decision
    if "tool_calls" not in source.additional_kwargs:
        print("  -ROUTE TO PLAIN LLM-")
        return "plain_answer"
    if len(source.additional_kwargs["tool_calls"]) == 0:
        raise "Router could not decide source"

    # Choose datasource
    datasource = source.additional_kwargs["tool_calls"][0]["function"]["name"]
    if datasource == 'web_search':
        print("  -ROUTE TO WEB SEARCH-")
        return "web_search"
    elif datasource == 'vectorstore':
        print("  -ROUTETO VECTORSTORE-")
        return "vectorstore"


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


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("retrieval_grade", retrieval_grade)  # retrieval grade
workflow.add_node("rag_generate", rag_generate)  # rag
workflow.add_node("plain_answer", plain_answer)  # llm

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
        "plain_answer": "plain_answer",
    },
)
workflow.add_edge("retrieve", "retrieval_grade")
workflow.add_edge("web_search", "retrieval_grade")
workflow.add_conditional_edges(
    "retrieval_grade",
    route_retrieval,
    {
        "web_search": "web_search",
        "rag_generate": "rag_generate",
    },
)
workflow.add_conditional_edges(
    "rag_generate",
    grade_rag_generation,
    {
        "not supported": "rag_generate",  # Hallucinations: re-generate
        "not useful":
        "web_search",  # Fails to answer question: fall-back to web-search
        "useful": END,
    },
)
workflow.add_edge("plain_answer", END)

# Compile
app = workflow.compile()

def run(question):
    inputs = {"question": question}
    for output in app.stream(inputs):
        print("\n")

    # Final generation
    if 'rag_generate' in output.keys():
        print(output['rag_generate']['generation'])
    elif 'plain_answer' in output.keys():
        print(output['plain_answer']['generation'])


def get_prompt():
    print("\n<------輸入'exit'來結束對話------->\n ")

    while True:
        message = input("請輸入問題 : ")
        if message.lower() == '000':
            print('Exiting...')
            break
        else:
            try:
                # Use the agent to process the prompt
                answer = run(message)
                print(answer)
            except Exception as e:
                print(e)

get_prompt()