import os

from langchain_community.tools import TavilySearchResults
from langchain.chat_models import init_chat_model
from langchain.schema import Document
from pydantic import BaseModel
from pydantic import Field

api_key=os.environ.get("OPENAI_API_KEY")

model = init_chat_model("gpt-4o-mini", model_provider="openai")
# output = model.invoke("Hello, world!")
# print(output.content)

os.environ['TAVILY_API_KEY'] = "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"
web_search_tool = TavilySearchResults()
docs = web_search_tool.invoke({"query": "金融海嘯是幾年發生"})
web_results = [Document(page_content=d["content"]) for d in docs]
print(docs)
print("========================")
print(web_results)

# docs 的資料結構：
# [
#     {
#         "url": "<搜尋結果的網址>",
#         "content": "<該頁面擷取的文字內容>"
#     },
#     ...
# ]

