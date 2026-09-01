"""RAG Agent 后端服务。

负责文档的向量化入库、知识检索，以及基于工具调用（Function Calling）的
流式问答。启动前需要保证 Ollama 已拉取以下模型：
- all-minilm:l6-v2（文本向量化）
- minimax-m3:cloud（对话生成）
"""

import os
from datetime import datetime
from typing import List

import ast
import json
import operator

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sse_starlette.sse import EventSourceResponse

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-minilm:l6-v2")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.5:9b")
PERSIST_DIR = "./chroma_db"

app = FastAPI(title="RAG Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)

vector_store = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", ".", " ", ""],
)


@tool
def calculate(expression: str) -> str:
    """计算四则运算表达式，例如 "123+45*8"。"""
    safe_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.n
        if isinstance(node, ast.BinOp) and type(node.op) in safe_ops:
            return safe_ops[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError("不支持的表达式")

    try:
        result = _eval(ast.parse(expression, mode="eval").body)
        return f"{expression} = {result}"
    except Exception as exc:
        return f"计算失败：{exc}"


@tool
def get_time() -> str:
    """返回当前系统时间。"""
    return datetime.now().strftime("当前时间：%Y-%m-%d %H:%M:%S")


@tool
def search_knowledge(query: str) -> str:
    """在知识库中检索与 query 相关的文档内容。"""
    docs = vector_store.similarity_search(query, k=3)
    if not docs:
        return "知识库中未检索到相关内容"
    context = "\n".join(doc.page_content for doc in docs)
    return f"知识库检索结果：\n{context}"


tools = [calculate, get_time, search_knowledge]
tool_by_name = {item.name: item for item in tools}

llm = ChatOllama(model=CHAT_MODEL, temperature=0.1).bind_tools(tools)


def _parse_pdf(file: UploadFile) -> List[Document]:
    reader = PdfReader(file.file)
    texts = [page.extract_text() for page in reader.pages]
    content = "\n".join(text for text in texts if text)
    return text_splitter.create_documents([content])


def _sse_payload(stage: str, content: str) -> dict:
    data = json.dumps({"stage": stage, "content": content}, ensure_ascii=False)
    return {"event": "message", "data": data}


@app.get("/")
async def health_check():
    return {"service": "rag-agent-api", "status": "ok"}


@app.get("/documents")
async def list_documents():
    """返回知识库中的全部文档片段，供前端初始化展示。"""
    data = vector_store.get(include=["documents"])
    documents = [
        {"id": doc_id, "content": content}
        for doc_id, content in zip(data["ids"], data["documents"])
    ]
    return {"code": 0, "total": len(documents), "documents": documents}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if file.filename.lower().endswith(".pdf"):
        docs = _parse_pdf(file)
    elif file.filename.lower().endswith(".txt"):
        content = (await file.read()).decode("utf-8", errors="ignore")
        docs = text_splitter.create_documents([content])
    else:
        return {"code": 1, "msg": "仅支持 PDF / TXT 文件"}

    vector_store.add_documents(docs)
    return {"code": 0, "msg": "文档入库成功", "chunk_count": len(docs)}


@app.get("/clear_db")
async def clear_knowledge_base():
    ids = vector_store.get()["ids"]
    if ids:
        vector_store.delete(ids=ids)
    return {"code": 0, "msg": "知识库已清空"}


async def _agent_loop(user_query: str, max_loop: int = 3):
    messages = [HumanMessage(content=user_query)]
    loop = 0

    while loop < max_loop:
        loop += 1
        response = await llm.ainvoke(messages)

        if not response.tool_calls:
            yield _sse_payload("stream_content", str(response.content))
            break

        messages.append(response)
        for call in response.tool_calls:
            yield _sse_payload("tool_call", f"正在调用工具：{call['name']}")
            tool_fn = tool_by_name.get(call["name"])
            if tool_fn is None:
                result = f"未知工具：{call['name']}"
            else:
                result = tool_fn.invoke(call["args"])
            yield _sse_payload("tool_result", str(result))
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    else:
        yield _sse_payload("stream_content", "已超过最大工具调用轮次，本次对话结束。")

    yield _sse_payload("final_answer", "done")


@app.get("/agent_stream")
async def agent_stream(user_query: str, max_loop: int = 3):
    # sep="\n"：显式使用 LF 作为 SSE 事件行分隔符，
    # 避免 sse-starlette 默认的 \r\n 与前端按 "\n\n" 切块的解析逻辑不匹配。
    return EventSourceResponse(_agent_loop(user_query, max_loop), sep="\n")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
