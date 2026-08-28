# FastAPI web框架核心组件
from fastapi import FastAPI, UploadFile, File
# 跨域中间件，解决前端浏览器跨域
from fastapi.middleware.cors import CORSMiddleware
# SSE服务端推送响应组件
from sse_starlette.sse import EventSourceResponse
# pydantic做参数模型校验
from pydantic import BaseModel
# 类型注解
from typing import Optional, List, Dict, Any
# Langchain Ollama封装：聊天大模型 + Embedding向量化
from langchain_ollama import OllamaEmbeddings, ChatOllama
# Chroma本地向量数据库
from langchain_chroma import Chroma
# 文档递归分割器，RAG分块
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Langchain文档对象
from langchain_core.documents import Document
# Langchain function call 工具装饰器、工具集合
from langchain_core.tools import tool
# 消息对象，构建大模型对话历史
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# PDF解析库
from pypdf import PdfReader
# json序列化，SSE推送报文
import json
# 时间工具，给Agent调用
from datetime import datetime
# 数学计算安全eval
import ast
import operator

# ====================== 1.初始化FastAPI应用 ======================
app = FastAPI(title="RAG‑Agent Function‑Call 后端")

# 跨域配置，开发环境允许前端Vue 5173跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====================== 2.Agent可用工具定义 @tool ======================
@tool
def calc_math(expression: str) -> str:
    """
    数学计算器工具，用于计算数学表达式
    参数 expression: 数学表达式字符串，例如 "123+45*8"
    """
    # 安全计算，限制运算符，防止恶意代码执行
    safe_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv
    }

    def _safe_eval(node):
        if isinstance(node, ast.Constant):
            return node.n
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in safe_ops:
                raise ValueError("不支持该运算符")
            left = _safe_eval(node.left)
            right = _safe_eval(node.right)
            return safe_ops[op_type](left, right)
        raise ValueError("表达式非法")

    try:
        tree = ast.parse(expression, mode="eval")
        res = _safe_eval(tree.body)
        return f"计算结果：{expression} = {res}"
    except Exception as e:
        return f"计算失败：{str(e)}"


@tool
def get_current_time() -> str:
    """
    获取当前系统时间工具，用户询问时间、日期时调用
    """
    now = datetime.now()
    return f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def rag_search_knowledge(query: str) -> str:
    """
    知识库检索工具，当问题需要查询上传文档里的内容时调用此工具
    参数 query：检索知识库的查询词
    """
    # 向量相似度检索top3
    docs = vector_store.similarity_search(query, k=3)
    print(f"[RAG检索] query={query}, 检索到文档数量：{len(docs)}")
    context = "\n".join([d.page_content for d in docs])
    if not context.strip():
        return "知识库未检索到相关资料"
    return f"【知识库检索结果】\n{context}"


# 工具映射字典：工具名字映射到实际函数，大模型输出工具名就执行对应函数
tool_map = {
    "calc_math": calc_math,
    "get_current_time": get_current_time,
    "rag_search_knowledge": rag_search_knowledge
}

# ====================== 3.向量库配置 RAG知识库 ======================
PERSIST_DIR = "./chroma_db"
# Ollama本地Embedding模型，确保已经 ollama pull all-minilm
embedding = OllamaEmbeddings(model="all-minilm:l6-v2")
# 实例化持久化向量库
vector_store = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding
)

# 文档分割器配置
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", ".", " ", ""]
)

# 初始化大模型，开启bind_tools绑定工具，实现function‑call
llm = ChatOllama(
    model="minimax-m3:cloud",
    temperature=0.1,
).bind_tools(
    # 重点：直接传函数对象，不是字符串
    [
        calc_math,
        get_current_time,
        rag_search_knowledge
    ]
)

# ====================== 4.文档解析工具函数 ======================
def load_pdf_to_docs(file: UploadFile) -> List[Document]:
    """解析PDF,切分成Document文档块"""
    reader = PdfReader(file.file)
    full_text = ""
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            full_text += txt + "\n"
    docs = text_splitter.create_documents([full_text])
    return docs

# ====================== 5.上传、清空向量库接口 ======================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传PDF/TXT文档入库接口"""
    if file.filename.endswith(".pdf"):
        docs = load_pdf_to_docs(file)
    elif file.filename.endswith(".txt"):
        content = await file.read()
        text = content.decode("utf‑8", errors="ignore")
        docs = text_splitter.create_documents([text])
    else:
        return {"code": -1, "msg": "仅支持pdf/txt"}
    vector_store.add_documents(docs)
    return {"code": 0, "chunk_count": len(docs), "msg": "文档入库成功"}


@app.get("/clear_db")
async def clear_vector_db():
    """清空向量数据库"""
    all_ids = vector_store.get()["ids"]
    if all_ids:
        vector_store.delete(ids=all_ids)
    return {"code": 0, "msg": "向量库已清空"}


# ====================== 6.Agent核心SSE生成器（Function‑Call循环） ======================
async def agent_functioncall_generator(user_query: str, max_loop: int = 3):
    """
    Agent主循环生成器,SSE不断yield推送给前端
    :param user_query 用户提问
    :param max_loop: 最大工具调用轮次，防止死循环
    """
    # 初始化消息列表，保存对话历史
    messages = [HumanMessage(content=user_query)]
    loop_count = 0

    # Agent循环：大模型判断是否调用工具，最多循环max_loop次
    while loop_count < max_loop:
        loop_count += 1
        # 调用大模型，得到输出（可能带工具调用指令，也可能直接返回答案）
        ai_response = await llm.ainvoke(messages)
        print(f"[Agent] loop={loop_count}, ai_response={ai_response}")
        # 如果大模型决定调用工具，ai_response.tool_calls不为空
        if ai_response.tool_calls:
            # 推送事件给前端：告诉前端，Agent准备调用工具
            yield json.dumps({
                "stage": "tool_call",
                "content": f"🔧准备调用工具：{ai_response.tool_calls[0]['name']}"
            }, ensure_ascii=False)

            # 将大模型的AI响应加入消息上下文
            messages.append(ai_response)

            # 遍历要调用的全部工具（本示例一次一个）
            for tool_call in ai_response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                # 执行对应的工具函数
                selected_tool = tool_map[tool_name]
                tool_result = selected_tool.invoke(tool_args)
                print(f"[工具执行] {tool_name}({tool_args}) => {tool_result}")
                # 推送工具执行结果到前端
                yield json.dumps({
                    "stage": "tool_result",
                    "content": f"📌工具返回结果：\n{tool_result}"
                }, ensure_ascii=False)

                # 将工具返回结果包装为ToolMessage，放回消息上下文给大模型
                messages.append(ToolMessage(content=tool_result, tool_call_id=tool_id))
        else:
            # ✅没有工具调用，直接输出最终答案，流式输出文本
            yield json.dumps({
                "stage": "stream_content",
                "content": ai_response.content
            }, ensure_ascii=False)
            break
    else:
        # 达到最大循环轮次，强制终止Agent
        yield json.dumps({
            "stage": "stream_content",
            "content": "\n达到最大工具调用轮次,Agent终止。"
        }, ensure_ascii=False)

    # 推送结束标记
    yield json.dumps({
        "stage": "final_answer",
        "content": "done"
    }, ensure_ascii=False)


@app.get("/agent_stream")
async def agent_sse_endpoint(user_query: str, max_loop: int = 3):
    """
    SSE对外接口,Vue前端EventSource访问
    GET /agent_stream?user_query=xxx&max_loop=3
    """
    generator = agent_functioncall_generator(user_query, max_loop)
    return EventSourceResponse(generator)


# ======================服务启动入口======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)