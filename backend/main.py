"""RAG 多 Agent 后端服务。

架构（Supervisor 编排，单文件、零额外框架依赖）：

    Planner ──► Retriever ──► ToolRunner ──► Writer ──► Critic
       │          多路并行检索      按需         │          │
       │                                        └── 未通过 ──┘
       ├── 每个 Agent 只拿到自己需要的上下文，互不污染
       └── 闲聊 / 通用问题短路：need_knowledge=false 且无工具时直接作答，
           不进检索与评审（省掉 Writer-Critic 的多次 LLM 往返）

相比单 Agent（ReAct 边想边调工具）的差异：
1. 检索从「一个工具」升级为「独立 Agent」——先改写查询，再多路并行检索、去重合并；
2. 生成与校验分离——Writer 写、Critic 查，带 [n] 引用溯源，未通过自动回炉重写；
3. 每步落盘 checkpoint，进程中断可用同一 run_id 续跑，不必从头重来。

SSE 事件与前端既有解析逻辑（api.ts / api.js）保持兼容：
- stream_content : 最终答案，前端追加到气泡
- tool_call      : "正在调用工具：xxx"
- tool_result    : 工具返回结果
- final_answer   : "done"，前端据此 break
- agent_step     : 新增，前端未识别该 stage 会自动忽略，无需改前端

启动前请保证 Ollama 已拉取以下模型：
- all-minilm:l6-v2（文本向量化）
- qwen3.5:9b（对话生成）
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import operator
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sse_starlette.sse import EventSourceResponse

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-minilm:l6-v2")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.5:9b")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")
RUN_DIR = Path(os.getenv("AGENT_RUN_DIR", "./agent_runs"))

RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "3"))  # 每个改写查询各召回几条
MAX_QUERIES = int(os.getenv("MAX_QUERIES", "3"))        # 最多改写几条查询
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "6"))          # 去重合并后喂给 Writer 的上限
MAX_REVISION = int(os.getenv("MAX_REVISION", "2"))      # Critic 不通过时最多重写几次
SESSION_TURNS = 6                                       # 记忆保留的对话轮数

app = FastAPI(title="RAG Multi-Agent API")

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

# ---------------------------------------------------------------------------
# 工具：不再是 ReAct 循环，而是由 Planner 选中、ToolRunner 直接调用
# ---------------------------------------------------------------------------
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def calculate(expression: str) -> str:
    """计算四则运算表达式，例如 "123+45*8"。"""

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.n
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError("不支持的表达式")

    try:
        result = _eval(ast.parse(expression, mode="eval").body)
        return f"{expression} = {result}"
    except Exception as exc:
        return f"计算失败：{exc}"


def get_time() -> str:
    """返回当前系统时间。"""
    return datetime.now().strftime("当前时间：%Y-%m-%d %H:%M:%S")


TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "calculate": {"fn": calculate, "arg": "expression", "desc": "四则运算"},
    "get_time": {"fn": get_time, "arg": None, "desc": "获取当前系统时间"},
}

# ---------------------------------------------------------------------------
# LLM 工厂 & 通用小工具
# ---------------------------------------------------------------------------


def _make_llm(json_mode: bool = False) -> ChatOllama:
    """json_mode=True 时强制模型输出 JSON，用于规划 / 评审这类结构化节点。"""
    return ChatOllama(model=CHAT_MODEL, temperature=0.1, format="json" if json_mode else None)


_llm = _make_llm()
_llm_json = _make_llm(json_mode=True)


def _loads_json(text: str) -> Dict[str, Any]:
    """从模型输出里尽量稳地抠出一个 JSON 对象。"""
    raw = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if -1 < start < end:
        try:
            data = json.loads(raw[start : end + 1])
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _content_of(response) -> str:
    return str(getattr(response, "content", response) or "")


def _sse_payload(stage: str, content: str) -> dict:
    data = json.dumps({"stage": stage, "content": content}, ensure_ascii=False)
    return {"event": "message", "data": data}


def _parse_pdf(file: UploadFile) -> List[Document]:
    reader = PdfReader(file.file)
    texts = [page.extract_text() for page in reader.pages]
    content = "\n".join(text for text in texts if text)
    return text_splitter.create_documents([content])


# ---------------------------------------------------------------------------
# 会话记忆（前端目前不传 session_id，默认落到 default 会话）
# ---------------------------------------------------------------------------
_SESSIONS: Dict[str, List[Dict[str, str]]] = {}


def _remember(session_id: str, question: str, answer: str) -> None:
    bucket = _SESSIONS.setdefault(session_id, [])
    bucket.append({"role": "user", "content": question})
    bucket.append({"role": "assistant", "content": answer})
    del bucket[: max(0, len(bucket) - SESSION_TURNS * 2)]


def _history_text(session_id: str) -> str:
    bucket = _SESSIONS.get(session_id) or []
    if not bucket:
        return "（无历史对话，这是本轮的第一个问题）"
    lines = []
    for item in bucket:
        speaker = "用户" if item["role"] == "user" else "助手"
        lines.append(f"{speaker}：{item['content']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checkpoint：每步落盘，支持中断续跑
# ---------------------------------------------------------------------------
def _run_path(run_id: str) -> Path:
    return RUN_DIR / f"{run_id}.json"


def _save_run(run_id: str, query: str, steps: Dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "query": query,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "steps": steps,
    }
    _run_path(run_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_run(run_id: str) -> Dict[str, Any]:
    path = _run_path(run_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


# ===========================================================================
# Agent 1｜Planner：拆解问题、决定用哪些能力、改写检索词
# ===========================================================================
PLANNER_PROMPT = """你是 RAG 系统的规划 Agent。判断回答用户问题需要哪些能力，并产出检索用的查询词。

可用能力：
- knowledge：检索本地知识库（用户上传的文档）
- calculate：四则运算
- get_time：获取当前时间

规则：
1. 只输出 JSON，不要任何解释文字。
2. queries 是「改写后的检索查询」，要求：
   - 补全代词，把「它 / 这个 / 那个」还原成具体名词；
   - 从不同角度拆成 2~3 条，彼此不重复；
   - 每条是完整短句或名词短语，适合同向量检索匹配。
3. 如果只是打招呼、闲聊，或问的是与上传文档无关的通用常识，need_knowledge 设为 false，queries 设为 []。
4. 只有明确的算数才加 calculate，只有明确问「现在几点 / 今天几号」才加 get_time，宁可少加。
5. 上一轮助手回答中已经出现过的检索词不要重复生成。

只输出以下格式的 JSON：
{"need_knowledge": true, "queries": ["..."], "need_tools": [], "reason": "一句话说明判断依据"}"""


async def planner_agent(user_query: str, history: str) -> Dict[str, Any]:
    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"【历史对话】\n{history}\n\n【当前问题】\n{user_query}"),
    ]
    raw = _content_of(await _llm_json.ainvoke(messages))
    data = _loads_json(raw)

    queries: List[str] = []
    for item in data.get("queries") or []:
        text = str(item).strip()
        if text and text not in queries:
            queries.append(text)
    queries = queries[:MAX_QUERIES]

    tools = [name for name in (data.get("need_tools") or []) if name in TOOL_REGISTRY]
    need_knowledge = bool(data.get("need_knowledge", True))
    if need_knowledge and not queries:
        # 模型没给出改写结果时兜底用原问题，保证检索链路不断
        queries = [user_query]

    return {
        "need_knowledge": need_knowledge,
        "queries": queries,
        "need_tools": tools,
        "reason": str(data.get("reason", "")).strip(),
    }


# ===========================================================================
# 短路分支｜闲聊 / 通用问题：不进检索与评审，一次 LLM 直接作答
# Planner 已判定 need_knowledge=false 且无工具可用，此时 Writer 的
# 「只能用给定资料、否则写资料中未提及」约束会把「你好」逼成拒答，
# 而 Critic 面对空资料也无事可做。所以这里绕开整条流水线。
# ===========================================================================
SMALLTALK_PROMPT = """你是问答助手。本轮问题无需检索知识库，也没有可用工具。

要求：
1. 直接用你自己的语言能力和常识回答，禁止出现「资料」「知识库」「未提及」这类字眼。
2. 如果是打招呼或闲聊，友好简短地回应，并顺带一句话说明你还能做什么
   （基于用户上传的文档回答问题、做四则运算、报当前时间）。
3. 如果是与文档无关的通用常识问题，直接给出答案，控制在 200 字以内。
4. 用中文回答，直接输出正文，不要复述本提示、不要写过程性说明。"""


async def smalltalk_agent(user_query: str, history: str) -> str:
    messages = [
        SystemMessage(content=SMALLTALK_PROMPT),
        HumanMessage(content=f"【历史对话】\n{history}\n\n【当前问题】\n{user_query}"),
    ]
    return _content_of(await _llm.ainvoke(messages)).strip()


# ===========================================================================
# Agent 2｜Retriever：多路并行检索 + 去重 + 编号（独立于生成）
# ===========================================================================
async def _retrieve_one(query: str, k: int) -> List[Document]:
    # Chroma 的相似度检索是同步阻塞调用，丢进线程池才能真正并行
    return await asyncio.to_thread(vector_store.similarity_search, query, k=k)


async def retriever_agent(queries: List[str], k: int = RETRIEVE_TOP_K) -> List[Dict[str, Any]]:
    if not queries:
        return []
    results = await asyncio.gather(
        *(_retrieve_one(query, k) for query in queries), return_exceptions=True
    )

    merged: List[Dict[str, Any]] = []
    seen: set = set()
    for query, docs in zip(queries, results):
        if isinstance(docs, BaseException):
            continue
        for doc in docs:
            text = (doc.page_content or "").strip()
            if not text:
                continue
            digest = hashlib.md5(text.encode("utf-8")).hexdigest()
            if digest in seen:  # 同一段内容被多个查询命中时只保留一次
                continue
            seen.add(digest)
            merged.append({"content": text, "metadata": doc.metadata or {}, "matched_query": query})

    return merged[:MAX_CHUNKS]


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "（知识库为空，或本次未检索到相关内容）"
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        source = chunk["metadata"].get("source") or chunk["metadata"].get("title") or "本地知识库"
        blocks.append(f"[{index}]（来源：{source}）\n{chunk['content']}")
    return "\n\n".join(blocks)


def _citation_check(answer: str, chunk_count: int) -> List[str]:
    """不依赖模型的硬校验：引用编号是否越界、有资料却零引用。

    chunk_count 为 0 表示本轮没有任何资料可供引用，此时不存在引用契约：
    答案里的 [n] 只是模型沿用了历史对话里的编号，属于无意义的残留，
    不构成事实性问题。直接放行，否则会把纯工具轮次误判成「引用越界」，
    拖进永远改不好的重写循环。
    """
    if chunk_count <= 0:
        return []
    cited = {int(x) for x in re.findall(r"\[(\d+)\]", answer)}
    issues = []
    invalid = sorted(i for i in cited if i < 1 or i > chunk_count)
    if invalid:
        issues.append(f"引用了不存在的资料编号：{['[%d]' % i for i in invalid]}")
    if not cited:
        issues.append("答案未标注任何 [n] 引用")
    return issues


# ===========================================================================
# Agent 3｜ToolRunner：只执行 Planner 选中的工具，不即兴发挥
# ===========================================================================
async def tool_runner_agent(plan: Dict[str, Any], user_query: str) -> List[Dict[str, str]]:
    outputs: List[Dict[str, str]] = []
    for name in plan.get("need_tools") or []:
        spec = TOOL_REGISTRY.get(name)
        if not spec:
            continue
        try:
            if spec["arg"] is None:
                arg_text, result = "-", await asyncio.to_thread(spec["fn"])
            else:
                arg_text = _tool_argument(name, user_query)
                result = await asyncio.to_thread(spec["fn"], **{spec["arg"]: arg_text})
        except Exception as exc:
            result = f"工具执行异常：{exc}"
            arg_text = user_query
        outputs.append({"name": name, "arg": str(arg_text), "result": str(result)})
    return outputs


def _tool_argument(name: str, user_query: str) -> str:
    """从用户问题里抽出工具入参；抽不到时退化为原句（calculate 会自行报错）。"""
    if name == "calculate":
        expr = re.sub(r"[^0-9+\-*/()\.\s]", "", user_query).strip()
        return expr or user_query
    return user_query


# ===========================================================================
# Agent 4｜Writer：只依据给定资料撰写，强制带 [n] 引用
# ===========================================================================
WRITER_PROMPT = """你是撰写 Agent，负责基于给定资料回答用户问题。

硬性要求：
1. 只使用【资料】和【工具结果】里的信息，禁止使用你自己的先验知识编造。
2. 每处引用资料的观点后面必须紧跟编号，如 [1]、[2]，编号必须来自给定的资料编号，不得自造。
3. 资料里没有的信息，明确写「资料中未提及」，不要猜、不要圆。
4. 用中文回答，先给结论再给依据，分点时可加小标题，控制在 300 字以内。
5. 直接输出答案正文，不要复述本提示、不要写「根据以上要求」这类过程性说明。"""


async def writer_agent(
    user_query: str,
    chunks: List[Dict[str, Any]],
    tool_outputs: List[Dict[str, str]],
    history: str,
    feedback: str = "",
) -> str:
    tool_text = "\n".join(f"- {t['name']}({t['arg']}) => {t['result']}" for t in tool_outputs) or "（本次未调用工具）"
    repair = f"\n\n【上一稿被驳回的原因，请针对性修正】\n{feedback}" if feedback else ""
    messages = [
        SystemMessage(content=WRITER_PROMPT),
        HumanMessage(
            content=(
                f"【历史对话】\n{history}\n\n"
                f"【资料】\n{_format_context(chunks)}\n\n"
                f"【工具结果】\n{tool_text}\n\n"
                f"【用户问题】\n{user_query}{repair}"
            )
        ),
    ]
    return _content_of(await _llm.ainvoke(messages)).strip()


# ===========================================================================
# Agent 5｜Critic：生成后校验，不过关就打回重写
# ===========================================================================
CRITIC_PROMPT = """你是评审 Agent，负责在答案返回给用户前做事实性校验。

逐条检查：
1. 答案里每个 [n] 编号，是否都能在【资料】中找到对应编号，且那条资料确实支撑该说法？
2. 有没有不带引用、却陈述成事实的断言（幻觉）？
3. 有没有和【资料】直接矛盾的地方？
4. 用户的问题是否被正面回答（而不是绕开）？

严格输出以下格式的 JSON：
{"passed": true 或 false, "issues": ["问题描述"], "fixed_answer": "passed 为 false 时给出修正后的完整答案，否则为空字符串"}"""


async def critic_agent(
    user_query: str,
    chunks: List[Dict[str, Any]],
    draft: str,
) -> Dict[str, Any]:
    messages = [
        SystemMessage(content=CRITIC_PROMPT),
        HumanMessage(
            content=(
                f"【资料】\n{_format_context(chunks)}\n\n"
                f"【用户问题】\n{user_query}\n\n"
                f"【待审答案】\n{draft}"
            )
        ),
    ]
    data = _loads_json(_content_of(await _llm_json.ainvoke(messages)))

    issues = [str(i).strip() for i in (data.get("issues") or []) if str(i).strip()]
    return {
        "passed": bool(data.get("passed", False)),
        "issues": issues,
        "fixed_answer": str(data.get("fixed_answer") or "").strip(),
    }


# ===========================================================================
# Orchestrator：Supervisor 编排（含 checkpoint 续跑）
# ===========================================================================
async def _emit_answer(answer: str):
    """把已校验的答案切块下发，保留前端逐字追加的观感。"""
    for index in range(0, len(answer), 24):
        yield _sse_payload("stream_content", answer[index : index + 24])
        await asyncio.sleep(0.01)


async def _agent_loop(
    user_query: str,
    max_loop: int = MAX_REVISION,
    session_id: str = "default",
    run_id: str | None = None,
    resume: bool = False,
):
    run_id = run_id or uuid.uuid4().hex
    # 调用方可以放宽重写次数，但不超过 MAX_REVISION 配置的上限
    revision_limit = max(1, min(max_loop, MAX_REVISION))
 
    saved = _load_run(run_id) if resume else {}
    steps: Dict[str, Any] = saved.get("steps") or {} if saved.get("query") == user_query else {}
    history = _history_text(session_id)

    yield _sse_payload("agent_step", f"Supervisor 接管｜run_id={run_id}")

    # --- ① Planner ---
    plan = steps.get("planner")
    if plan:
        yield _sse_payload("agent_step", "Planner：命中 checkpoint，跳过规划")
    else:
        yield _sse_payload("agent_step", "Planner：拆解问题、改写检索词…")
        plan = await planner_agent(user_query, history)
        steps["planner"] = plan
        _save_run(run_id, user_query, steps)
    queries = plan.get("queries") or []
    yield _sse_payload(
        "agent_step",
        f"Planner 结论：{plan.get('reason') or '—'}｜检索词：{'；'.join(queries) or '无需检索'}",
    )

    # --- 短路：闲聊 / 与文档无关的通用问题，不进 Writer-Critic 流水线 ---
    if not plan.get("need_knowledge") and not plan.get("need_tools"):
        answer = str(steps.get("smalltalk") or "")
        if answer:
            yield _sse_payload("agent_step", "闲聊/通用问题：命中 checkpoint，跳过生成")
        else:
            yield _sse_payload("agent_step", "闲聊/通用问题：跳过检索与评审，直接作答…")
            answer = await smalltalk_agent(user_query, history)
            steps["smalltalk"] = answer
            _save_run(run_id, user_query, steps)
        if not answer:
            answer = "抱歉，本次没有生成有效回答。请换一种问法，或先上传相关文档。"
        async for payload in _emit_answer(answer):
            yield payload
        _remember(session_id, user_query, answer)
        yield _sse_payload("final_answer", "done")
        return

    # --- ② Retriever ---
    chunks = steps.get("retriever")
    if chunks is None:
        use_queries = queries if plan.get("need_knowledge") else []
        yield _sse_payload("tool_call", f"Retriever：并行检索 {len(use_queries)} 路查询")
        chunks = await retriever_agent(use_queries)
        yield _sse_payload("tool_result", f"召回去重后得到 {len(chunks)} 个片段")
        steps["retriever"] = chunks
        _save_run(run_id, user_query, steps)
    else:
        yield _sse_payload("tool_result", f"Retriever：命中 checkpoint，复用 {len(chunks)} 个片段")

    # --- ③ ToolRunner ---
    tool_outputs = steps.get("tools")
    if tool_outputs is None:
        tool_outputs = await tool_runner_agent(plan, user_query)
        for item in tool_outputs:
            yield _sse_payload("tool_call", f"正在调用工具：{item['name']}")
            yield _sse_payload("tool_result", item["result"])
        steps["tools"] = tool_outputs
        _save_run(run_id, user_query, steps)
    else:
        for item in tool_outputs:
            yield _sse_payload("tool_result", f"{item['name']}（复用 checkpoint）：{item['result']}")

    # --- ④ Writer ⇄ Critic 回炉循环 ---
    saved_writer = steps.get("writer") or {}
    saved_critic = steps.get("critic") or {}
    draft = str(saved_writer.get("draft") or "")
    verdict: Dict[str, Any] = {
        "passed": bool(saved_critic.get("passed")),
        "issues": [],
        "fixed_answer": "",
    }

    if draft and saved_critic.get("passed"):
        yield _sse_payload("agent_step", "Writer / Critic：命中 checkpoint，跳过生成与评审")
    else:
        feedback = ""
        start_attempt = int(saved_writer.get("attempt") or 0) + 1
        for attempt in range(start_attempt, revision_limit + 2):
            yield _sse_payload("agent_step", f"Writer：撰写第 {attempt} 稿…")
            draft = await writer_agent(user_query, chunks, tool_outputs, history, feedback)

            # 先跑零成本的硬校验（纯正则，不含模型调用）
            hard_issues = _citation_check(draft, len(chunks))
            if hard_issues:
                # 有硬伤就必定回炉，这一次模型评审的结论用不上，直接省掉
                yield _sse_payload(
                    "agent_step",
                    f"Critic：硬校验发现 {len(hard_issues)} 处问题，跳过模型评审",
                )
                verdict = {"passed": False, "issues": [], "fixed_answer": ""}
            elif not chunks:
                # 本轮没有资料可供核对，Critic 的检查项都无事可做（纯工具轮次），直接放行
                yield _sse_payload("agent_step", "Critic：本轮无资料可核对，跳过模型评审")
                verdict = {"passed": True, "issues": [], "fixed_answer": ""}
            else:
                yield _sse_payload("agent_step", "Critic：校验引用编号与事实一致性…")
                verdict = await critic_agent(user_query, chunks, draft)

            issues = hard_issues + verdict["issues"]

            steps["writer"] = {"draft": draft, "attempt": attempt}
            steps["critic"] = {**verdict, "hard_issues": hard_issues}
            _save_run(run_id, user_query, steps)

            if verdict["passed"] and not issues:
                yield _sse_payload("agent_step", "Critic：校验通过")
                break
            if attempt > revision_limit:
                yield _sse_payload("agent_step", "Critic：已达最大重写次数，输出当前最优稿")
                break
            feedback = "；".join(issues)[:600]
            yield _sse_payload("agent_step", f"Critic：未通过 → {feedback}，回炉重写")

    final = draft.strip()
    if not verdict.get("passed") and verdict.get("fixed_answer"):
        final = verdict["fixed_answer"]
    if not final:
        final = "抱歉，本次没有生成有效回答。请换一种问法，或先上传相关文档。"

    async for payload in _emit_answer(final):
        yield payload

    _remember(session_id, user_query, final)
    yield _sse_payload("final_answer", "done")


# ===========================================================================
# FastAPI 路由
# ===========================================================================
@app.get("/")
async def health_check():
    return {"service": "rag-multi-agent-api", "status": "ok"}


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
    _SESSIONS.clear()
    return {"code": 0, "msg": "知识库已清空"}


@app.get("/agent_runs/{run_id}")
async def get_agent_run(run_id: str):
    """查看某次问答的完整 Agent 执行过程（排障 / 演示用）。"""
    record = _load_run(run_id)
    if not record:
        return {"code": 1, "msg": "未找到该 run_id 的执行记录"}
    return {"code": 0, **record}


@app.get("/agent_stream")
async def agent_stream(
    user_query: str,
    max_loop: int = MAX_REVISION,
    session_id: str = "default",
    run_id: str | None = None,
    resume: bool = False,
):
    # sep="\n"：显式使用 LF 作为 SSE 事件行分隔符，
    # 避免 sse-starlette 默认的 \r\n 与前端按 "\n\n" 切块的解析逻辑不匹配。
    stream = _agent_loop(user_query, max_loop, session_id, run_id, resume)
    return EventSourceResponse(stream, sep="\n")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
