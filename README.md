# RAG Full Project

基于文档检索（RAG）的智能问答示例项目，包含后端与两个前端实现。

## 项目结构

```
├── backend/       # FastAPI 后端：文档向量化入库、知识检索、Agent 流式问答
├── frontend/      # React + TypeScript 前端
└── frontend-vue/  # Vue 3 前端（与 React 版功能、界面一致）
```

## 快速开始

### 1. 启动后端

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

后端依赖本地 [Ollama](https://ollama.com/)，启动前需要拉取模型：

```bash
ollama pull all-minilm:l6-v2
ollama pull qwen3.5:9b
```

### 2. 启动前端

```bash
cd frontend        # React 版
npm install
npm run dev
```

```bash
cd frontend-vue    # Vue 版
npm install
npm run dev
```

两个前端均通过 Vite 代理将 `/api` 转发至 `http://127.0.0.1:8000`。

## 功能

- 上传 PDF / TXT 文档，自动分块并向量化入库
- 首次进入页面自动展示知识库内容
- 基于知识库的流式问答，支持数学计算与时间查询
