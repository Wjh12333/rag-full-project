# RAG Full Project

这是一个包含前后端的 RAG（Retrieval-Augmented Generation）示例项目。

项目结构概览：

- `backend/`：后端（Python），包含 FastAPI 或类似服务，负责文档索引与检索。
- `frontend/`：React + Vite 前端（TypeScript）。
- `frontend-vue/`：Vue + Vite 前端示例（可选）。

快速启动（开发环境）：

1. 启动后端（在 `backend` 目录）：

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

2. 启动 React 前端（在 `frontend` 目录）：

```bash
cd frontend
npm install
npm run dev
```

Git 注意事项：

- 已在仓库根添加 `.gitignore`，包含 `backend/venv/` 和常见的 `node_modules` 忽略项。
- 如果推送到远程出现 SSH 权限问题，请配置 SSH Key 或临时切换为 HTTPS remote：

```bash
# 临时切换到 HTTPS
git remote set-url origin https://github.com/Wjh12333/rag-full-project.git
git push origin main
```

如果需要，我可以：
- 帮你把 README 推送到远程（需可用的推送权限），或
- 帮你完善 README 中的运行/部署细节。

---
最后更新：已将 `backend/venv/` 添加到 `.gitignore` 并提交到本地分支 `main`。

## 后端模型与 LLM 安装说明

后端需要一个嵌入模型（用于向量化文档）和一个语言模型/推理后端（用于生成回答）。下面列出常见的几种可选方案与安装步骤：

1) 使用托管 API（推荐，最简单）

- 支持：OpenAI、Azure OpenAI、其他云服务。
- 步骤：在 `backend` 下安装依赖并设置环境变量：

```powershell
cd backend
python -m pip install -U pip
python -m pip install -r requirements.txt  # 如果已有 requirements.txt
# 设置环境变量（示例：Windows Powershell）
$env:OPENAI_API_KEY = "sk-..."
```

后端代码应读取 `OPENAI_API_KEY`（或其它提供商对应的环境变量）并调用 API 获得嵌入与生成。

2) 使用 Hugging Face Transformers（本地/GPU 推理）

- 安装示例：

```bash
cd backend
python -m pip install -U pip
python -m pip install transformers accelerate huggingface_hub sentence-transformers chromadb torch
```

- 下载模型示例（通过 huggingface_hub）：

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2")  # 嵌入模型
snapshot_download(repo_id="meta-llama/Llama-2-7b-chat-hf")          # 示例 LLM（需有访问权限）
```

- 注意：大型模型需要 GPU + 足够显存，或使用量化/bitsandbytes 减小显存占用。

3) 使用本地轻量 LLM（llama.cpp / ggml）

- 适用于离线/低成本推理。先把模型转换为 ggml 格式，然后使用 `llama.cpp` 或对应 Python 绑定运行。
- 参考：将模型文件放在 `backend/models/`，然后在后端配置路径加载。

4) 使用 Ollama / 本地容器化模型

- 安装 Ollama（https://ollama.com/）或使用容器化镜像，然后拉取模型：

```bash
# 安装并拉取模型（示例）
ollama pull llama2
# 本地服务启动后，后端可以通过 Ollama API / socket 调用
```

5) 嵌入模型建议

- 推荐轻量嵌入：`sentence-transformers/all-MiniLM-L6-v2`（速度快，效果好）
- 如果使用 OpenAI：可用 `text-embedding-3-small` 或 `text-embedding-3-large`（取决于费用/精度需求）。

常见后端依赖（建议写入 `backend/requirements.txt`）：

```
fastapi
uvicorn
transformers
sentence-transformers
chromadb
huggingface-hub
torch  # 或根据 GPU/CPU 环境选择合适的安装方式
openai  # 如果使用 OpenAI
```

小结：
- 开发/测试：用 OpenAI API（或云 API）最快；
- 本地部署：Hugging Face 或 Ollama 更灵活，但需要模型权重和计算资源；
- 嵌入与向量库：`sentence-transformers` + `chromadb` 是常见组合。

如需，我可以把上述示例命令整理为 `backend/INSTALL.md`，或直接把推荐依赖写入 `backend/requirements.txt` 并运行一次安装测试。 
