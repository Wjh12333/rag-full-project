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
