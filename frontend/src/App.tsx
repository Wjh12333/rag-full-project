import { useCallback, useEffect, useRef, useState } from 'react'
import { clearKnowledgeBase, fetchDocuments, streamAgent, uploadDocument } from './api'
import type { ChatMessage, KnowledgeItem } from './types'
import './App.css'

const PREVIEW_LIMIT = 200

function preview(text: string): string {
  return text.length > PREVIEW_LIMIT ? `${text.slice(0, PREVIEW_LIMIT)}…` : text
}

export default function App() {
  const [documents, setDocuments] = useState<KnowledgeItem[]>([])
  const [loadingDocs, setLoadingDocs] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState('')

  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const chatBodyRef = useRef<HTMLDivElement | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const idSeqRef = useRef(0)

  const nextId = () => ++idSeqRef.current

  const loadDocuments = useCallback(async () => {
    setLoadingDocs(true)
    try {
      const data = await fetchDocuments()
      setDocuments(data.documents)
    } catch (err) {
      setError('加载知识库失败，请确认后端服务已启动')
      console.error(err)
    } finally {
      setLoadingDocs(false)
    }
  }, [])

  useEffect(() => {
    void loadDocuments()
  }, [loadDocuments])

  useEffect(() => {
    const el = chatBodyRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    setError('')
    try {
      const { chunk_count } = await uploadDocument(file)
      await loadDocuments()
      setMessages(prev => [
        ...prev,
        { id: nextId(), role: 'assistant', content: `已导入《${file.name}》，共 ${chunk_count} 个片段` },
      ])
    } catch (err) {
      setError('文档导入失败，请确认文件为 PDF / TXT 格式')
      console.error(err)
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleClear = async () => {
    if (!window.confirm('确定清空知识库中的全部内容？')) return
    setClearing(true)
    setError('')
    try {
      await clearKnowledgeBase()
      setDocuments([])
    } catch (err) {
      setError('清空知识库失败')
      console.error(err)
    } finally {
      setClearing(false)
    }
  }

  const handleSend = async () => {
    const text = input.trim()
    if (!text || streaming) return
    setInput('')
    setError('')

    const userMsg: ChatMessage = { id: nextId(), role: 'user', content: text }
    const assistantMsg: ChatMessage = { id: nextId(), role: 'assistant', content: '' }
    setMessages(prev => [...prev, userMsg, assistantMsg])
    setStreaming(true)

    const controller = new AbortController()
    abortRef.current = controller
    try {
      for await (const event of streamAgent(text, controller.signal)) {
        if (event.stage === 'stream_content') {
          setMessages(prev =>
            prev.map(m => (m.id === assistantMsg.id ? { ...m, content: m.content + event.content } : m)),
          )
        } else if (event.stage === 'tool_call' || event.stage === 'tool_result') {
          setMessages(prev =>
            prev.map(m => (m.id === assistantMsg.id ? { ...m, content: `${m.content}\n${event.content}\n` } : m)),
          )
        } else if (event.stage === 'final_answer') {
          break
        }
      }
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        setError('请求失败，请确认后端服务已启动')
        console.error(err)
      }
    } finally {
      setStreaming(false)
      abortRef.current = null
    }
  }

  const handleStop = () => {
    abortRef.current?.abort()
  }

  const handleReset = () => {
    setMessages([])
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void handleSend()
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>知识库助手</h1>
        <p>基于文档检索的智能问答</p>
      </header>

      <main className="app-body">
        <aside className="side-panel">
          <section className="panel-card">
            <div className="panel-title">知识库</div>
            <div className="toolbar">
              <button
                type="button"
                className="btn btn-primary"
                disabled={uploading || clearing}
                onClick={() => fileInputRef.current?.click()}
              >
                {uploading ? '导入中…' : '导入文档'}
              </button>
              <button
                type="button"
                className="btn btn-danger"
                disabled={clearing || uploading || documents.length === 0}
                onClick={() => void handleClear()}
              >
                {clearing ? '清空中…' : '清空'}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.txt"
                hidden
                onChange={e => void handleFileChange(e)}
              />
            </div>
            <p className="hint">支持 PDF / TXT，内容自动分块入库</p>
            <div className="doc-list">
              {loadingDocs ? (
                <p className="empty">加载中…</p>
              ) : documents.length === 0 ? (
                <p className="empty">知识库暂无内容，可以上传文档</p>
              ) : (
                <>
                  <p className="stat">共 {documents.length} 个片段</p>
                  {documents.map(doc => (
                    <div className="doc-item" key={doc.id}>
                      <p>{preview(doc.content)}</p>
                    </div>
                  ))}
                </>
              )}
            </div>
          </section>
        </aside>

        <section className="chat-panel">
          <div className="chat-window" ref={chatBodyRef}>
            {messages.length === 0 ? (
              <div className="chat-empty">
                <p>你好，我是知识库助手</p>
                <p>可以询问已上传文档中的内容，也支持计算和时间查询</p>
              </div>
            ) : (
              messages.map(msg => (
                <div key={msg.id} className={`msg ${msg.role === 'user' ? 'msg-user' : 'msg-ai'}`}>
                  <div className="msg-bubble">{msg.content || '…'}</div>
                </div>
              ))
            )}
          </div>
          {error && <div className="error-bar">{error}</div>}
          <div className="chat-input">
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入问题，Enter 发送，Shift+Enter 换行"
              rows={3}
            />
            <div className="chat-actions">
              <button type="button" className="btn btn-plain" onClick={handleReset}>
                清空会话
              </button>
              {streaming ? (
                <button type="button" className="btn btn-danger" onClick={handleStop}>
                  停止
                </button>
              ) : (
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={!input.trim()}
                  onClick={() => void handleSend()}
                >
                  发送
                </button>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}

