import type { KnowledgeItem } from './types'

const API_BASE = '/api'

export interface DocumentList {
  total: number
  documents: KnowledgeItem[]
}

export interface StreamEvent {
  stage: string
  content: string
}

export async function fetchDocuments(): Promise<DocumentList> {
  const res = await fetch(`${API_BASE}/documents`)
  const data = await res.json()
  if (!res.ok || data.code !== 0) {
    throw new Error(data.msg ?? '加载知识库失败')
  }
  return data as DocumentList
}

export async function uploadDocument(file: File): Promise<{ chunk_count: number }> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: form })
  const data = await res.json()
  if (!res.ok || data.code !== 0) {
    throw new Error(data.msg ?? '文档导入失败')
  }
  return data as { chunk_count: number }
}

export async function clearKnowledgeBase(): Promise<void> {
  const res = await fetch(`${API_BASE}/clear_db`)
  const data = await res.json()
  if (!res.ok || data.code !== 0) {
    throw new Error(data.msg ?? '清空知识库失败')
  }
}

function parseSSEBlock(block: string): StreamEvent | null {
  let data = ''
  for (const line of block.split('\n')) {
    if (line.startsWith('data:')) {
      data += line.slice(5).trimStart()
    }
  }
  if (!data) return null
  try {
    return JSON.parse(data) as StreamEvent
  } catch {
    return null
  }
}

export async function* streamAgent(
  userQuery: string,
  signal?: AbortSignal
): AsyncGenerator<StreamEvent> {
  const url = `${API_BASE}/agent_stream?user_query=${encodeURIComponent(userQuery)}&max_loop=3`
  const res = await fetch(url, { signal })
  if (!res.ok || !res.body) {
    throw new Error(`请求失败 (${res.status})`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      // 兼容 SSE 的 \r\n / \n / \r 三种行尾，统一转为 \n 后再按空行切块，
      // 否则 sse-starlette 默认的 \r\n\r\n 分隔会无法被 "\n\n" 匹配到
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n|\r/g, '\n')
      const blocks = buffer.split('\n\n')
      buffer = blocks.pop() ?? ''
      for (const block of blocks) {
        const event = parseSSEBlock(block)
        if (event) yield event
      }
    }
    // 流结束时处理残留的最后一块
    buffer = buffer.replace(/\r\n|\r/g, '\n')
    const event = parseSSEBlock(buffer)
    if (event) yield event
  } finally {
    reader.releaseLock()
  }
}
