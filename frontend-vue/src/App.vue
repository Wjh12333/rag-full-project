<template>
  <div class="app">
    <header class="app-header">
      <h1>知识库助手</h1>
      <p>基于文档检索的智能问答</p>
    </header>

    <main class="app-body">
      <aside class="side-panel">
        <section class="panel-card">
          <div class="panel-title">知识库</div>
          <div class="toolbar">
            <button type="button" class="btn btn-primary" :disabled="uploading || clearing" @click="fileInputRef?.click()">
              {{ uploading ? '导入中…' : '导入文档' }}
            </button>
            <button type="button" class="btn btn-danger" :disabled="clearing || uploading || documents.length === 0" @click="handleClear">
              {{ clearing ? '清空中…' : '清空' }}
            </button>
            <input ref="fileInputRef" type="file" accept=".pdf,.txt" hidden @change="handleUpload" />
          </div>
          <p class="hint">支持 PDF / TXT，内容自动分块入库</p>
          <div class="doc-list">
            <p v-if="loadingDocs" class="empty">加载中…</p>
            <p v-else-if="documents.length === 0" class="empty">知识库暂无内容，可以上传文档</p>
            <template v-else>
              <p class="stat">共 {{ documents.length }} 个片段</p>
              <div v-for="doc in documents" :key="doc.id" class="doc-item">
                <p>{{ preview(doc.content) }}</p>
              </div>
            </template>
          </div>
        </section>
      </aside>

      <section class="chat-panel">
        <div class="chat-window" ref="chatBodyRef">
          <div v-if="messages.length === 0" class="chat-empty">
            <p>你好，我是知识库助手</p>
            <p>可以询问已上传文档中的内容，也支持计算和时间查询</p>
          </div>
          <div
            v-for="msg in messages"
            :key="msg.id"
            :class="['msg', msg.role === 'user' ? 'msg-user' : 'msg-ai']"
          >
            <div class="msg-bubble">{{ msg.content || '…' }}</div>
          </div>
        </div>

        <div v-if="error" class="error-bar">{{ error }}</div>

        <div class="chat-input">
          <textarea
            v-model="input"
            rows="3"
            placeholder="输入问题，Enter 发送，Shift+Enter 换行"
            @keydown="handleKeydown"
          ></textarea>
          <div class="chat-actions">
            <button type="button" class="btn btn-plain" @click="resetChat">清空会话</button>
            <button v-if="streaming" type="button" class="btn btn-danger" @click="handleStop">停止</button>
            <button v-else type="button" class="btn btn-primary" :disabled="!input.trim()" @click="handleSend">发送</button>
          </div>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup>
import { nextTick, onMounted, reactive, ref, watch } from 'vue'
import { clearKnowledgeBase, fetchDocuments, streamAgent, uploadDocument } from './api'

const PREVIEW_LIMIT = 200

const documents = ref([])
const loadingDocs = ref(true)
const uploading = ref(false)
const clearing = ref(false)
const messages = ref([])
const input = ref('')
const streaming = ref(false)
const error = ref('')

const fileInputRef = ref(null)
const chatBodyRef = ref(null)
const abortRef = ref(null)
let idSeq = 0

const nextId = () => ++idSeq

function preview(text) {
  return text.length > PREVIEW_LIMIT ? `${text.slice(0, PREVIEW_LIMIT)}…` : text
}

async function loadDocuments() {
  loadingDocs.value = true
  try {
    const data = await fetchDocuments()
    documents.value = data.documents
  } catch (err) {
    error.value = '加载知识库失败，请确认后端服务已启动'
    console.error(err)
  } finally {
    loadingDocs.value = false
  }
}

onMounted(loadDocuments)

watch(
  messages,
  async () => {
    await nextTick()
    const el = chatBodyRef.value
    if (el) el.scrollTop = el.scrollHeight
  },
  { deep: true }
)

async function handleUpload(event) {
  const file = event.target.files?.[0]
  if (!file) return
  uploading.value = true
  error.value = ''
  try {
    const { chunk_count } = await uploadDocument(file)
    await loadDocuments()
    messages.value.push({
      id: nextId(),
      role: 'assistant',
      content: `已导入《${file.name}》，共 ${chunk_count} 个片段`,
    })
  } catch (err) {
    error.value = '文档导入失败，请确认文件为 PDF / TXT 格式'
    console.error(err)
  } finally {
    uploading.value = false
    if (fileInputRef.value) fileInputRef.value.value = ''
  }
}

async function handleClear() {
  if (!window.confirm('确定清空知识库中的全部内容？')) return
  clearing.value = true
  error.value = ''
  try {
    await clearKnowledgeBase()
    documents.value = []
  } catch (err) {
    error.value = '清空知识库失败'
    console.error(err)
  } finally {
    clearing.value = false
  }
}

async function handleSend() {
  const text = input.value.trim()
  if (!text || streaming.value) return
  input.value = ''
  error.value = ''

  const userMsg = { id: nextId(), role: 'user', content: text }
  const assistantMsg = reactive({ id: nextId(), role: 'assistant', content: '' })
  messages.value.push(userMsg, assistantMsg)
  streaming.value = true

  const controller = new AbortController()
  abortRef.value = controller
  try {
    for await (const event of streamAgent(text, controller.signal)) {
      if (event.stage === 'stream_content') {
        assistantMsg.content += event.content
      } else if (event.stage === 'tool_call' || event.stage === 'tool_result') {
        assistantMsg.content += `\n${event.content}\n`
      } else if (event.stage === 'final_answer') {
        break
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      error.value = '请求失败，请确认后端服务已启动'
      console.error(err)
    }
  } finally {
    streaming.value = false
    abortRef.value = null
  }
}

function handleStop() {
  abortRef.value?.abort()
}

function resetChat() {
  messages.value = []
}

function handleKeydown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    handleSend()
  }
}
</script>
