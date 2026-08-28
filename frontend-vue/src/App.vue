<template>
  <!-- 页面最外层容器 -->
  <div class="container">
    <!-- ElementPlus栅格布局，gutter控制列之间间距 -->
    <el-row :gutter="20">
      <!-- 左侧8列：文档上传面板 -->
      <el-col span="8">
        <!-- 卡片组件，hover悬浮阴影 -->
        <el-card shadow="hover">
          <!-- 卡片头部插槽 -->
          <template #header>
            <div>文档上传入库</div>
          </template>

          <!-- 文件上传组件，action不填，使用自定义http-request覆盖默认上传行为 -->
          <!-- accept限制可选文件后缀：pdf、txt -->
          <el-upload
            action=""
            :http-request="customUpload"
            :show-file-list="true"
            accept=".pdf,.txt"
          >
            <el-button type="primary">选择文档</el-button>
          </el-upload>

          <!-- 分割线 -->
          <el-divider />

          <!-- 清空向量库按钮，clearLoading控制loading加载状态 -->
          <el-button type="warning" @click="clearVectorStore" :loading="clearLoading">清空向量库</el-button>
        </el-card>
      </el-col>

      <!-- 右侧16列：对话聊天窗口 -->
      <el-col span="16">
        <el-card shadow="hover">
          <template #header>
            <div>RAG‑Agent 对话</div>
          </template>

          <!-- 聊天消息容器，设置高度，超出滚动 -->
          <div class="chat-window">
            <!-- 循环渲染聊天列表，idx作为key -->
            <div v-for="(msg, idx) in chatList" :key="idx" class="msg-item">
              <!-- 根据消息角色区分样式：human用户 / ai助手 -->
              <div :class="msg.role === 'human' ? 'human' : 'ai'">
                <strong>{{ msg.role === 'human' ? '用户' : 'AI' }}</strong>
                <!-- 消息内容展示 -->
                <div class="content">{{ msg.content }}</div>
              </div>
            </div>
          </div>

          <el-divider />

          <!-- 文本域输入框，双向绑定queryText，多行输入 -->
          <el-input
            v-model="queryText"
            type="textarea"
            rows="3"
            placeholder="请输入你的问题..."
          />

          <!-- 按钮区域 -->
          <div class="btn-row">
            <!-- 发送按钮，loading控制请求中状态，点击触发sendQuery -->
            <el-button type="primary" @click="sendQuery" :loading="loading">发送提问</el-button>
            <!-- 清空会话按钮 -->
            <el-button @click="resetChat">清空会话</el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
// 导入vue组合式API ref：定义响应式变量
import { ref } from 'vue'
// 导入axios，用于http接口请求
import axios from 'axios'
// ElementPlus全局消息提示组件，必须确保main.js已经完整注册ElementPlus
import { ElMessage } from 'element-plus'

/**
 * chatList：聊天消息列表
 * 每条消息结构 {role:'human'|'ai', content:'消息文本'}
 */
const chatList = ref([])

/**
 * queryText：输入框绑定的用户提问文本
 */
const queryText = ref('')

/**
 * loading：发送请求的loading状态，true代表请求/流式处理中，按钮置灰
 */
const loading = ref(false)

/**
 * clearLoading：清空向量库按钮loading状态
 */
const clearLoading = ref(false)


/**
 * 自定义文件上传函数，替换el‑upload默认的ajax上传
 * @param {Object} opt upload组件内部参数对象，opt.file为选中文件对象
 */
const customUpload = async (opt) => {
  // 创建FormData表单对象，用于上传二进制文件
  const formData = new FormData()
  // 把选中文件追加到表单，字段名file，和后端接口参数对应
  formData.append('file', opt.file)
  try {
    // post请求调用后端上传接口；vite代理会把/api转发到127.0.0.1:8000
    const res = await axios.post('/api/upload', formData, {
      // multipart/form-data 文件上传请求头
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    // 成功提示
    ElMessage.success('文档入库成功：' + res.data.chunk_count + '个分块')
  } catch (err) {
    // 异常捕获，上传失败提示
    ElMessage.error('文档处理失败，请检查文件格式')
    console.error(err)
  }
}


/**
 * sendQuery：发送用户问题，使用EventSource接收SSE流式返回
 * 对接后端 /api/agent_stream SSE接口
 */
const sendQuery = async () => {
  // 简单校验：输入为空直接返回，不发起请求
  if (!queryText.value.trim()) return

  // 获取用户输入文本
  const userQuestion = queryText.value.trim()

  // 1.把用户消息推入聊天列表
  chatList.value.push({ role: 'human', content: userQuestion })
  // 2.插入一条空AI消息，后续流式片段不断追加到content
  chatList.value.push({ role: 'ai', content: '' })

  // 清空输入框
  queryText.value = ''
  // 打开加载状态
  loading.value = true

  // 获取刚刚新增的ai空消息对象，用来不断追加流式文本
  const aiMsg = chatList.value[chatList.value.length - 1]

  // 创建EventSource实例，浏览器原生SSE客户端；url编码用户问题防止特殊字符
  const eventSource = new EventSource(`/api/agent_stream?user_query=${encodeURIComponent(userQuestion)}&max_loop=3`)

  eventSource.onmessage = (e) => {
  const item = JSON.parse(e.data)
  if(item.stage === 'tool_call'){
    // Agent调用工具提示，追加到聊天框
    aiMsg.content += "\n" + item.content + "\n"
  }else if(item.stage === 'tool_result'){
    // 工具返回结果
    aiMsg.content += "\n" + item.content + "\n"
  }else if(item.stage === 'stream_content'){
    aiMsg.content += item.content
  }else if(item.stage === 'final_answer'){
    eventSource.close()
    loading.value = false
  }
}


  // onerror：SSE连接异常、断开、报错触发
  eventSource.onerror = () => {
    eventSource.close() // 出错关闭连接释放资源
    loading.value = false
    ElMessage.warning('流式连接断开')
  }
}


/**
 * clearVectorStore：调用后端接口清空Chroma向量库
 */
const clearVectorStore = async () => {
  clearLoading.value = true // 打开按钮loading
  try {
    await axios.get('/api/clear_db')
    ElMessage.success("向量库已清空")
  } catch (err) {
    ElMessage.error("清空向量库失败")
    console.error(err)
  } finally {
    clearLoading.value = false // 无论成功失败关闭loading
  }
}


/**
 * resetChat：清空前端聊天会话记录，不会影响后端向量库
 */
const resetChat = () => {
  chatList.value = []
}
</script>

<style scoped>
/* scoped代表样式仅作用当前组件 */
.container {
  padding: 24px; /* 页面整体内边距 */
}

/* 聊天窗口容器样式 */
.chat-window {
  height: 450px; /* 固定高度 */
  overflow-y: auto; /* 内容超出垂直滚动 */
  padding: 8px;
  border: 1px solid #eee;
}

/* 单条消息外层间距 */
.msg-item {
  margin: 12px 0;
}

/* 用户消息气泡样式 */
.human {
  background: #e6f7ff;
  padding: 10px;
  border-radius: 6px;
}

/* AI消息气泡样式 */
.ai {
  background: #f5f7fa;
  padding: 10px;
  border-radius: 6px;
}

/* 按钮区域上边距 */
.btn-row {
  margin-top: 12px;
}
</style>