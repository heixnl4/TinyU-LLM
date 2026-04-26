<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { loadModel, unloadModel, getModelStatus } from '../api/model'
import { listCheckpoints } from '../api/files'
import type { FileInfo } from '../api/files'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

const checkpoints = ref<FileInfo[]>([])
const selectedWeight = ref('')
const selectedLora = ref('')
const loraOptions = ref<FileInfo[]>([])

const modelConfig = ref({
  hidden_size: 512,
  num_hidden_layers: 4,
  num_attention_heads: 8,
  num_key_value_heads: 2,
  use_moe: false,
})

const genParams = ref({
  max_new_tokens: 150,
  temperature: 0.8,
  top_k: 50,
  top_p: 0.9,
})

const isModelLoaded = ref(false)
const loadingModel = ref(false)
const messages = ref<Message[]>([])
const inputPrompt = ref('')
const sending = ref(false)
const chatContainer = ref<HTMLDivElement>()

let ws: WebSocket | null = null

async function fetchCheckpoints() {
  try {
    const res: any = await listCheckpoints()
    checkpoints.value = res.data || []
    loraOptions.value = checkpoints.value.filter((f: FileInfo) => f.name.includes('lora'))
  } catch (e) {}
}

async function fetchModelStatus() {
  try {
    const res: any = await getModelStatus()
    isModelLoaded.value = res.data?.model_loaded || false
  } catch (e) {}
}

async function handleLoad() {
  if (!selectedWeight.value) {
    ElMessage.warning('请选择权重文件')
    return
  }
  loadingModel.value = true
  try {
    const payload = {
      weight_path: selectedWeight.value,
      lora_path: selectedLora.value || null,
      ...modelConfig.value,
    }
    const res: any = await loadModel(payload)
    ElMessage.success(res.message || '模型加载成功')
    isModelLoaded.value = true
  } catch (e: any) {
    ElMessage.error(e)
  } finally {
    loadingModel.value = false
  }
}

async function handleUnload() {
  try {
    await unloadModel()
    ElMessage.success('模型已卸载')
    isModelLoaded.value = false
  } catch (e: any) {
    ElMessage.error(e)
  }
}

async function handleSend() {
  if (!inputPrompt.value.trim()) return
  if (!isModelLoaded.value) {
    ElMessage.warning('请先加载模型')
    return
  }

  const prompt = inputPrompt.value.trim()
  messages.value.push({ role: 'user', content: prompt })
  inputPrompt.value = ''
  sending.value = true

  // 添加一个空的助手消息
  messages.value.push({ role: 'assistant', content: '' })
  await nextTick(() => scrollToBottom())

  connectChatWS()

  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      prompt,
      ...genParams.value,
    }))
  } else {
    // 如果 WebSocket 没连上，回退到 HTTP 接口
    try {
      const { chat } = await import('../api/model')
      const res: any = await chat({ prompt, ...genParams.value })
      const lastMsg = messages.value[messages.value.length - 1]
      lastMsg.content = res.data?.response || '（无响应）'
    } catch (e: any) {
      const lastMsg = messages.value[messages.value.length - 1]
      lastMsg.content = '错误: ' + e
    } finally {
      sending.value = false
      await nextTick(() => scrollToBottom())
    }
  }
}

function connectChatWS() {
  if (ws) return
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = `${protocol}//${window.location.host}/api/ws/chat`
  ws = new WebSocket(wsUrl)

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      const lastMsg = messages.value[messages.value.length - 1]
      if (data.token) {
        lastMsg.content += data.token
        nextTick(() => scrollToBottom())
      }
      if (data.done) {
        sending.value = false
      }
      if (data.error) {
        lastMsg.content = '错误: ' + data.error
        sending.value = false
      }
    } catch (e) {}
  }

  ws.onclose = () => {
    ws = null
  }
}

function scrollToBottom() {
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight
  }
}

onMounted(() => {
  fetchCheckpoints()
  fetchModelStatus()
})

onUnmounted(() => {
  if (ws) ws.close()
})
</script>

<template>
  <el-row :gutter="20">
    <!-- 左侧配置面板 -->
    <el-col :span="8">
      <el-card shadow="hover">
        <template #header>
          <div class="card-header">
            <el-icon><Setting /></el-icon>
            <span>模型加载</span>
          </div>
        </template>

        <el-form label-width="120px" size="small">
          <el-form-item label="权重文件">
            <el-select v-model="selectedWeight" placeholder="选择权重" style="width: 100%">
              <el-option
                v-for="ckpt in checkpoints"
                :key="ckpt.path"
                :label="ckpt.name"
                :value="ckpt.path"
              />
            </el-select>
          </el-form-item>

          <el-form-item label="LoRA 权重">
            <el-select v-model="selectedLora" clearable placeholder="可选" style="width: 100%">
              <el-option
                v-for="ckpt in loraOptions"
                :key="ckpt.path"
                :label="ckpt.name"
                :value="ckpt.path"
              />
            </el-select>
          </el-form-item>

          <el-divider content-position="left">模型架构</el-divider>
          <el-form-item label="隐藏层维度">
            <el-input-number v-model="modelConfig.hidden_size" :min="128" :max="4096" :step="128" />
          </el-form-item>
          <el-form-item label="层数">
            <el-input-number v-model="modelConfig.num_hidden_layers" :min="1" :max="32" />
          </el-form-item>
          <el-form-item label="注意力头数">
            <el-input-number v-model="modelConfig.num_attention_heads" :min="1" :max="64" />
          </el-form-item>
          <el-form-item label="KV 头数">
            <el-input-number v-model="modelConfig.num_key_value_heads" :min="1" :max="64" />
          </el-form-item>
          <el-form-item label="MoE">
            <el-switch v-model="modelConfig.use_moe" />
          </el-form-item>

          <el-divider content-position="left">生成参数</el-divider>
          <el-form-item label="Max Tokens">
            <el-input-number v-model="genParams.max_new_tokens" :min="1" :max="2048" />
          </el-form-item>
          <el-form-item label="Temperature">
            <el-slider v-model="genParams.temperature" :min="0.1" :max="2" :step="0.1" show-input />
          </el-form-item>
          <el-form-item label="Top-K">
            <el-input-number v-model="genParams.top_k" :min="0" :max="100" />
          </el-form-item>
          <el-form-item label="Top-P">
            <el-slider v-model="genParams.top_p" :min="0" :max="1" :step="0.05" show-input />
          </el-form-item>

          <el-form-item>
            <el-button v-if="!isModelLoaded" type="primary" :loading="loadingModel" @click="handleLoad">
              加载模型
            </el-button>
            <el-button v-else type="danger" @click="handleUnload">卸载模型</el-button>
            <el-button @click="fetchCheckpoints">刷新列表</el-button>
          </el-form-item>
        </el-form>
      </el-card>
    </el-col>

    <!-- 右侧对话面板 -->
    <el-col :span="16">
      <el-card shadow="hover" class="chat-card">
        <template #header>
          <div class="card-header">
            <el-icon><ChatDotRound /></el-icon>
            <span>模型对话</span>
            <el-tag :type="isModelLoaded ? 'success' : 'info'" size="small">
              {{ isModelLoaded ? '模型已加载' : '模型未加载' }}
            </el-tag>
          </div>
        </template>

        <div ref="chatContainer" class="chat-container">
          <div
            v-for="(msg, idx) in messages"
            :key="idx"
            :class="['message', msg.role]"
          >
            <div class="message-avatar">
              <el-avatar :size="32" :icon="msg.role === 'user' ? 'User' : 'ChatDotRound'" />
            </div>
            <div class="message-content">
              <div class="message-role">{{ msg.role === 'user' ? '用户' : '模型' }}</div>
              <div class="message-text">{{ msg.content }}</div>
            </div>
          </div>

          <el-empty v-if="messages.length === 0" description="开始对话吧" />
        </div>

        <div class="chat-input">
          <el-input
            v-model="inputPrompt"
            type="textarea"
            :rows="2"
            placeholder="输入你的问题..."
            @keydown.enter.prevent="handleSend"
          />
          <el-button
            type="primary"
            :disabled="!isModelLoaded || sending"
            :loading="sending"
            @click="handleSend"
          >
            发送
          </el-button>
        </div>
      </el-card>
    </el-col>
  </el-row>
</template>

<style scoped>
.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
}

.chat-card {
  height: calc(100vh - 140px);
  display: flex;
  flex-direction: column;
}

:deep(.el-card__body) {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.chat-container {
  flex: 1;
  overflow-y: auto;
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.message {
  display: flex;
  gap: 12px;
  align-items: flex-start;
}

.message.assistant {
  flex-direction: row-reverse;
}

.message-content {
  max-width: 80%;
  padding: 10px 14px;
  border-radius: 8px;
  background-color: #f0f2f5;
}

.message.user .message-content {
  background-color: #409eff;
  color: #fff;
}

.message-role {
  font-size: 12px;
  color: #999;
  margin-bottom: 4px;
}

.message.user .message-role {
  color: rgba(255,255,255,0.8);
}

.message-text {
  font-size: 14px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-all;
}

.chat-input {
  display: flex;
  gap: 10px;
  padding-top: 10px;
  border-top: 1px solid #e4e7ed;
}

.chat-input .el-input {
  flex: 1;
}
</style>
