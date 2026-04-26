<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { getStatus } from '../api/status'
import { getTrainStatus } from '../api/train'
import { getModelStatus } from '../api/model'
import type { FileInfo } from '../api/files'
import { listCheckpoints, listDatasets } from '../api/files'

const serverStatus = ref<any>(null)
const trainStatus = ref<any>(null)
const modelStatus = ref<any>(null)
const checkpoints = ref<FileInfo[]>([])
const datasets = ref<FileInfo[]>([])
const loading = ref(false)

let timer: ReturnType<typeof setInterval> | null = null

async function fetchAll() {
  loading.value = true
  try {
    const [s, t, m, c, d] = await Promise.all([
      getStatus(),
      getTrainStatus(),
      getModelStatus(),
      listCheckpoints(),
      listDatasets(),
    ])
    serverStatus.value = s
    trainStatus.value = t
    modelStatus.value = m
    checkpoints.value = c?.data || []
    datasets.value = d?.data || []
  } catch (e) {
    console.error(e)
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  fetchAll()
  timer = setInterval(fetchAll, 3000)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
})

function formatBytes(bytes: number) {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}
</script>

<template>
  <div v-loading="loading">
    <el-row :gutter="20">
      <el-col :span="8">
        <el-card shadow="hover">
          <template #header>
            <div class="card-header">
              <el-icon><Odometer /></el-icon>
              <span>服务状态</span>
            </div>
          </template>
          <div v-if="serverStatus">
            <p>后端状态: <el-tag :type="serverStatus.code === 0 ? 'success' : 'danger'">{{ serverStatus.code === 0 ? '正常' : '异常' }}</el-tag></p>
            <p>任务运行中: <el-tag :type="serverStatus.task_running ? 'warning' : 'info'">{{ serverStatus.task_running ? '是' : '否' }}</el-tag></p>
            <p>模型已加载: <el-tag :type="serverStatus.model_loaded ? 'success' : 'info'">{{ serverStatus.model_loaded ? '是' : '否' }}</el-tag></p>
          </div>
          <el-empty v-else description="暂无数据" />
        </el-card>
      </el-col>

      <el-col :span="8">
        <el-card shadow="hover">
          <template #header>
            <div class="card-header">
              <el-icon><DataLine /></el-icon>
              <span>当前任务</span>
            </div>
          </template>
          <div v-if="trainStatus?.data">
            <p>任务类型: <el-tag>{{ trainStatus.data.task_type }}</el-tag></p>
            <p>任务状态: <el-tag :type="trainStatus.data.status === 'running' ? 'warning' : 'info'">{{ trainStatus.data.status }}</el-tag></p>
            <p>任务ID: {{ trainStatus.data.task_id }}</p>
            <p>启动时间: {{ trainStatus.data.start_time }}</p>
          </div>
          <el-empty v-else description="暂无运行中的任务" />
        </el-card>
      </el-col>

      <el-col :span="8">
        <el-card shadow="hover">
          <template #header>
            <div class="card-header">
              <el-icon><Cpu /></el-icon>
              <span>模型状态</span>
            </div>
          </template>
          <div v-if="modelStatus?.data?.model_loaded">
            <p>设备: {{ modelStatus.data.device }}</p>
            <p>基础权重: <el-text truncated style="max-width: 200px">{{ modelStatus.data.base_weight }}</el-text></p>
            <p>LoRA权重: {{ modelStatus.data.lora_weight || '无' }}</p>
            <p>加载时间: {{ modelStatus.data.loaded_at }}</p>
          </div>
          <el-empty v-else description="模型未加载" />
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top: 20px">
      <el-col :span="12">
        <el-card shadow="hover">
          <template #header>
            <div class="card-header">
              <el-icon><Folder /></el-icon>
              <span>可用权重 ({{ checkpoints.length }})</span>
            </div>
          </template>
          <el-table :data="checkpoints.slice(0, 10)" size="small" max-height="300">
            <el-table-column prop="name" label="文件名" show-overflow-tooltip />
            <el-table-column prop="size" label="大小" width="100">
              <template #default="{ row }">{{ formatBytes(row.size) }}</template>
            </el-table-column>
            <el-table-column prop="mtime" label="修改时间" width="160" />
          </el-table>
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card shadow="hover">
          <template #header>
            <div class="card-header">
              <el-icon><Document /></el-icon>
              <span>可用数据集 ({{ datasets.length }})</span>
            </div>
          </template>
          <el-table :data="datasets" size="small" max-height="300">
            <el-table-column prop="name" label="文件名" show-overflow-tooltip />
            <el-table-column prop="size" label="大小" width="100">
              <template #default="{ row }">{{ formatBytes(row.size) }}</template>
            </el-table-column>
            <el-table-column prop="mtime" label="修改时间" width="160" />
          </el-table>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
}
</style>
