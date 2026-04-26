<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, TitleComponent } from 'echarts/components'
import VChart from 'vue-echarts'
import { getTrainStatus, getTrainLogs, getTrainMetrics, stopTrain } from '../api/train'

use([CanvasRenderer, LineChart, GridComponent, TooltipComponent, LegendComponent, TitleComponent])

const logs = ref<string[]>([])
const isRunning = ref(false)
const taskInfo = ref<any>(null)
const wsConnected = ref(false)

const lossData = ref<Array<{ x: number; y: number }>>([])
const lrData = ref<Array<{ x: number; y: number }>>([])

const chartOption = ref({
  title: { text: 'Loss 曲线', left: 'center' },
  tooltip: { trigger: 'axis' },
  legend: { data: ['Loss', 'Aux Loss'], bottom: 0 },
  grid: { left: '3%', right: '4%', bottom: '15%', top: '15%', containLabel: true },
  xAxis: { type: 'value', name: 'Step' },
  yAxis: { type: 'value', name: 'Loss' },
  series: [
    { name: 'Loss', type: 'line', smooth: true, data: [] as any[], itemStyle: { color: '#409eff' } },
    { name: 'Aux Loss', type: 'line', smooth: true, data: [] as any[], itemStyle: { color: '#e6a23c' } },
  ],
})

const lrChartOption = ref({
  title: { text: '学习率变化', left: 'center' },
  tooltip: { trigger: 'axis' },
  grid: { left: '3%', right: '4%', bottom: '10%', top: '15%', containLabel: true },
  xAxis: { type: 'value', name: 'Step' },
  yAxis: { type: 'value', name: 'LR' },
  series: [
    { name: 'LR', type: 'line', smooth: true, data: [] as any[], itemStyle: { color: '#67c23a' } },
  ],
})

const logContainer = ref<HTMLDivElement>()
let ws: WebSocket | null = null
let pollTimer: ReturnType<typeof setInterval> | null = null

async function fetchStatus() {
  try {
    const status: any = await getTrainStatus()
    taskInfo.value = status.data
    isRunning.value = status.data?.status === 'running'
  } catch (e) {}
}

async function fetchLogs() {
  try {
    const res: any = await getTrainLogs(100)
    logs.value = (res.data || []).map((l: any) => `[${l.time}] ${l.message}`)
    await nextTick(() => {
      if (logContainer.value) {
        logContainer.value.scrollTop = logContainer.value.scrollHeight
      }
    })
  } catch (e) {}
}

async function fetchMetrics() {
  try {
    const res: any = await getTrainMetrics()
    const data = res.data || {}
    if (data.loss) {
      chartOption.value.series[0].data = data.loss.map((p: any) => [p.x, p.y])
    }
    if (data.aux_loss) {
      chartOption.value.series[1].data = data.aux_loss.map((p: any) => [p.x, p.y])
    }
    if (data.learning_rate) {
      lrChartOption.value.series[0].data = data.learning_rate.map((p: any) => [p.x, p.y])
    }
  } catch (e) {}
}

async function handleStop() {
  try {
    const res: any = await stopTrain()
    ElMessage.success(res.message || '已发送停止信号')
    isRunning.value = false
  } catch (e: any) {
    ElMessage.error(e)
  }
}

function connectWS() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = `${protocol}//${window.location.host}/api/ws/train`
  ws = new WebSocket(wsUrl)

  ws.onopen = () => { wsConnected.value = true }
  ws.onclose = () => { wsConnected.value = false }
  ws.onerror = () => { wsConnected.value = false }

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      if (data.type === 'train_update') {
        if (data.logs?.length) {
          logs.value = data.logs.map((l: any) => `[${l.time}] ${l.message}`)
          nextTick(() => {
            if (logContainer.value) {
              logContainer.value.scrollTop = logContainer.value.scrollHeight
            }
          })
        }
        if (data.metrics?.loss) {
          chartOption.value.series[0].data = data.metrics.loss.map((p: any) => [p.x, p.y])
        }
        if (data.metrics?.aux_loss) {
          chartOption.value.series[1].data = data.metrics.aux_loss.map((p: any) => [p.x, p.y])
        }
        if (data.metrics?.learning_rate) {
          lrChartOption.value.series[0].data = data.metrics.learning_rate.map((p: any) => [p.x, p.y])
        }
        if (data.task) {
          taskInfo.value = data.task
          isRunning.value = data.task.status === 'running'
        }
      }
    } catch (e) {}
  }
}

function disconnectWS() {
  if (ws) {
    ws.close()
    ws = null
  }
}

onMounted(() => {
  fetchStatus()
  fetchLogs()
  fetchMetrics()
  connectWS()
  pollTimer = setInterval(() => {
    fetchStatus()
    if (!wsConnected.value) connectWS()
  }, 5000)
})

onUnmounted(() => {
  disconnectWS()
  if (pollTimer) clearInterval(pollTimer)
})
</script>

<template>
  <div>
    <div style="margin-bottom: 20px; display: flex; align-items: center; gap: 16px">
      <h3 style="margin: 0">训练监控</h3>
      <el-tag :type="isRunning ? 'warning' : 'info'" size="large">
        {{ isRunning ? '训练中' : '空闲' }}
      </el-tag>
      <el-tag :type="wsConnected ? 'success' : 'danger'" size="small">
        WebSocket {{ wsConnected ? '已连接' : '未连接' }}
      </el-tag>
      <el-button v-if="isRunning" type="danger" @click="handleStop">
        <el-icon><VideoPause /></el-icon> 停止训练
      </el-button>
    </div>

    <el-row :gutter="20">
      <el-col :span="12">
        <el-card shadow="hover">
          <template #header>
            <div class="card-header"><el-icon><DataLine /></el-icon><span>Loss 曲线</span></div>
          </template>
          <v-chart class="chart" :option="chartOption" autoresize />
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card shadow="hover">
          <template #header>
            <div class="card-header"><el-icon><TrendCharts /></el-icon><span>学习率变化</span></div>
          </template>
          <v-chart class="chart" :option="lrChartOption" autoresize />
        </el-card>
      </el-col>
    </el-row>

    <el-card shadow="hover" style="margin-top: 20px">
      <template #header>
        <div class="card-header"><el-icon><Document /></el-icon><span>实时日志</span></div>
      </template>
      <div ref="logContainer" class="log-viewer">
        <div v-for="(line, idx) in logs" :key="idx" class="log-line">{{ line }}</div>
        <el-empty v-if="logs.length === 0" description="暂无日志" />
      </div>
    </el-card>
  </div>
</template>

<style scoped>
.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
}
.chart {
  height: 300px;
}
.log-viewer {
  height: 400px;
  overflow-y: auto;
  background-color: #1a1a2e;
  color: #c0c0c0;
  padding: 12px;
  border-radius: 4px;
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 13px;
  line-height: 1.6;
}
.log-line {
  white-space: pre-wrap;
  word-break: break-all;
}
</style>
