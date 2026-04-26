import { createRouter, createWebHashHistory } from 'vue-router'
import Dashboard from '../views/Dashboard.vue'
import PretrainConfig from '../views/PretrainConfig.vue'
import SFTConfig from '../views/SFTConfig.vue'
import TrainingMonitor from '../views/TrainingMonitor.vue'
import ModelChat from '../views/ModelChat.vue'

const routes = [
  { path: '/', component: Dashboard, meta: { title: '仪表盘' } },
  { path: '/pretrain', component: PretrainConfig, meta: { title: '预训练配置' } },
  { path: '/sft', component: SFTConfig, meta: { title: 'SFT 微调' } },
  { path: '/monitor', component: TrainingMonitor, meta: { title: '训练监控' } },
  { path: '/chat', component: ModelChat, meta: { title: '模型对话' } },
]

export default createRouter({
  history: createWebHashHistory(),
  routes,
})
