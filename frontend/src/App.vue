<script setup lang="ts">
import { ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'

const route = useRoute()
const router = useRouter()
const isCollapse = ref(false)

const menuItems = [
  { index: '/', icon: 'Odometer', title: '仪表盘' },
  { index: '/pretrain', icon: 'Cpu', title: '预训练配置' },
  { index: '/sft', icon: 'SetUp', title: 'SFT 微调' },
  { index: '/monitor', icon: 'DataLine', title: '训练监控' },
  { index: '/chat', icon: 'ChatDotRound', title: '模型对话' },
]

function handleSelect(path: string) {
  router.push(path)
}
</script>

<template>
  <el-container class="app-container">
    <!-- 侧边栏 -->
    <el-aside :width="isCollapse ? '64px' : '200px'" class="sidebar">
      <div class="logo">
        <el-icon size="24"><Cpu /></el-icon>
        <span v-if="!isCollapse" class="logo-text">TinyU-LLM</span>
      </div>
      <el-menu
        :default-active="route.path"
        :collapse="isCollapse"
        :collapse-transition="false"
        router
        class="nav-menu"
        @select="handleSelect"
      >
        <el-menu-item v-for="item in menuItems" :key="item.index" :index="item.index">
          <el-icon>
            <component :is="item.icon" />
          </el-icon>
          <template #title>{{ item.title }}</template>
        </el-menu-item>
      </el-menu>
      <div class="collapse-btn" @click="isCollapse = !isCollapse">
        <el-icon size="18">
          <component :is="isCollapse ? 'Expand' : 'Fold'" />
        </el-icon>
      </div>
    </el-aside>

    <!-- 主内容区 -->
    <el-container>
      <el-header class="app-header">
        <h2>{{ route.meta.title || 'TinyU-LLM' }}</h2>
      </el-header>
      <el-main class="app-main">
        <keep-alive>
          <router-view />
        </keep-alive>
      </el-main>
    </el-container>
  </el-container>
</template>

<style scoped>
.app-container {
  height: 100vh;
}

.sidebar {
  background-color: #1a1a2e;
  color: #fff;
  display: flex;
  flex-direction: column;
  transition: width 0.3s;
}

.logo {
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  border-bottom: 1px solid rgba(255,255,255,0.1);
}

.logo-text {
  font-size: 18px;
  font-weight: bold;
  color: #fff;
  white-space: nowrap;
}

.nav-menu {
  flex: 1;
  border-right: none;
  background-color: transparent;
}

:deep(.nav-menu .el-menu-item) {
  color: #b0b0c0;
}

:deep(.nav-menu .el-menu-item.is-active) {
  color: #409eff;
  background-color: rgba(64, 158, 255, 0.1);
}

:deep(.nav-menu .el-menu-item:hover) {
  color: #fff;
  background-color: rgba(255,255,255,0.05);
}

.collapse-btn {
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  color: #b0b0c0;
  border-top: 1px solid rgba(255,255,255,0.1);
}

.collapse-btn:hover {
  color: #fff;
  background-color: rgba(255,255,255,0.05);
}

.app-header {
  background-color: #fff;
  border-bottom: 1px solid #e4e7ed;
  display: flex;
  align-items: center;
}

.app-header h2 {
  margin: 0;
  font-size: 18px;
  color: #333;
}

.app-main {
  background-color: #f5f7fa;
  padding: 20px;
  overflow-y: auto;
}
</style>
