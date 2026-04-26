<script setup lang="ts">
import { reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { startPretrain } from '../api/train'
import { listDatasets } from '../api/files'

const submitting = ref(false)
const formRef = ref()

const form = reactive({
  epochs: 2,
  batch_size: 32,
  learning_rate: 0.0005,
  seed: 42,
  max_length: 512,
  grad_clip: 1.0,
  accumulation_steps: 4,
  dtype: 'bfloat16',
  hidden_size: 512,
  num_hidden_layers: 4,
  num_attention_heads: 8,
  num_key_value_heads: 2,
  use_moe: false,
  use_compile: false,
  use_swanlab: false,
  data_path: './dataset/pretrain_hq.jsonl',
  checkpoint_dir: './checkpoints',
  output_dir: './out',
  save_steps: 1000,
  log_interval: 100,
  project_name: 'TinyU-LLM-Pretrain',
  run_name: 'run-web',
})

const dtypeOptions = [
  { label: 'float16', value: 'float16' },
  { label: 'bfloat16', value: 'bfloat16' },
  { label: 'float32', value: 'float32' },
]

async function handleSubmit() {
  submitting.value = true
  try {
    const res: any = await startPretrain(form)
    ElMessage.success(res.message || '预训练任务已启动')
  } catch (e: any) {
    ElMessage.error(e)
  } finally {
    submitting.value = false
  }
}

async function refreshDatasets() {
  try {
    const res: any = await listDatasets()
    if (res.data?.length > 0) {
      form.data_path = res.data[0].path
      ElMessage.success('已自动选择最新的数据集')
    }
  } catch (e) {
    console.error(e)
  }
}
</script>

<template>
  <el-form ref="formRef" :model="form" label-width="140px" style="max-width: 800px">
    <h3 style="margin-bottom: 20px">预训练参数配置</h3>

    <el-divider content-position="left">基础训练参数</el-divider>
    <el-form-item label="训练轮数 (epochs)">
      <el-input-number v-model="form.epochs" :min="1" :max="100" />
    </el-form-item>
    <el-form-item label="Batch Size">
      <el-input-number v-model="form.batch_size" :min="1" :max="256" />
    </el-form-item>
    <el-form-item label="学习率">
      <el-input-number v-model="form.learning_rate" :min="1e-6" :max="1e-1" :step="1e-5" :precision="6" style="width: 180px" />
    </el-form-item>
    <el-form-item label="随机种子">
      <el-input-number v-model="form.seed" :min="0" :max="99999" />
    </el-form-item>
    <el-form-item label="最大序列长度">
      <el-input-number v-model="form.max_length" :min="64" :max="8192" :step="64" />
    </el-form-item>
    <el-form-item label="梯度裁剪阈值">
      <el-input-number v-model="form.grad_clip" :min="0.1" :max="10" :step="0.1" />
    </el-form-item>
    <el-form-item label="梯度累积步数">
      <el-input-number v-model="form.accumulation_steps" :min="1" :max="64" />
    </el-form-item>
    <el-form-item label="训练精度">
      <el-select v-model="form.dtype" style="width: 180px">
        <el-option v-for="opt in dtypeOptions" :key="opt.value" :label="opt.label" :value="opt.value" />
      </el-select>
    </el-form-item>

    <el-divider content-position="left">模型架构参数</el-divider>
    <el-form-item label="隐藏层维度">
      <el-input-number v-model="form.hidden_size" :min="128" :max="4096" :step="128" />
    </el-form-item>
    <el-form-item label="隐藏层层数">
      <el-input-number v-model="form.num_hidden_layers" :min="1" :max="32" />
    </el-form-item>
    <el-form-item label="注意力头数">
      <el-input-number v-model="form.num_attention_heads" :min="1" :max="64" />
    </el-form-item>
    <el-form-item label="KV 头数">
      <el-input-number v-model="form.num_key_value_heads" :min="1" :max="64" />
    </el-form-item>
    <el-form-item label="启用 MoE">
      <el-switch v-model="form.use_moe" />
    </el-form-item>
    <el-form-item label="启用 torch.compile">
      <el-switch v-model="form.use_compile" />
    </el-form-item>

    <el-divider content-position="left">路径与保存</el-divider>
    <el-form-item label="数据集路径">
      <el-input v-model="form.data_path" style="width: 400px">
        <template #append>
          <el-button @click="refreshDatasets">自动选择</el-button>
        </template>
      </el-input>
    </el-form-item>
    <el-form-item label="Checkpoint 目录">
      <el-input v-model="form.checkpoint_dir" style="width: 400px" />
    </el-form-item>
    <el-form-item label="输出目录">
      <el-input v-model="form.output_dir" style="width: 400px" />
    </el-form-item>
    <el-form-item label="保存间隔 (步)">
      <el-input-number v-model="form.save_steps" :min="1" />
    </el-form-item>

    <el-divider content-position="left">日志配置</el-divider>
    <el-form-item label="日志间隔 (步)">
      <el-input-number v-model="form.log_interval" :min="1" />
    </el-form-item>
    <el-form-item label="项目名称">
      <el-input v-model="form.project_name" style="width: 300px" />
    </el-form-item>
    <el-form-item label="实验名称">
      <el-input v-model="form.run_name" style="width: 300px" />
    </el-form-item>
    <el-form-item label="使用 SwanLab">
      <el-switch v-model="form.use_swanlab" />
    </el-form-item>

    <el-form-item>
      <el-button type="primary" size="large" :loading="submitting" @click="handleSubmit">
        <el-icon><VideoPlay /></el-icon>
        启动预训练
      </el-button>
    </el-form-item>
  </el-form>
</template>
