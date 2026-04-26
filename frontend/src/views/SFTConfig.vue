<script setup lang="ts">
import { reactive, ref, onMounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { startSFT } from '../api/train'
import { listDatasets, listCheckpoints } from '../api/files'

const submitting = ref(false)
const formRef = ref()
const STORAGE_KEY = 'tinyu_sft_config'

const defaultForm = {
  epochs: 3,
  batch_size: 16,
  learning_rate: 0.00005,
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
  data_path: './dataset/sft_mini_512_part.jsonl',
  checkpoint_dir: './checkpoints',
  output_dir: './out',
  save_steps: 10,
  log_interval: 5,
  project_name: 'TinyU-LLM-SFT',
  run_name: 'lora-run-web',
  pretrain_run_name: 'run-web',
  pretrained_model_path: null as string | null,
  lora_rank: 8,
  lora_alpha: 32,
  lora_dropout: 0.1,
  target_modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
}

const form = reactive({ ...defaultForm })

// 页面挂载时从 localStorage 恢复配置
onMounted(() => {
  const saved = localStorage.getItem(STORAGE_KEY)
  if (saved) {
    try {
      const parsed = JSON.parse(saved)
      Object.assign(form, parsed)
    } catch (e) {
      console.error('恢复配置失败', e)
    }
  }
})

// 监听 form 变化，自动保存到 localStorage
watch(form, (val) => {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(val))
}, { deep: true })

const dtypeOptions = [
  { label: 'float16', value: 'float16' },
  { label: 'bfloat16', value: 'bfloat16' },
  { label: 'float32', value: 'float32' },
]

async function handleSubmit() {
  submitting.value = true
  try {
    const payload = { ...form }
    const res: any = await startSFT(payload)
    ElMessage.success(res.message || 'SFT 任务已启动')
  } catch (e: any) {
    ElMessage.error(e)
  } finally {
    submitting.value = false
  }
}

async function selectLatestDataset() {
  try {
    const res: any = await listDatasets()
    if (res.data?.length > 0) {
      form.data_path = res.data[0].path
      ElMessage.success('已自动选择最新的数据集')
    }
  } catch (e) { console.error(e) }
}

async function selectLatestWeight() {
  try {
    const res: any = await listCheckpoints()
    const weights = res.data?.filter((f: any) => f.name.includes('weight') || f.name.includes('epoch'))
    if (weights?.length > 0) {
      form.pretrained_model_path = weights[0].path
      ElMessage.success('已自动选择最新的权重')
    }
  } catch (e) { console.error(e) }
}

function resetDefaults() {
  Object.assign(form, defaultForm)
  localStorage.removeItem(STORAGE_KEY)
  ElMessage.success('已恢复默认配置')
}
</script>

<template>
  <el-form ref="formRef" :model="form" label-width="160px" style="max-width: 800px">
    <h3 style="margin-bottom: 20px">SFT (LoRA) 微调参数配置</h3>

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
    <el-form-item label="最大序列长度">
      <el-input-number v-model="form.max_length" :min="64" :max="8192" :step="64" />
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

    <el-divider content-position="left">路径配置</el-divider>
    <el-form-item label="预训练权重路径">
      <el-input v-model="form.pretrained_model_path" style="width: 400px" placeholder="留空则自动查找">
        <template #append>
          <el-button @click="selectLatestWeight">自动选择</el-button>
        </template>
      </el-input>
    </el-form-item>
    <el-form-item label="数据集路径">
      <el-input v-model="form.data_path" style="width: 400px">
        <template #append>
          <el-button @click="selectLatestDataset">自动选择</el-button>
        </template>
      </el-input>
    </el-form-item>
    <el-form-item label="输出目录">
      <el-input v-model="form.output_dir" style="width: 400px" />
    </el-form-item>
    <el-form-item label="预训练 Run 名">
      <el-input v-model="form.pretrain_run_name" style="width: 200px" placeholder="用于自动查找权重" />
    </el-form-item>

    <el-divider content-position="left">LoRA 参数</el-divider>
    <el-form-item label="LoRA Rank">
      <el-input-number v-model="form.lora_rank" :min="1" :max="128" />
    </el-form-item>
    <el-form-item label="LoRA Alpha">
      <el-input-number v-model="form.lora_alpha" :min="1" :max="256" />
    </el-form-item>
    <el-form-item label="LoRA Dropout">
      <el-input-number v-model="form.lora_dropout" :min="0" :max="1" :step="0.05" />
    </el-form-item>
    <el-form-item label="目标模块">
      <el-select v-model="form.target_modules" multiple style="width: 300px">
        <el-option label="q_proj" value="q_proj" />
        <el-option label="k_proj" value="k_proj" />
        <el-option label="v_proj" value="v_proj" />
        <el-option label="o_proj" value="o_proj" />
        <el-option label="gate_proj" value="gate_proj" />
        <el-option label="up_proj" value="up_proj" />
        <el-option label="down_proj" value="down_proj" />
      </el-select>
    </el-form-item>

    <el-divider content-position="left">日志配置</el-divider>
    <el-form-item label="实验名称">
      <el-input v-model="form.run_name" style="width: 300px" />
    </el-form-item>
    <el-form-item label="日志间隔 (步)">
      <el-input-number v-model="form.log_interval" :min="1" />
    </el-form-item>

    <el-form-item>
      <el-button type="primary" size="large" :loading="submitting" @click="handleSubmit">
        <el-icon><VideoPlay /></el-icon>
        启动 SFT 微调
      </el-button>
      <el-button size="large" @click="resetDefaults">
        <el-icon><RefreshLeft /></el-icon>
        恢复默认
      </el-button>
    </el-form-item>
  </el-form>
</template>
