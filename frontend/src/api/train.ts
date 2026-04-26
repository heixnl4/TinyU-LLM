import client from './client'

export interface PretrainConfig {
  epochs?: number
  batch_size?: number
  learning_rate?: number
  seed?: number
  max_length?: number
  grad_clip?: number
  accumulation_steps?: number
  dtype?: 'float16' | 'bfloat16' | 'float32'
  hidden_size?: number
  num_hidden_layers?: number
  num_attention_heads?: number
  num_key_value_heads?: number
  use_moe?: boolean
  use_compile?: boolean
  use_swanlab?: boolean
  data_path?: string
  checkpoint_dir?: string
  output_dir?: string
  save_steps?: number
  log_interval?: number
  project_name?: string
  run_name?: string
}

export interface SFTConfig extends PretrainConfig {
  pretrained_model_path?: string | null
  lora_rank?: number
  lora_alpha?: number
  lora_dropout?: number
  target_modules?: string[]
  pretrain_run_name?: string
}

export function startPretrain(config: PretrainConfig) {
  return client.post('/api/train/pretrain', config)
}

export function startSFT(config: SFTConfig) {
  return client.post('/api/train/sft', config)
}

export function stopTrain() {
  return client.post('/api/train/stop')
}

export function getTrainStatus() {
  return client.get('/api/train/status')
}

export function getTrainHistory() {
  return client.get('/api/train/history')
}

export function getTrainLogs(lastN = 200) {
  return client.get(`/api/train/logs?last_n=${lastN}`)
}

export function getTrainMetrics() {
  return client.get('/api/train/metrics')
}
