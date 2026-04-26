import client from './client'

export interface LoadWeightParams {
  weight_path: string
  hidden_size?: number
  num_hidden_layers?: number
  num_attention_heads?: number
  num_key_value_heads?: number
  use_moe?: boolean
  lora_path?: string | null
}

export interface ChatParams {
  prompt: string
  max_new_tokens?: number
  temperature?: number
  top_k?: number
  top_p?: number
}

export function loadModel(params: LoadWeightParams) {
  return client.post('/api/model/load', params)
}

export function unloadModel() {
  return client.post('/api/model/unload')
}

export function getModelStatus() {
  return client.get('/api/model/status')
}

export function chat(params: ChatParams) {
  return client.post('/api/chat', params)
}
