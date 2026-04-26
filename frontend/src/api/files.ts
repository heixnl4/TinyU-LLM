import client from './client'

export interface FileInfo {
  path: string
  name: string
  size: number
  mtime: string
}

export function listCheckpoints() {
  return client.get('/api/files/checkpoints')
}

export function listDatasets() {
  return client.get('/api/files/datasets')
}

export function uploadDataset(file: File) {
  const form = new FormData()
  form.append('file', file)
  return client.post('/api/files/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}
