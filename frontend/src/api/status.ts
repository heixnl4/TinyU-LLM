import client from './client'

export function getStatus() {
  return client.get('/api/status')
}
