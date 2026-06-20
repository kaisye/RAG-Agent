export interface Document {
  id: string
  filename: string
  status: 'pending' | 'uploaded' | 'parsing' | 'chunking' | 'embedding' | 'ready' | 'failed' | 'error'
  created_at: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
}
