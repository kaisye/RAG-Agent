export interface Document {
  id: string
  filename: string
  status: 'pending' | 'uploaded' | 'parsing' | 'chunking' | 'embedding' | 'ready' | 'failed' | 'error'
  created_at: string
}

export type Citation =
  | { type: 'text';  document_id: string; page: number; snippet: string }
  | { type: 'image'; document_id: string; page: number; thumbnail_url: string }

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
}
