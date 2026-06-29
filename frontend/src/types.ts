export interface Document {
  id: string
  filename: string
  status: 'pending' | 'uploaded' | 'parsing' | 'chunking' | 'embedding' | 'ready' | 'failed' | 'error'
  created_at: string
}

export interface Project {
  id: string
  name: string
  description: string
  created_at: string
  document_ids: string[]
}

export type Citation =
  | { type: 'text';  document_id: string; page: number; snippet: string }
  | { type: 'image'; document_id: string; page: number; thumbnail_url: string }

export interface DebugChunk {
  chunk_id: string
  page: number
  score: number
  snippet: string
}

export interface DebugInfo {
  sub_queries: string[]
  hyde_docs: string[]
  vector_hits: DebugChunk[]
  bm25_hits: DebugChunk[]
  rrf_candidates: DebugChunk[]
  reranked: DebugChunk[]
  latency_ms: { transform?: number; search?: number; rerank?: number }
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  debug?: DebugInfo
}
