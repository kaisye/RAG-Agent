export interface Citation {
  chunk_id:      string
  document_id:   string
  page:          number
  text?:         string
  snippet?:      string
  type:          "text" | "image"
  thumbnail_url?: string | null
  score?:        number
}

export interface Document {
  id:         string
  filename:   string
  status:     "uploaded" | "processing" | "ready" | "error"
  created_at: string
  file_path?: string
}

export interface PageSnippet {
  document_id: string
  page:        number
  snippet:     string
  images:      string[]
}
