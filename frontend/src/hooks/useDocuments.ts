import { useState, useEffect, useCallback } from 'react'
import type { Document } from '../types'

export function useDocuments() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchDocuments = useCallback(async () => {
    try {
      const res = await fetch('/api/documents')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setDocuments(await res.json())
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => {
    fetchDocuments()
  }, [fetchDocuments])

  const upload = useCallback(async (file: File) => {
    setUploading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch('/api/documents', { method: 'POST', body: form })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail ?? `HTTP ${res.status}`)
      }
      // POST returns {id, filename, status, created_at} — same shape as GET list
      const raw = await res.json()
      const doc: Document = {
        id: raw.id ?? raw.document_id,   // handle both field names defensively
        filename: raw.filename ?? file.name,
        status: raw.status ?? 'uploaded',
        created_at: raw.created_at ?? new Date().toISOString(),
      }
      setDocuments(prev => [doc, ...prev])
      // Poll status until terminal state
      const poll = setInterval(async () => {
        const s = await fetch(`/api/documents/${doc.id}/status`)
        if (!s.ok) return
        const { status } = await s.json()
        setDocuments(prev =>
          prev.map(d => (d.id === doc.id ? { ...d, status } : d))
        )
        if (status === 'ready' || status === 'failed' || status === 'error') clearInterval(poll)
      }, 2000)
    } catch (e) {
      setError(String(e))
    } finally {
      setUploading(false)
    }
  }, [])

  const remove = useCallback(async (id: string) => {
    try {
      await fetch(`/api/documents/${id}`, { method: 'DELETE' })
      setDocuments(prev => prev.filter(d => d.id !== id))
    } catch (e) {
      setError(String(e))
    }
  }, [])

  return { documents, uploading, error, upload, remove, refresh: fetchDocuments }
}
