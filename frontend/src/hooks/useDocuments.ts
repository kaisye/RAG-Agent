import { useState, useCallback, useEffect, useRef } from 'react'
import type { Document } from '../types'

const API = '/api/documents'
const POLL_INTERVAL_MS = 2000

export function useDocuments() {
  const [documents, setDocuments]     = useState<Document[]>([])
  const [uploading, setUploading]     = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [loadError, setLoadError]     = useState<string | null>(null)

  const pollersRef = useRef<Map<string, ReturnType<typeof setInterval>>>(new Map())

  const fetchDocuments = useCallback(async () => {
    try {
      const res = await fetch(API)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data: Document[] = await res.json()
      setDocuments(data)
      setLoadError(null)
      data.forEach(doc => {
        if (doc.status === 'processing' || doc.status === 'uploaded') {
          startPolling(doc.id)
        }
      })
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : String(err))
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    fetchDocuments()
    return () => {
      pollersRef.current.forEach(id => clearInterval(id))
      pollersRef.current.clear()
    }
  }, [fetchDocuments])

  const startPolling = useCallback((documentId: string) => {
    if (pollersRef.current.has(documentId)) return

    const intervalId = setInterval(async () => {
      try {
        const res = await fetch(`${API}/${documentId}/status`)
        if (!res.ok) return
        const status: { id: string; status: string } = await res.json()

        setDocuments(prev =>
          prev.map(d => d.id === documentId ? { ...d, status: status.status as Document['status'] } : d)
        )

        if (status.status === 'ready' || status.status === 'error' || status.status === 'failed') {
          clearInterval(pollersRef.current.get(documentId))
          pollersRef.current.delete(documentId)
        }
      } catch {
        // transient network error — keep polling
      }
    }, POLL_INTERVAL_MS)

    pollersRef.current.set(documentId, intervalId)
  }, [])

  const upload = useCallback(async (file: File): Promise<Document | null> => {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadError('Chỉ chấp nhận file PDF.')
      return null
    }
    setUploadError(null)
    setUploading(true)

    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(API, { method: 'POST', body: form })
      if (!res.ok) {
        const detail = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(detail.detail ?? res.statusText)
      }

      const raw = await res.json()
      const doc: Document = {
        id:         raw.document_id ?? raw.id,
        filename:   raw.filename ?? file.name,
        status:     raw.status ?? 'uploaded',
        created_at: new Date().toISOString(),
      }

      setDocuments(prev => [doc, ...prev])
      startPolling(doc.id)
      return doc
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : String(err))
      return null
    } finally {
      setUploading(false)
    }
  }, [startPolling])

  const remove = useCallback(async (documentId: string): Promise<boolean> => {
    const pollerId = pollersRef.current.get(documentId)
    if (pollerId) { clearInterval(pollerId); pollersRef.current.delete(documentId) }

    try {
      const res = await fetch(`${API}/${documentId}`, { method: 'DELETE' })
      if (!res.ok && res.status !== 404) throw new Error(`HTTP ${res.status}`)
      setDocuments(prev => prev.filter(d => d.id !== documentId))
      return true
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : String(err))
      return false
    }
  }, [])

  return {
    documents,
    uploading,
    uploadError,
    loadError,
    upload,
    remove,
    refresh: fetchDocuments,
    clearUploadError: () => setUploadError(null),
  }
}
