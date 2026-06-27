import { useState, useCallback, useEffect, useRef } from "react"
import type { Document } from "../types/document"

const API = "/api/documents"
const POLL_INTERVAL_MS = 2000

export function useDocuments() {
  const [documents, setDocuments]   = useState<Document[]>([])
  const [uploading, setUploading]   = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [loadError, setLoadError]   = useState<string | null>(null)

  // Map of document_id → interval id for active pollers
  const pollersRef = useRef<Map<string, ReturnType<typeof setInterval>>>(new Map())

  // ── Fetch all documents ──────────────────────────────────────────────────

  const fetchDocuments = useCallback(async () => {
    try {
      const res = await fetch(API)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data: Document[] = await res.json()
      setDocuments(data)
      setLoadError(null)

      // Start polling for any document still in progress
      data.forEach(doc => {
        if (doc.status === "processing" || doc.status === "uploaded") {
          startPolling(doc.id)
        }
      })
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : String(err))
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch on mount
  useEffect(() => {
    fetchDocuments()
    return () => {
      // Clear all pollers on unmount
      pollersRef.current.forEach(id => clearInterval(id))
      pollersRef.current.clear()
    }
  }, [fetchDocuments])

  // ── Polling a single document status ─────────────────────────────────────

  const startPolling = useCallback((documentId: string) => {
    if (pollersRef.current.has(documentId)) return // already polling

    const intervalId = setInterval(async () => {
      try {
        const res = await fetch(`${API}/${documentId}/status`)
        if (!res.ok) return
        const status: { id: string; status: string } = await res.json()

        setDocuments(prev =>
          prev.map(d => d.id === documentId ? { ...d, status: status.status as Document["status"] } : d)
        )

        // Stop polling when terminal state reached
        if (status.status === "ready" || status.status === "error") {
          clearInterval(pollersRef.current.get(documentId))
          pollersRef.current.delete(documentId)
        }
      } catch {
        // Transient network error — keep polling
      }
    }, POLL_INTERVAL_MS)

    pollersRef.current.set(documentId, intervalId)
  }, [])

  // ── Upload ───────────────────────────────────────────────────────────────

  const uploadDocument = useCallback(async (file: File): Promise<Document | null> => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Chỉ chấp nhận file PDF.")
      return null
    }
    setUploadError(null)
    setUploading(true)

    try {
      const form = new FormData()
      form.append("file", file)

      const res = await fetch(API, { method: "POST", body: form })
      if (!res.ok) {
        const detail = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(detail.detail ?? res.statusText)
      }

      const newDoc: { document_id: string; filename: string; status: string } = await res.json()

      const doc: Document = {
        id:         newDoc.document_id,
        filename:   newDoc.filename,
        status:     newDoc.status as Document["status"],
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

  // ── Delete ───────────────────────────────────────────────────────────────

  const deleteDocument = useCallback(async (documentId: string): Promise<boolean> => {
    // Stop polling immediately
    const pollerId = pollersRef.current.get(documentId)
    if (pollerId) { clearInterval(pollerId); pollersRef.current.delete(documentId) }

    try {
      const res = await fetch(`${API}/${documentId}`, { method: "DELETE" })
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
    uploadDocument,
    deleteDocument,
    refresh: fetchDocuments,
    clearUploadError: () => setUploadError(null),
  }
}
