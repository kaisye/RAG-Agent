import { useState, useEffect, useCallback } from 'react'
import type { Project } from '../types'

export function useProjects() {
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchProjects = useCallback(async () => {
    try {
      const res = await fetch('/api/projects')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setProjects(await res.json())
    } catch (e) {
      setError(String(e))
    }
  }, [])

  useEffect(() => {
    fetchProjects()
  }, [fetchProjects])

  const createProject = useCallback(async (name: string, description: string = '') => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, description }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail ?? `HTTP ${res.status}`)
      }
      const project: Project = await res.json()
      setProjects(prev => [project, ...prev])
      return project
    } catch (e) {
      setError(String(e))
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  const deleteProject = useCallback(async (id: string) => {
    try {
      await fetch(`/api/projects/${id}`, { method: 'DELETE' })
      setProjects(prev => prev.filter(p => p.id !== id))
    } catch (e) {
      setError(String(e))
    }
  }, [])

  const addDocument = useCallback(async (projectId: string, documentId: string) => {
    try {
      const res = await fetch(`/api/projects/${projectId}/documents`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_id: documentId }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setProjects(prev =>
        prev.map(p =>
          p.id === projectId && !p.document_ids.includes(documentId)
            ? { ...p, document_ids: [...p.document_ids, documentId] }
            : p
        )
      )
    } catch (e) {
      setError(String(e))
    }
  }, [])

  const removeDocument = useCallback(async (projectId: string, documentId: string) => {
    try {
      await fetch(`/api/projects/${projectId}/documents/${documentId}`, { method: 'DELETE' })
      setProjects(prev =>
        prev.map(p =>
          p.id === projectId
            ? { ...p, document_ids: p.document_ids.filter(id => id !== documentId) }
            : p
        )
      )
    } catch (e) {
      setError(String(e))
    }
  }, [])

  return {
    projects,
    loading,
    error,
    createProject,
    deleteProject,
    addDocument,
    removeDocument,
    refresh: fetchProjects,
  }
}
