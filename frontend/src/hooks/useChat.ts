import { useState, useCallback, useRef } from 'react'
import type { Citation, DebugInfo, Message } from '../types'

function uuid() {
  return crypto.randomUUID()
}

export interface HistoryTurn {
  role: string
  content: string
}

interface ChatOptions {
  documentId?: string
  projectId?: string
  maxHistoryTurns?: number   // 0 = unlimited (default)
}

export function useChat({ documentId, projectId, maxHistoryTurns = 0 }: ChatOptions) {
  const [messages, setMessages] = useState<Message[]>([])
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastHistory, setLastHistory] = useState<HistoryTurn[]>([])
  const abortRef = useRef<AbortController | null>(null)

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || streaming) return
    setError(null)

    const userMsg: Message = { id: uuid(), role: 'user', content: text }
    const assistantId = uuid()
    const assistantMsg: Message = { id: assistantId, role: 'assistant', content: '', citations: [] }

    setMessages(prev => [...prev, userMsg, assistantMsg])
    setStreaming(true)

    const allHistory = messages.map(m => ({ role: m.role, content: m.content }))
    // Slice to maxHistoryTurns pairs (1 pair = 1 user + 1 assistant msg = 2 items)
    const history = maxHistoryTurns > 0
      ? allHistory.slice(-(maxHistoryTurns * 2))
      : allHistory
    setLastHistory(history)
    abortRef.current = new AbortController()

    const body: Record<string, unknown> = { message: text, history }
    if (projectId) body.project_id = projectId
    else if (documentId) body.document_id = documentId

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: abortRef.current.signal,
      })

      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail ?? `HTTP ${res.status}`)
      }

      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const payload = line.slice(6)
          if (payload === '[DONE]') break

          try {
            const parsed = JSON.parse(payload)

            if (parsed.error) {
              setError(parsed.error)
              break
            }

            if (parsed.delta) {
              // Append streaming token
              setMessages(prev =>
                prev.map(m =>
                  m.id === assistantId
                    ? { ...m, content: m.content + parsed.delta }
                    : m
                )
              )
            }

            if (parsed.citations) {
              setMessages(prev =>
                prev.map(m =>
                  m.id === assistantId
                    ? { ...m, citations: parsed.citations as Citation[] }
                    : m
                )
              )
            }

            if (parsed.debug) {
              setMessages(prev =>
                prev.map(m =>
                  m.id === assistantId
                    ? { ...m, debug: parsed.debug as DebugInfo }
                    : m
                )
              )
            }
          } catch {
            // ignore malformed SSE line
          }
        }
      }
    } catch (e: unknown) {
      if ((e as Error).name === 'AbortError') return
      setError(String(e))
      setMessages(prev => prev.filter(m => m.id !== assistantId || m.content))
    } finally {
      setStreaming(false)
    }
  }, [messages, documentId, projectId, streaming, maxHistoryTurns])

  const stop = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const clear = useCallback(() => {
    setMessages([])
    setError(null)
  }, [])

  return { messages, streaming, error, lastHistory, sendMessage, stop, clear }
}
