import { useState, useCallback } from "react"
import type { PipelineConfig } from "../types/pipeline"
import type { Citation } from "../types/document"

export interface Message {
  role:      "user" | "assistant"
  content:   string
  citations: Citation[]
}

export function useChat(documentId: string, config: PipelineConfig) {
  const [messages, setMessages]    = useState<Message[]>([])
  const [isStreaming, setStreaming] = useState(false)
  const [citations, setCitations]  = useState<Citation[]>([])
  const [error, setError]          = useState<string | null>(null)

  const sendMessage = useCallback(async (text: string) => {
    if (isStreaming || !text.trim()) return

    setError(null)
    setCitations([])

    // Add user message, then placeholder for assistant
    const userMsg: Message   = { role: "user",      content: text, citations: [] }
    const assistantMsg: Message = { role: "assistant", content: "", citations: [] }
    setMessages(prev => [...prev, userMsg, assistantMsg])
    setStreaming(true)

    // assistantIdx points to the placeholder we just pushed
    const assistantIdx = messages.length + 1

    try {
      const res = await fetch("/api/chat", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message:     text,
          document_id: documentId,
          // Send history without the placeholder we just added
          history: messages.map(m => ({ role: m.role, content: m.content })),
          config,
        }),
      })

      if (!res.ok || !res.body) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`)
      }

      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer    = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        // SSE events are separated by double newline
        const parts = buffer.split("\n\n")
        buffer = parts.pop() ?? ""

        for (const part of parts) {
          const line = part.trim()
          if (!line.startsWith("data: ")) continue
          const raw = line.slice(6)

          if (raw === "[DONE]") {
            setStreaming(false)
            return
          }

          let event: { type: string; data: unknown }
          try {
            event = JSON.parse(raw)
          } catch {
            continue
          }

          if (event.type === "context") {
            const citation = event.data as Citation
            setCitations(prev => [...prev, citation])
            // Also attach citation to the current assistant message
            setMessages(prev => {
              const next = [...prev]
              const msg  = next[assistantIdx]
              if (!msg) return prev
              next[assistantIdx] = { ...msg, citations: [...msg.citations, citation] }
              return next
            })
          } else if (event.type === "token") {
            const token = event.data as string
            setMessages(prev => {
              const next = [...prev]
              const msg  = next[assistantIdx]
              if (!msg) return prev
              next[assistantIdx] = { ...msg, content: msg.content + token }
              return next
            })
          } else if (event.type === "error") {
            throw new Error(String(event.data))
          }
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setError(msg)
      // Mark assistant placeholder as error message
      setMessages(prev => {
        const next = [...prev]
        const assistant = next[assistantIdx]
        if (assistant && !assistant.content) {
          next[assistantIdx] = { ...assistant, content: `⚠ ${msg}` }
        }
        return next
      })
    } finally {
      setStreaming(false)
    }
  }, [messages, documentId, config, isStreaming])

  const clearHistory = useCallback(() => {
    setMessages([])
    setCitations([])
    setError(null)
  }, [])

  return { messages, sendMessage, isStreaming, citations, error, clearHistory }
}
