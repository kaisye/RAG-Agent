import { useState } from 'react'
import type { DebugChunk, DebugInfo } from '../types'

interface Props {
  debug: DebugInfo
}

const STEP_COLORS = ['#fce4ec', '#e3f2fd', '#e8f5e9', '#fff3e0', '#f3e5f5']
const SCORE_COLORS = ['#c62828', '#1565c0', '#2e7d32', '#e65100', '#6a1b9a']

function ChunkRow({ chunk, scoreColor }: { chunk: DebugChunk; scoreColor: string }) {
  return (
    <div
      style={{
        padding: '4px 10px',
        fontSize: '11px',
        borderBottom: '1px solid #f5f5f5',
        display: 'grid',
        gridTemplateColumns: '62px 46px 1fr',
        gap: '6px',
        alignItems: 'start',
      }}
    >
      <span style={{ color: scoreColor, fontWeight: 700, fontFamily: 'monospace' }}>
        {chunk.score.toFixed(4)}
      </span>
      <span style={{ color: '#888' }}>Tr.{chunk.page + 1}</span>
      <span style={{ color: '#444', lineHeight: 1.4 }}>{chunk.snippet || '—'}</span>
    </div>
  )
}

function Step({
  index,
  title,
  badge,
  children,
}: {
  index: number
  title: string
  badge: string
  children: React.ReactNode
}) {
  const [open, setOpen] = useState(true)
  return (
    <div
      style={{
        border: '1px solid #e0e0e0',
        borderRadius: '6px',
        marginBottom: '6px',
        overflow: 'hidden',
      }}
    >
      <div
        onClick={() => setOpen(v => !v)}
        style={{
          padding: '5px 10px',
          background: STEP_COLORS[index],
          cursor: 'pointer',
          fontSize: '12px',
          fontWeight: 600,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          userSelect: 'none',
        }}
      >
        <span>{['①','②','③','④','⑤'][index]} {title}</span>
        <span style={{ fontWeight: 400, color: '#666', fontSize: '11px' }}>
          {badge} {open ? '▲' : '▼'}
        </span>
      </div>
      {open && <div style={{ background: '#fff' }}>{children}</div>}
    </div>
  )
}

export function DebugPanel({ debug }: Props) {
  const [open, setOpen] = useState(false)
  const { sub_queries, hyde_docs, vector_hits, bm25_hits, rrf_candidates, reranked, latency_ms } = debug
  const totalMs = (latency_ms.transform ?? 0) + (latency_ms.search ?? 0) + (latency_ms.rerank ?? 0)

  return (
    <div style={{ marginTop: '6px', paddingLeft: '2px' }}>
      <button
        onClick={() => setOpen(v => !v)}
        style={{
          fontSize: '11px',
          color: '#9e9e9e',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          padding: '2px 0',
          fontFamily: 'inherit',
          display: 'inline-flex',
          alignItems: 'center',
          gap: '4px',
        }}
      >
        <span style={{ fontSize: '9px' }}>{open ? '▼' : '▶'}</span>
        Pipeline debug &nbsp;·&nbsp; {totalMs}ms total
      </button>

      {open && (
        <div style={{ marginTop: '6px' }}>

          {/* Step 1 – Query Transform */}
          <Step index={0} title="Query Transform" badge={`${latency_ms.transform ?? 0}ms`}>
            <div style={{ padding: '8px 10px', fontSize: '12px' }}>
              {sub_queries.length > 1 ? (
                <>
                  <div style={{ color: '#888', fontSize: '11px', marginBottom: '4px' }}>
                    Query Decomposition → {sub_queries.length} sub-queries
                  </div>
                  {sub_queries.map((q, i) => (
                    <div key={i} style={{ color: '#333', padding: '1px 0' }}>• {q}</div>
                  ))}
                </>
              ) : (
                <div style={{ color: '#333' }}>
                  <span style={{ color: '#888', fontSize: '11px' }}>Query: </span>
                  <em>{sub_queries[0]}</em>
                </div>
              )}

              {hyde_docs.length > 0 && (
                <div style={{ marginTop: '8px' }}>
                  <div style={{ color: '#888', fontSize: '11px', marginBottom: '3px' }}>
                    HyDE — hypothetical document:
                  </div>
                  <div
                    style={{
                      color: '#555',
                      fontStyle: 'italic',
                      background: '#fafafa',
                      padding: '6px 8px',
                      borderRadius: '4px',
                      fontSize: '11px',
                      lineHeight: 1.5,
                      borderLeft: '3px solid #ef9a9a',
                    }}
                  >
                    {hyde_docs[0].slice(0, 250)}{hyde_docs[0].length > 250 ? '…' : ''}
                  </div>
                </div>
              )}

              {sub_queries.length === 1 && hyde_docs.length === 0 && (
                <div style={{ color: '#bbb', fontSize: '11px', marginTop: '4px' }}>
                  HyDE: off &nbsp;·&nbsp; Query Decomposition: off
                </div>
              )}
            </div>
          </Step>

          {/* Step 2 – Vector Search */}
          <Step index={1} title="Vector Search (embedding)" badge={`${vector_hits.length} hits`}>
            {vector_hits.length === 0 ? (
              <div style={{ padding: '8px 10px', color: '#aaa', fontSize: '12px' }}>No results</div>
            ) : (
              vector_hits.map((h, i) => (
                <ChunkRow key={i} chunk={h} scoreColor={SCORE_COLORS[1]} />
              ))
            )}
          </Step>

          {/* Step 3 – BM25 */}
          <Step
            index={2}
            title="BM25 Search (keyword)"
            badge={`${bm25_hits.length} hits · ${latency_ms.search ?? 0}ms`}
          >
            {bm25_hits.length === 0 ? (
              <div style={{ padding: '8px 10px', color: '#aaa', fontSize: '12px' }}>
                Disabled or no keyword matches
              </div>
            ) : (
              bm25_hits.map((h, i) => (
                <ChunkRow key={i} chunk={h} scoreColor={SCORE_COLORS[2]} />
              ))
            )}
          </Step>

          {/* Step 4 – RRF Merge */}
          <Step index={3} title="RRF Merge" badge={`${rrf_candidates.length} candidates`}>
            {rrf_candidates.length === 0 ? (
              <div style={{ padding: '8px 10px', color: '#aaa', fontSize: '12px' }}>No candidates</div>
            ) : (
              rrf_candidates.map((h, i) => (
                <ChunkRow key={i} chunk={h} scoreColor={SCORE_COLORS[3]} />
              ))
            )}
          </Step>

          {/* Step 5 – Rerank */}
          <Step
            index={4}
            title="Rerank (cross-encoder)"
            badge={`${reranked.length} final · ${latency_ms.rerank ?? 0}ms`}
          >
            {reranked.length === 0 ? (
              <div style={{ padding: '8px 10px', color: '#aaa', fontSize: '12px' }}>No results</div>
            ) : (
              reranked.map((h, i) => (
                <ChunkRow key={i} chunk={h} scoreColor={SCORE_COLORS[4]} />
              ))
            )}
          </Step>

        </div>
      )}
    </div>
  )
}
