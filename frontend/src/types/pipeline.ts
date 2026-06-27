export type ChunkingStrategy  = "recursive" | "semantic"
export type RetrievalStrategy = "vector" | "bm25" | "hybrid_interleaving" | "hybrid_rrf"
export type QueryTransform    = "none" | "hyde" | "decomposition"
export type RerankStrategy    = "none" | "cross_encoder" | "mmr"

export interface PipelineConfig {
  chunking_strategy:  ChunkingStrategy
  retrieval_strategy: RetrievalStrategy
  query_transform:    QueryTransform
  rerank_strategy:    RerankStrategy
  top_k_retrieval:    number
  rrf_k:              number
  top_k_final:        number
  mmr_lambda:         number
}

export const DEFAULT_CONFIG: PipelineConfig = {
  chunking_strategy:  "recursive",
  retrieval_strategy: "vector",
  query_transform:    "none",
  rerank_strategy:    "none",
  top_k_retrieval:    10,
  rrf_k:              60,
  top_k_final:        3,
  mmr_lambda:         0.5,
}
