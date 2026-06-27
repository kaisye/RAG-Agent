import type { PipelineConfig } from "./pipeline"

export interface RagasScores {
  faithfulness:      number
  answer_relevancy:  number
  context_precision: number
  context_recall:    number
}

export interface AblationResult {
  config:      PipelineConfig
  scores:      RagasScores
  num_samples: number
  document_id?: string
}

export interface AblationRow {
  "Thực nghiệm":       string
  Faithfulness:        number
  "Δ Faith":           number
  "Answer Relevancy":  number
  "Δ AR":              number
  "Context Precision": number
  "Δ CP":              number
  "Context Recall":    number
  "Δ CR":              number
}

export interface AblationStatus {
  running:        boolean
  done:           boolean
  error:          string
  current_config: number
  total_configs:  number
  current_label:  string
  results:        RagasScores[]
}
