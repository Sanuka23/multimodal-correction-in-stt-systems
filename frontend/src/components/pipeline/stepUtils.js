/**
 * Canonical execution order. Steps emitted out-of-order in MongoDB
 * are sorted by this list. Anything not in the list is appended.
 */
export const STEP_ORDER = [
  'request_received',
  'model_load',
  'vocab_merge',
  'candidate_detection',
  'topic_classification',
  'web_vocab_enrichment',
  'candidate_validation',
  'ocr_extraction',
  'ocr_vocab_extraction',
  'whisper_pass2',
  'avsr_extraction',
  'avsr_pass2',
  'llm_reconciliation',
  'ml_inference',
  'data_collection',
  'apply_corrections',
  'complete',
]

const STATUS_RANK = {
  pending: 0,
  running: 1,
  skipped: 2,
  failed: 3,
  completed: 4,
}

/** Coalesce duplicate `name` rows; keep the most-progressed status. */
export function dedupeStepsByName(steps) {
  if (!Array.isArray(steps) || steps.length === 0) return []
  const byName = new Map()
  for (const s of steps) {
    if (!s || !s.name) continue
    const prev = byName.get(s.name)
    if (!prev) {
      byName.set(s.name, s)
      continue
    }
    const rPrev = STATUS_RANK[prev.status] ?? 0
    const rNew = STATUS_RANK[s.status] ?? 0
    if (rNew > rPrev) {
      byName.set(s.name, s)
    } else if (rNew === rPrev) {
      const enrich = (x) =>
        (x.duration_ms != null ? 1 : 0) +
        (x.details && Object.keys(x.details).length ? 1 : 0)
      if (enrich(s) > enrich(prev)) byName.set(s.name, s)
    }
  }
  return [...byName.values()]
}

/** Dedup + canonical sort. Used by both PipelineFlow and StepTimingsPanel. */
export function canonicalSteps(steps) {
  const deduped = dedupeStepsByName(steps)
  if (deduped.length <= 1) return deduped
  const orderIndex = new Map(STEP_ORDER.map((name, i) => [name, i]))
  return deduped.sort((a, b) => {
    const ai = orderIndex.has(a.name) ? orderIndex.get(a.name) : 1000
    const bi = orderIndex.has(b.name) ? orderIndex.get(b.name) : 1000
    return ai - bi
  })
}
