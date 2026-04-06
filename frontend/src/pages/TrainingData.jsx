import { useState, useMemo, Fragment } from 'react'
import { Database, ChevronDown, ChevronLeft, ChevronRight, ChevronUp, FileText, Eye } from 'lucide-react'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'
import { useTrainingDatasets, useTrainingData } from '../api/queries'

const ACCENT_COLORS = {
  south_asian: '#7bd0ff',
  american: '#4edea3',
  european: '#ffb95f',
  unknown: '#45464d',
}

const ACCENT_LABELS = {
  south_asian: 'South Asian',
  american: 'American',
  european: 'European',
  unknown: 'Unknown',
}

const DATASET_DESCRIPTIONS = {
  collected_data: 'Main collected training corpus',
  training_pairs: 'Curated positive/negative pairs',
  hard_negatives: 'Difficult negative examples',
  live_corrections: 'Real-time user corrections',
}

function DatasetCard({ dataset, isSelected, onClick }) {
  const name = dataset.name || dataset.dataset || 'unknown'
  const total = dataset.total || dataset.count || 0
  const files = dataset.files || []

  return (
    <button
      onClick={onClick}
      className={`flux-card rounded-xl p-5 text-left transition-all w-full ${
        isSelected
          ? 'ring-2 ring-primary shadow-[0_0_20px_rgba(123,208,255,0.15)]'
          : 'hover:bg-surface-container'
      }`}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <Database size={14} className={isSelected ? 'text-primary' : 'text-on-surface-variant'} />
          <span className="font-label text-xs font-bold uppercase tracking-wider text-on-surface">
            {name.replace(/_/g, ' ')}
          </span>
        </div>
        {isSelected && (
          <span className="text-[9px] font-label uppercase text-primary bg-primary/10 px-2 py-0.5 rounded-full">
            Selected
          </span>
        )}
      </div>
      <p className="font-headline text-2xl font-extrabold text-on-surface mb-1">
        {total.toLocaleString()}
      </p>
      <p className="text-[10px] text-on-surface-variant font-label mb-3">
        {DATASET_DESCRIPTIONS[name] || 'Training dataset'}
      </p>
      {files.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {files.map((f, i) => (
            <span
              key={typeof f === 'string' ? f : f.name || i}
              className="text-[9px] font-label bg-surface-container-high text-on-surface-variant px-2 py-0.5 rounded-md"
            >
              {typeof f === 'string' ? f : f.name}
            </span>
          ))}
        </div>
      )}
    </button>
  )
}

function AccentPieChart({ metadata }) {
  const accents = metadata?.accent_distribution || metadata?.accents || {}
  const data = Object.entries(accents).map(([key, value]) => ({
    name: ACCENT_LABELS[key] || key,
    value: typeof value === 'number' ? value : 0,
    color: ACCENT_COLORS[key] || '#45464d',
  }))

  if (data.length === 0) return null

  return (
    <div className="flux-card rounded-xl p-5">
      <h4 className="font-label text-[10px] uppercase tracking-widest text-on-surface-variant mb-4">
        Accent Distribution
      </h4>
      <div className="flex items-center gap-6">
        <div className="w-32 h-32">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                innerRadius={28}
                outerRadius={52}
                strokeWidth={0}
              >
                {data.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: '#131b2e',
                  border: '1px solid rgba(144,144,151,0.2)',
                  borderRadius: '8px',
                  fontSize: '11px',
                  color: '#dae2fd',
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="space-y-2 flex-1">
          {data.map((d) => (
            <div key={d.name} className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: d.color }} />
              <span className="text-[10px] font-label text-on-surface-variant flex-1">{d.name}</span>
              <span className="text-[10px] font-label font-bold text-on-surface">
                {d.value.toLocaleString()}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function RatioBar({ metadata }) {
  const positive = metadata?.positive_count ?? metadata?.positive ?? 0
  const negative = metadata?.negative_count ?? metadata?.negative ?? 0
  const total = positive + negative

  if (total === 0) return null

  const posPct = Math.round((positive / total) * 100)
  const negPct = 100 - posPct

  return (
    <div className="flux-card rounded-xl p-5">
      <h4 className="font-label text-[10px] uppercase tracking-widest text-on-surface-variant mb-4">
        Positive / Negative Ratio
      </h4>
      <div className="flex items-center gap-4 mb-3">
        <span className="text-xs font-label text-secondary font-bold">{positive.toLocaleString()} pos</span>
        <span className="text-xs font-label text-error font-bold">{negative.toLocaleString()} neg</span>
      </div>
      <div className="w-full h-3 bg-surface-container-highest rounded-full overflow-hidden flex">
        <div
          className="h-full bg-secondary transition-all duration-300"
          style={{ width: `${posPct}%` }}
        />
        <div
          className="h-full bg-error transition-all duration-300"
          style={{ width: `${negPct}%` }}
        />
      </div>
      <div className="flex justify-between mt-2">
        <span className="text-[10px] font-label text-on-surface-variant">{posPct}% positive</span>
        <span className="text-[10px] font-label text-on-surface-variant">{negPct}% negative</span>
      </div>
    </div>
  )
}

function ExpandedRow({ row }) {
  const messages = row.messages || []

  return (
    <tr>
      <td colSpan={8} className="p-0">
        <div className="bg-surface-container-lowest p-4 mx-2 mb-2 rounded-lg border border-outline-variant/20">
          <p className="font-label text-[10px] uppercase tracking-widest text-on-surface-variant mb-3">
            Messages Content
          </p>
          {messages.length > 0 ? (
            <div className="space-y-3">
              {messages.map((msg, i) => (
                <div key={i} className="flex gap-3">
                  <span
                    className={`font-label text-[10px] uppercase font-bold shrink-0 w-16 ${
                      msg.role === 'system'
                        ? 'text-tertiary'
                        : msg.role === 'user'
                          ? 'text-primary'
                          : 'text-secondary'
                    }`}
                  >
                    {msg.role}
                  </span>
                  <pre className="text-xs text-on-surface-variant font-mono whitespace-pre-wrap leading-relaxed flex-1 overflow-hidden">
                    {msg.content}
                  </pre>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-on-surface-variant font-mono">
              {row.text || row.content || JSON.stringify(row, null, 2)}
            </p>
          )}
        </div>
      </td>
    </tr>
  )
}

export default function TrainingData() {
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [selectedFile, setSelectedFile] = useState('')
  const [page, setPage] = useState(1)
  const [expandedRow, setExpandedRow] = useState(null)
  const limit = 20

  const { data: datasets, isLoading: datasetsLoading } = useTrainingDatasets()

  const selectedInfo = useMemo(() => {
    if (!selectedDataset || !datasets) return null
    return (Array.isArray(datasets) ? datasets : datasets?.datasets || []).find(
      (d) => (d.name || d.dataset) === selectedDataset
    )
  }, [selectedDataset, datasets])

  const files = useMemo(() => {
    return selectedInfo?.files || []
  }, [selectedInfo])

  const { data: trainingData, isLoading: dataLoading } = useTrainingData(
    selectedDataset
      ? { dataset: selectedDataset, file: selectedFile || undefined, page, limit }
      : {}
  )

  const datasetsList = useMemo(() => {
    if (!datasets) return []
    return Array.isArray(datasets) ? datasets : datasets?.datasets || []
  }, [datasets])

  const examples = useMemo(() => {
    if (!trainingData) return []
    return Array.isArray(trainingData) ? trainingData : trainingData?.examples || trainingData?.data || []
  }, [trainingData])

  const totalPages = useMemo(() => {
    if (!trainingData) return 1
    const total = trainingData.total || trainingData.total_count || 0
    return Math.max(1, Math.ceil(total / limit))
  }, [trainingData, limit])

  const metadata = selectedInfo?.metadata || selectedInfo?.stats || null

  const handleSelectDataset = (name) => {
    setSelectedDataset(name)
    setSelectedFile('')
    setPage(1)
    setExpandedRow(null)
  }

  return (
    <div>
      {/* Header */}
      <header className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <Database size={20} className="text-primary" />
          <h2 className="font-headline text-3xl font-extrabold text-on-surface tracking-tight">
            Training Data
          </h2>
        </div>
        <p className="text-on-surface-variant font-body max-w-xl">
          Browse training datasets, inspect examples, and analyze data distribution across accents and categories.
        </p>
      </header>

      {/* Dataset Cards */}
      <section className="mb-8">
        <h3 className="font-label text-[10px] uppercase tracking-widest text-on-surface-variant mb-4">
          Available Datasets
        </h3>
        {datasetsLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="flux-card rounded-xl p-5 animate-pulse">
                <div className="h-4 bg-surface-container-high rounded w-24 mb-3" />
                <div className="h-8 bg-surface-container-high rounded w-16 mb-2" />
                <div className="h-3 bg-surface-container-high rounded w-32" />
              </div>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {datasetsList.map((ds) => {
              const name = ds.name || ds.dataset || ''
              return (
                <DatasetCard
                  key={name}
                  dataset={ds}
                  isSelected={selectedDataset === name}
                  onClick={() => handleSelectDataset(name)}
                />
              )
            })}
          </div>
        )}
      </section>

      {/* Stats Panel */}
      {selectedDataset && metadata && (
        <section className="mb-8 grid grid-cols-1 md:grid-cols-2 gap-4">
          <AccentPieChart metadata={metadata} />
          <RatioBar metadata={metadata} />
        </section>
      )}

      {/* Examples Table */}
      {selectedDataset && (
        <section>
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-4">
            <h3 className="font-headline text-lg font-bold text-on-surface flex items-center gap-2">
              <FileText size={16} className="text-primary" />
              Examples
              {dataLoading && (
                <span className="text-[10px] font-label text-on-surface-variant ml-2">Loading...</span>
              )}
            </h3>
            {files.length > 0 && (
              <div className="relative">
                <select
                  value={selectedFile}
                  onChange={(e) => {
                    setSelectedFile(e.target.value)
                    setPage(1)
                    setExpandedRow(null)
                  }}
                  className="appearance-none bg-surface-container-high text-on-surface text-xs font-label pl-3 pr-8 py-2 rounded-lg border border-outline-variant/30 focus:outline-none focus:ring-1 focus:ring-primary cursor-pointer"
                >
                  <option value="">All Files</option>
                  {files.map((f, i) => {
                    const name = typeof f === 'string' ? f : f.name
                    return (
                      <option key={name || i} value={name}>
                        {name}
                      </option>
                    )
                  })}
                </select>
                <ChevronDown
                  size={14}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-on-surface-variant pointer-events-none"
                />
              </div>
            )}
          </div>

          <div className="flux-card rounded-xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest border-b border-outline-variant/20">
                    <th className="py-3 px-4 w-10">#</th>
                    <th className="py-3 px-4">Source</th>
                    <th className="py-3 px-4">Accent</th>
                    <th className="py-3 px-4">Term</th>
                    <th className="py-3 px-4">Category</th>
                    <th className="py-3 px-4">Error Found</th>
                    <th className="py-3 px-4">Applied</th>
                    <th className="py-3 px-4 w-10" />
                  </tr>
                </thead>
                <tbody className="text-sm font-body">
                  {examples.length === 0 && !dataLoading ? (
                    <tr>
                      <td colSpan={8} className="py-12 text-center text-on-surface-variant text-sm">
                        No examples found for this dataset.
                      </td>
                    </tr>
                  ) : (
                    examples.map((row, idx) => {
                      const rowIdx = (page - 1) * limit + idx + 1
                      const isExpanded = expandedRow === idx
                      const applied = row.applied ?? row.is_positive ?? null
                      const hasOcr = row.has_ocr || row.ocr_context

                      return (
                        <Fragment key={row._id || idx}>
                          <tr
                            onClick={() => setExpandedRow(isExpanded ? null : idx)}
                            className={`cursor-pointer transition-colors border-b border-outline-variant/10 ${
                              isExpanded
                                ? 'bg-surface-container-high'
                                : 'hover:bg-surface-container-high/50'
                            }`}
                          >
                            <td className="py-3 px-4 text-on-surface-variant text-xs font-label">
                              {rowIdx}
                            </td>
                            <td className="py-3 px-4 text-on-surface text-xs font-label truncate max-w-[140px]">
                              {row.source || row.video_id || row.file_id || '\u2014'}
                            </td>
                            <td className="py-3 px-4">
                              <span className="text-[10px] font-label text-on-surface-variant">
                                {row.accent || '\u2014'}
                              </span>
                            </td>
                            <td className="py-3 px-4">
                              <span className="text-xs font-label text-primary font-medium">
                                {row.term || row.correction_term || '\u2014'}
                              </span>
                            </td>
                            <td className="py-3 px-4">
                              <span className="text-[10px] font-label text-on-surface-variant">
                                {row.category || '\u2014'}
                              </span>
                            </td>
                            <td className="py-3 px-4">
                              <span className="text-xs font-mono text-on-surface-variant truncate max-w-[160px] block">
                                {row.error_found || row.whisper_error || '\u2014'}
                              </span>
                            </td>
                            <td className="py-3 px-4">
                              <div className="flex gap-1.5">
                                {applied === true || applied === 1 ? (
                                  <span className="text-[9px] font-label font-bold uppercase px-2 py-0.5 rounded-full bg-secondary/15 text-secondary">
                                    Applied
                                  </span>
                                ) : applied === false || applied === 0 ? (
                                  <span className="text-[9px] font-label font-bold uppercase px-2 py-0.5 rounded-full bg-error/15 text-error">
                                    Negative
                                  </span>
                                ) : null}
                                {hasOcr && (
                                  <span className="text-[9px] font-label font-bold uppercase px-2 py-0.5 rounded-full bg-primary/15 text-primary">
                                    OCR
                                  </span>
                                )}
                              </div>
                            </td>
                            <td className="py-3 px-4">
                              {isExpanded ? (
                                <ChevronUp size={14} className="text-primary" />
                              ) : (
                                <Eye size={14} className="text-on-surface-variant" />
                              )}
                            </td>
                          </tr>
                          {isExpanded && <ExpandedRow row={row} />}
                        </Fragment>
                      )
                    })
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between px-4 py-3 border-t border-outline-variant/20">
                <button
                  onClick={() => {
                    setPage((p) => Math.max(1, p - 1))
                    setExpandedRow(null)
                  }}
                  disabled={page <= 1}
                  className="flex items-center gap-1 text-xs font-label text-on-surface-variant hover:text-primary disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronLeft size={14} />
                  Previous
                </button>
                <span className="text-[10px] font-label text-on-surface-variant">
                  Page {page} of {totalPages}
                </span>
                <button
                  onClick={() => {
                    setPage((p) => Math.min(totalPages, p + 1))
                    setExpandedRow(null)
                  }}
                  disabled={page >= totalPages}
                  className="flex items-center gap-1 text-xs font-label text-on-surface-variant hover:text-primary disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  Next
                  <ChevronRight size={14} />
                </button>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Empty state */}
      {!selectedDataset && !datasetsLoading && (
        <div className="flux-card rounded-xl p-12 text-center">
          <Database size={32} className="text-on-surface-variant mx-auto mb-4 opacity-40" />
          <p className="text-on-surface-variant font-body text-sm">
            Select a dataset above to browse training examples.
          </p>
        </div>
      )}
    </div>
  )
}

