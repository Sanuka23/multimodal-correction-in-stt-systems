import { useState, useRef, useEffect, useCallback } from 'react'
import { Pencil, Upload, Save, Play, Pause, SkipBack, Download, Check, X, Volume2, Link, Loader } from 'lucide-react'
import api from '../api/client'
import PageHeader from '../components/ui/PageHeader'

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function fmtTime(s) {
  if (!s && s !== 0) return '0:00'
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}

/* ------------------------------------------------------------------ */
/*  Word component — clickable, highlights when active                 */
/* ------------------------------------------------------------------ */

function Word({ word, isActive, isEdited, onClick }) {
  return (
    <span
      onClick={onClick}
      className={`
        inline cursor-pointer px-[1px] py-[1px] rounded transition-all
        ${isActive ? 'bg-primary/30 text-primary font-semibold' : ''}
        ${isEdited ? 'text-secondary underline decoration-secondary/40' : ''}
        ${!isActive && !isEdited ? 'hover:bg-surface-container-high' : ''}
      `}
    >
      {word.corrected || word.word}{' '}
    </span>
  )
}

/* ================================================================== */
/*  Annotate Page                                                      */
/* ================================================================== */

export default function Annotate() {
  // File state
  const [videoUrl, setVideoUrl] = useState(null)
  const [transcript, setTranscript] = useState(null) // { segments, text }
  const [fileName, setFileName] = useState('')

  // ScreenApp URL mode
  const [screenappUrl, setScreenappUrl] = useState('')
  const [loadingUrl, setLoadingUrl] = useState(false)
  const [loadError, setLoadError] = useState('')

  const loadFromScreenApp = async () => {
    // Extract file_id from URL like http://localhost:8080/#/library/.../default/FILE_ID
    const url = screenappUrl.trim()
    let fileId = url
    // Try to extract from full URL
    const match = url.match(/\/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:\?|$|#)*/i)
    if (match) fileId = match[1]
    // Or it might be just the last path segment
    const parts = url.split('/')
    const lastPart = parts[parts.length - 1]?.split('?')[0]
    if (lastPart?.match(/^[0-9a-f]{8}-/i)) fileId = lastPart

    if (!fileId) { setLoadError('Could not extract file ID from URL'); return }

    setLoadingUrl(true)
    setLoadError('')
    try {
      const res = await api.get(`/api/annotate/file/${fileId}`)
      const data = res.data
      if (data.video_url) setVideoUrl(data.video_url)
      if (data.transcript) {
        setTranscript(data.transcript)
        setEditCount(0)
      }
      setFileName(data.name || fileId)
      if (!data.transcript) setLoadError('Transcript not available — upload JSON manually')
      if (!data.video_url) setLoadError('Video URL not available — upload video manually')
    } catch (err) {
      setLoadError(err.response?.data?.detail || 'Failed to load file')
    } finally {
      setLoadingUrl(false)
    }
  }

  // Player state
  const videoRef = useRef(null)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)

  // Edit state
  const [editingWord, setEditingWord] = useState(null) // { segIdx, wordIdx }
  const [editValue, setEditValue] = useState('')
  const [editCount, setEditCount] = useState(0)

  // Flat word list with segment/word indices for fast lookup
  const allWords = transcript ? transcript.segments.flatMap((seg, si) =>
    (seg.words || []).map((w, wi) => ({ ...w, segIdx: si, wordIdx: wi }))
  ) : []

  // Find current word based on playback time
  const activeWordKey = allWords.findIndex(
    w => currentTime >= w.start && currentTime <= (w.end || w.start + 0.3)
  )

  // Auto-scroll to active word
  const activeRef = useRef(null)
  useEffect(() => {
    if (activeRef.current && playing) {
      activeRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }, [activeWordKey, playing])

  // Time update loop
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    const onTime = () => setCurrentTime(video.currentTime)
    const onDur = () => setDuration(video.duration)
    video.addEventListener('timeupdate', onTime)
    video.addEventListener('loadedmetadata', onDur)
    return () => {
      video.removeEventListener('timeupdate', onTime)
      video.removeEventListener('loadedmetadata', onDur)
    }
  }, [videoUrl])

  /* --- File upload handlers --- */

  const handleVideoUpload = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setVideoUrl(URL.createObjectURL(file))
    setFileName(file.name)
  }

  const handleTranscriptUpload = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target.result)
        setTranscript(data)
        setEditCount(0)
      } catch { alert('Invalid JSON') }
    }
    reader.readAsText(file)
  }

  /* --- Playback controls --- */

  const togglePlay = () => {
    const v = videoRef.current
    if (!v) return
    if (v.paused) { v.play(); setPlaying(true) }
    else { v.pause(); setPlaying(false) }
  }

  const seekTo = (time) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time
      setCurrentTime(time)
    }
  }

  /* --- Word click → pause & edit --- */

  const handleWordClick = useCallback((segIdx, wordIdx, word) => {
    // Pause playback
    if (videoRef.current && !videoRef.current.paused) {
      videoRef.current.pause()
      setPlaying(false)
    }
    // Seek to word start
    seekTo(word.start)
    // Open editor
    setEditingWord({ segIdx, wordIdx })
    setEditValue(word.corrected || word.word)
  }, [])

  const saveEdit = () => {
    if (!editingWord || !transcript) return
    const { segIdx, wordIdx } = editingWord
    const newTranscript = { ...transcript, segments: [...transcript.segments] }
    const seg = { ...newTranscript.segments[segIdx] }
    const words = [...seg.words]
    const originalWord = words[wordIdx].word
    const newWord = editValue.trim()

    if (newWord && newWord !== originalWord) {
      words[wordIdx] = { ...words[wordIdx], corrected: newWord }
      setEditCount(c => c + 1)
    } else if (newWord === originalWord) {
      // Revert correction
      const { corrected, ...rest } = words[wordIdx]
      words[wordIdx] = rest
    }
    seg.words = words
    // Rebuild segment text
    seg.corrected_text = words.map(w => w.corrected || w.word).join(' ')
    newTranscript.segments[segIdx] = seg
    setTranscript(newTranscript)
    setEditingWord(null)
  }

  const cancelEdit = () => setEditingWord(null)

  /* --- Export corrected transcript --- */

  const exportCorrected = () => {
    if (!transcript) return
    const output = {
      original_file: fileName,
      exported_at: new Date().toISOString(),
      total_edits: editCount,
      segments: transcript.segments.map(seg => ({
        id: seg.id,
        start: seg.start,
        end: seg.end,
        speaker: seg.speaker,
        original_text: seg.text,
        corrected_text: seg.corrected_text || seg.text,
        words: (seg.words || []).map(w => ({
          word: w.word,
          corrected: w.corrected || null,
          start: w.start,
          end: w.end,
          speaker: w.speaker,
        })).filter(w => w.corrected) // only include edited words
      })).filter(seg => seg.words.length > 0) // only segments with edits
    }
    const blob = new Blob([JSON.stringify(output, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `corrected_${fileName || 'transcript'}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  /* --- Export as training pairs JSONL --- */

  const exportTrainingPairs = () => {
    if (!transcript) return
    const lines = []
    for (const seg of transcript.segments) {
      const editedWords = (seg.words || []).filter(w => w.corrected)
      if (editedWords.length === 0) continue
      for (const w of editedWords) {
        lines.push(JSON.stringify({
          original: w.word,
          corrected: w.corrected,
          context: seg.text,
          corrected_context: seg.corrected_text || seg.text,
          start: w.start,
          end: w.end,
          speaker: w.speaker,
          source: fileName,
        }))
      }
    }
    if (lines.length === 0) { alert('No edits to export'); return }
    const blob = new Blob([lines.join('\n')], { type: 'application/jsonl' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `training_pairs_${fileName || 'transcript'}.jsonl`
    a.click()
    URL.revokeObjectURL(url)
  }

  /* ---------------------------------------------------------------- */
  /*  Render                                                           */
  /* ---------------------------------------------------------------- */

  return (
    <div>
      <PageHeader
        eyebrow="Transcript Annotator"
        title="Annotate & Correct"
        description="Play audio, click any word to fix the transcript inline, and export ground-truth pairs for training."
        icon={Pencil}
      />

      {/* Load from ScreenApp URL */}
      {!transcript && (
        <div className="flux-card rounded-xl p-6 mb-6">
          <div className="flex items-center gap-2 mb-3">
            <Link size={16} className="text-primary" />
            <h3 className="text-sm font-medium text-on-surface">Load from ScreenApp</h3>
          </div>
          <div className="flex gap-3">
            <input
              value={screenappUrl}
              onChange={(e) => setScreenappUrl(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && loadFromScreenApp()}
              placeholder="Paste ScreenApp URL or file ID (e.g. 2a34dfca-6593-487d-...)"
              className="flex-1 bg-surface-container-high text-on-surface text-sm px-4 py-2.5 rounded-lg border border-outline-variant/30 focus:outline-none focus:ring-1 focus:ring-primary placeholder:text-on-surface-variant/50"
            />
            <button
              onClick={loadFromScreenApp}
              disabled={loadingUrl || !screenappUrl.trim()}
              className="px-5 py-2.5 bg-primary text-on-primary text-sm font-label font-semibold rounded-lg hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {loadingUrl ? <Loader size={14} className="animate-spin" /> : <Link size={14} />}
              Load
            </button>
          </div>
          {loadError && <p className="text-xs text-error mt-2">{loadError}</p>}

          <div className="flex items-center gap-3 my-4">
            <div className="flex-1 h-px bg-outline-variant/20" />
            <span className="text-[10px] text-on-surface-variant font-label uppercase">or upload files</span>
            <div className="flex-1 h-px bg-outline-variant/20" />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Video upload */}
            <label className="flex flex-col items-center justify-center p-8 border-2 border-dashed border-outline-variant/30 rounded-xl cursor-pointer hover:border-primary/50 transition-colors">
              <Upload size={32} className="text-on-surface-variant mb-3" />
              <p className="text-sm font-medium text-on-surface">Upload Video/Audio</p>
              <p className="text-xs text-on-surface-variant mt-1">MP4, WebM, MP3, WAV</p>
              {videoUrl && <p className="text-xs text-secondary mt-2">Loaded</p>}
              <input type="file" accept="video/*,audio/*" onChange={handleVideoUpload} className="hidden" />
            </label>

            {/* Transcript upload */}
            <label className="flex flex-col items-center justify-center p-8 border-2 border-dashed border-outline-variant/30 rounded-xl cursor-pointer hover:border-primary/50 transition-colors">
              <Pencil size={32} className="text-on-surface-variant mb-3" />
              <p className="text-sm font-medium text-on-surface">Upload Transcript JSON</p>
              <p className="text-xs text-on-surface-variant mt-1">ScreenApp format with word timestamps</p>
              <input type="file" accept=".json" onChange={handleTranscriptUpload} className="hidden" />
            </label>
          </div>
        </div>
      )}

      {/* Main workspace */}
      {transcript && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Left: Video Player + Controls */}
          <div className="lg:col-span-1">
            <div className="flux-card rounded-xl overflow-hidden sticky top-4">
              {/* Video */}
              {videoUrl ? (
                <video
                  ref={videoRef}
                  src={videoUrl}
                  className="w-full aspect-video bg-black"
                  onClick={togglePlay}
                />
              ) : (
                <div className="w-full aspect-video bg-surface-container flex items-center justify-center">
                  <label className="cursor-pointer text-center">
                    <Upload size={24} className="mx-auto text-on-surface-variant mb-2" />
                    <p className="text-xs text-on-surface-variant">Upload video</p>
                    <input type="file" accept="video/*,audio/*" onChange={handleVideoUpload} className="hidden" />
                  </label>
                </div>
              )}

              {/* Controls */}
              <div className="p-4">
                {/* Progress bar */}
                <div
                  className="h-1.5 bg-surface-container-high rounded-full mb-3 cursor-pointer"
                  onClick={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect()
                    const pct = (e.clientX - rect.left) / rect.width
                    seekTo(pct * duration)
                  }}
                >
                  <div
                    className="h-full bg-primary rounded-full transition-all"
                    style={{ width: `${duration ? (currentTime / duration) * 100 : 0}%` }}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <button onClick={() => seekTo(Math.max(0, currentTime - 5))} className="p-1.5 rounded-lg hover:bg-surface-container-high text-on-surface-variant">
                      <SkipBack size={16} />
                    </button>
                    <button onClick={togglePlay} className="p-2 rounded-lg bg-primary/10 text-primary hover:bg-primary/20">
                      {playing ? <Pause size={18} /> : <Play size={18} />}
                    </button>
                  </div>
                  <span className="font-mono text-xs text-on-surface-variant">
                    {fmtTime(currentTime)} / {fmtTime(duration)}
                  </span>
                </div>

                {/* Playback speed */}
                <div className="flex gap-1 mt-3">
                  {[0.5, 0.75, 1, 1.25, 1.5].map(rate => (
                    <button
                      key={rate}
                      onClick={() => { if (videoRef.current) videoRef.current.playbackRate = rate }}
                      className="text-[10px] px-2 py-1 rounded bg-surface-container-high text-on-surface-variant hover:text-primary"
                    >
                      {rate}x
                    </button>
                  ))}
                </div>
              </div>

              {/* Stats */}
              <div className="px-4 pb-4 border-t border-outline-variant/10 pt-3">
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div>
                    <p className="text-[10px] text-on-surface-variant font-label">SEGMENTS</p>
                    <p className="text-sm font-mono text-on-surface">{transcript.segments.length}</p>
                  </div>
                  <div>
                    <p className="text-[10px] text-on-surface-variant font-label">WORDS</p>
                    <p className="text-sm font-mono text-on-surface">{allWords.length}</p>
                  </div>
                  <div>
                    <p className="text-[10px] text-on-surface-variant font-label">EDITS</p>
                    <p className="text-sm font-mono text-secondary">{editCount}</p>
                  </div>
                </div>
              </div>

              {/* Export buttons */}
              <div className="px-4 pb-4 space-y-2">
                <button
                  onClick={exportCorrected}
                  disabled={editCount === 0}
                  className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-secondary/10 text-secondary text-xs font-label hover:bg-secondary/20 disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  <Save size={14} /> Export Corrections
                </button>
                <button
                  onClick={exportTrainingPairs}
                  disabled={editCount === 0}
                  className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-primary/10 text-primary text-xs font-label hover:bg-primary/20 disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  <Download size={14} /> Export Training Pairs
                </button>
              </div>
            </div>
          </div>

          {/* Right: Transcript with word-level editing */}
          <div className="lg:col-span-2">
            <div className="flux-card rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-label text-xs uppercase tracking-widest text-on-surface-variant flex items-center gap-2">
                  <Volume2 size={14} className="text-primary" />
                  Transcript — click any word to edit
                </h3>
                {editCount > 0 && (
                  <span className="text-xs font-label text-secondary">
                    {editCount} edit{editCount !== 1 ? 's' : ''}
                  </span>
                )}
              </div>

              <div className="space-y-4 max-h-[70vh] overflow-y-auto pr-2">
                {transcript.segments.map((seg, si) => (
                  <div key={seg.id ?? si} className="group">
                    {/* Speaker + timestamp */}
                    <div className="flex items-center gap-2 mb-1">
                      <span
                        className="text-[10px] font-label text-primary/70 cursor-pointer hover:text-primary"
                        onClick={() => seekTo(seg.start)}
                      >
                        {fmtTime(seg.start)}
                      </span>
                      {seg.speaker && (
                        <span className="text-[10px] font-label text-on-surface-variant">
                          {seg.speaker}
                        </span>
                      )}
                    </div>

                    {/* Words */}
                    <p className="text-sm leading-relaxed font-body">
                      {(seg.words || []).map((word, wi) => {
                        const globalIdx = allWords.findIndex(
                          w => w.segIdx === si && w.wordIdx === wi
                        )
                        const isActive = globalIdx === activeWordKey
                        const isEdited = !!word.corrected

                        // Inline editor
                        if (editingWord?.segIdx === si && editingWord?.wordIdx === wi) {
                          return (
                            <span key={wi} className="inline-flex items-center gap-1 mx-1">
                              <input
                                autoFocus
                                value={editValue}
                                onChange={(e) => setEditValue(e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter') saveEdit()
                                  if (e.key === 'Escape') cancelEdit()
                                }}
                                className="bg-primary/20 text-primary px-2 py-0.5 rounded text-sm font-mono w-auto min-w-[60px] outline-none border border-primary/40"
                                style={{ width: `${Math.max(60, editValue.length * 9)}px` }}
                              />
                              <button onClick={saveEdit} className="text-secondary hover:text-secondary/80">
                                <Check size={14} />
                              </button>
                              <button onClick={cancelEdit} className="text-error hover:text-error/80">
                                <X size={14} />
                              </button>
                            </span>
                          )
                        }

                        return (
                          <span key={wi} ref={isActive ? activeRef : null}>
                            <Word
                              word={word}
                              isActive={isActive}
                              isEdited={isEdited}
                              onClick={() => handleWordClick(si, wi, word)}
                            />
                          </span>
                        )
                      })}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
