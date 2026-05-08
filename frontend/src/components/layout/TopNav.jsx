import { Search, Bell, Activity } from 'lucide-react'
import { usePolling } from '../../hooks/usePolling'
import { useState } from 'react'

function StatusPill({ label, status, color }) {
  const colorMap = {
    green:  'bg-secondary/15 text-secondary border-secondary/20',
    cyan:   'bg-primary/15 text-primary border-primary/20',
    yellow: 'bg-tertiary/15 text-tertiary border-tertiary/20',
    red:    'bg-error/15 text-error border-error/20',
    gray:   'bg-surface-container-high text-on-surface-variant border-outline-variant/20',
  }
  const dotMap = {
    green:  'bg-secondary',
    cyan:   'bg-primary',
    yellow: 'bg-tertiary',
    red:    'bg-error',
    gray:   'bg-on-surface-variant',
  }
  return (
    <div
      className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-label font-semibold border tracking-wide uppercase ${colorMap[color] || colorMap.gray}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${status === 'active' ? 'live-dot' : ''} ${dotMap[color] || dotMap.gray}`} />
      {label}
    </div>
  )
}

export default function TopNav() {
  const [health, setHealth] = useState(null)

  usePolling(async () => {
    try {
      const res = await fetch('/api/health')
      if (res.ok) setHealth(await res.json())
    } catch {}
  }, 30000)

  return (
    <header className="h-14 flex items-center justify-between px-6 border-b border-outline-variant/15 bg-surface-container-low/60 backdrop-blur-md relative z-10">
      {/* Search */}
      <div className="flex items-center gap-2 bg-surface-container/60 rounded-xl px-3 py-1.5 w-72 border border-outline-variant/15 focus-within:border-primary/40 focus-within:bg-surface-container transition-colors">
        <Search size={14} className="text-on-surface-variant/70" />
        <input
          type="text"
          placeholder="Search corrections, jobs, terms…"
          className="bg-transparent text-[13px] text-on-surface placeholder:text-on-surface-variant/50 outline-none flex-1"
        />
        <kbd className="hidden md:inline text-[9px] font-mono text-on-surface-variant/50 bg-surface-container-high px-1.5 py-0.5 rounded">⌘K</kbd>
      </div>

      {/* Status pills */}
      <div className="flex items-center gap-2">
        <Activity size={13} className="text-on-surface-variant/60" />
        <StatusPill
          label={health?.model_loaded ? 'Model Active' : 'Model Loading'}
          status={health?.model_loaded ? 'active' : 'idle'}
          color={health?.model_loaded ? 'green' : 'yellow'}
        />
        <StatusPill
          label={`OCR · ${health?.ocr_engine || 'PaddleOCR'}`}
          status="active"
          color="cyan"
        />
        <StatusPill
          label={`AVSR · ${health?.avsr_mode || 'MediaPipe'}`}
          status="active"
          color="yellow"
        />
      </div>

      {/* Right actions */}
      <div className="flex items-center gap-1">
        <button
          className="w-9 h-9 rounded-xl text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high/60 transition-colors flex items-center justify-center relative"
          aria-label="Notifications"
        >
          <Bell size={16} />
          <span className="absolute top-2 right-2.5 w-1.5 h-1.5 rounded-full bg-tertiary" />
        </button>
        <div className="w-px h-5 bg-outline-variant/20 mx-1" />
        <div className="flex items-center gap-2 pr-1">
          <span className="w-7 h-7 rounded-full bg-gradient-to-br from-primary to-secondary text-on-primary text-[11px] font-bold flex items-center justify-center">
            SA
          </span>
        </div>
      </div>
    </header>
  )
}
