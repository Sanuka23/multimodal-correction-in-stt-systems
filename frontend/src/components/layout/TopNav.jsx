import { Search, Bell } from 'lucide-react'
import { useApi } from '../../hooks/useApi'
import { usePolling } from '../../hooks/usePolling'
import { useState } from 'react'

function StatusPill({ label, status, color }) {
  const colorMap = {
    green: 'bg-secondary/20 text-secondary',
    cyan: 'bg-primary/20 text-primary',
    yellow: 'bg-tertiary/20 text-tertiary',
    red: 'bg-error/20 text-error',
    gray: 'bg-surface-container-high text-on-surface-variant',
  }
  return (
    <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-label font-medium ${colorMap[color] || colorMap.gray}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${status === 'active' ? 'live-dot' : ''} ${color === 'green' ? 'bg-secondary' : color === 'cyan' ? 'bg-primary' : color === 'yellow' ? 'bg-tertiary' : 'bg-on-surface-variant'}`} />
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
    <header className="h-14 flex items-center justify-between px-6 border-b border-outline-variant/20 bg-surface-container-low/50 backdrop-blur-sm">
      {/* Search */}
      <div className="flex items-center gap-2 bg-surface-container rounded-full px-3 py-1.5 w-64">
        <Search size={14} className="text-on-surface-variant" />
        <input
          type="text"
          placeholder="Search corrections, jobs..."
          className="bg-transparent text-sm text-on-surface placeholder:text-on-surface-variant/50 outline-none flex-1"
        />
      </div>

      {/* Status pills */}
      <div className="flex items-center gap-2">
        <StatusPill
          label={`MODEL ${health?.model_loaded ? 'ACTIVE' : 'LOADING'}`}
          status={health?.model_loaded ? 'active' : 'idle'}
          color={health?.model_loaded ? 'green' : 'yellow'}
        />
        <StatusPill
          label={`OCR ${health?.ocr_engine || 'PaddleOCR'}`}
          status="active"
          color="cyan"
        />
        <StatusPill
          label={`AVSR ${health?.avsr_mode || 'MediaPipe'}`}
          status="active"
          color="yellow"
        />
      </div>

      {/* Right actions */}
      <div className="flex items-center gap-3">
        <button className="text-on-surface-variant hover:text-on-surface transition-colors">
          <Bell size={18} />
        </button>
      </div>
    </header>
  )
}
