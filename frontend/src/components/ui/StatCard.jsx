/**
 * Compact metric tile.
 *
 * Props:
 *   label    — uppercase label
 *   value    — main value (string or ReactNode)
 *   icon     — optional Lucide icon component
 *   delta    — optional small delta line (string or ReactNode)
 *   tone     — color of value: 'default' | 'primary' | 'secondary' | 'tertiary' | 'error'
 *   accent   — optional left-border accent ('primary', 'secondary', 'tertiary', 'error')
 *   loading  — show skeleton when true
 *   onClick  — optional click handler (renders as button)
 *
 * Legacy:
 *   valueColor — old prop, mapped onto tone if provided
 *   subtitle   — old prop, mapped onto delta if provided
 */
const VALUE_TONE = {
  default:   'text-on-surface',
  primary:   'text-primary',
  secondary: 'text-secondary',
  tertiary:  'text-tertiary',
  error:     'text-error',
}

const ICON_TONE = {
  default:   'text-on-surface-variant',
  primary:   'text-primary',
  secondary: 'text-secondary',
  tertiary:  'text-tertiary',
  error:     'text-error',
}

const ACCENT_BORDER = {
  primary:   'border-l-primary',
  secondary: 'border-l-secondary',
  tertiary:  'border-l-tertiary',
  error:     'border-l-error',
}

function legacyToneFromValueColor(vc) {
  if (!vc) return null
  if (vc.includes('primary')) return 'primary'
  if (vc.includes('secondary')) return 'secondary'
  if (vc.includes('tertiary')) return 'tertiary'
  if (vc.includes('error')) return 'error'
  return null
}

export default function StatCard({
  label,
  value,
  icon: Icon,
  delta,
  tone,
  accent,
  loading = false,
  onClick,
  // legacy
  valueColor,
  subtitle,
}) {
  const resolvedTone = tone || legacyToneFromValueColor(valueColor) || 'default'
  const resolvedDelta = delta ?? subtitle
  const valueCls = VALUE_TONE[resolvedTone] || VALUE_TONE.default
  const iconCls = ICON_TONE[resolvedTone] || ICON_TONE.default
  const accentCls = accent ? `border-l-[3px] ${ACCENT_BORDER[accent] || ''}` : ''

  const Component = onClick ? 'button' : 'div'

  return (
    <Component
      onClick={onClick}
      className={`
        group bg-surface-container rounded-2xl p-5
        border border-outline-variant/10 ${accentCls}
        ${onClick ? 'text-left cursor-pointer hover:bg-surface-container-high transition-colors w-full' : ''}
        relative overflow-hidden
      `}
    >
      <div className="flex items-start justify-between mb-2">
        <span className="font-label text-[10px] text-on-surface-variant/70 uppercase tracking-[0.2em]">
          {label}
        </span>
        {Icon && <Icon size={14} className={iconCls} strokeWidth={2} />}
      </div>
      {loading ? (
        <div className="h-8 w-24 bg-surface-container-high rounded animate-pulse" />
      ) : (
        <div className={`font-headline text-3xl font-black tracking-tight ${valueCls}`}>
          {value}
        </div>
      )}
      {resolvedDelta && !loading && (
        <div className="text-[10px] mt-1 text-on-surface-variant/80">{resolvedDelta}</div>
      )}
    </Component>
  )
}
