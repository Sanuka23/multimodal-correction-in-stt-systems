/**
 * Unified content card with optional title row + actions.
 * Drops in over the recurring `bg-surface-container rounded-3xl border` pattern
 * so every screen has the same softness, padding and hover behaviour.
 *
 * Props:
 *   title       — optional section title
 *   eyebrow     — optional uppercase mini-label above title
 *   icon        — optional Lucide icon component for title row
 *   actions     — right-aligned action node
 *   tone        — 'default' | 'glass' | 'elevated' | 'muted'
 *   accent      — optional accent color to tint the top border ('primary', 'secondary', 'tertiary', 'error')
 *   className   — extra classes
 *   children    — body
 */
const TONE = {
  default:  'bg-surface-container border border-outline-variant/10',
  glass:    'obsidian-glass border border-outline-variant/10',
  elevated: 'bg-surface-container-high border border-outline-variant/15 shadow-[0_8px_30px_-12px_rgba(0,0,0,0.5)]',
  muted:    'bg-surface-container-low border border-outline-variant/10',
}

const ACCENT_BORDER = {
  primary:   'before:bg-primary',
  secondary: 'before:bg-secondary',
  tertiary:  'before:bg-tertiary',
  error:     'before:bg-error',
}

export default function SectionCard({
  title,
  eyebrow,
  icon: Icon,
  actions,
  tone = 'default',
  accent,
  className = '',
  bodyClassName = '',
  children,
}) {
  const baseTone = TONE[tone] || TONE.default
  const accentCls = accent
    ? `relative before:content-[''] before:absolute before:left-0 before:top-0 before:bottom-0 before:w-[2px] before:rounded-l-2xl ${ACCENT_BORDER[accent] || ''}`
    : ''

  const showHeader = title || eyebrow || actions
  return (
    <div className={`rounded-2xl ${baseTone} ${accentCls} ${className}`}>
      {showHeader && (
        <div className="flex items-center justify-between gap-3 px-5 pt-4 pb-3 border-b border-outline-variant/10">
          <div className="min-w-0">
            {eyebrow && (
              <p className="font-label text-[10px] uppercase tracking-[0.28em] text-on-surface-variant/70 mb-0.5">
                {eyebrow}
              </p>
            )}
            {title && (
              <h3 className="flex items-center gap-2 font-headline text-base font-bold text-on-surface tracking-tight">
                {Icon && <Icon size={15} className="text-primary flex-shrink-0" strokeWidth={2} />}
                <span className="truncate">{title}</span>
              </h3>
            )}
          </div>
          {actions && (
            <div className="flex items-center gap-2 flex-shrink-0">{actions}</div>
          )}
        </div>
      )}
      <div className={`p-5 ${bodyClassName}`}>{children}</div>
    </div>
  )
}
