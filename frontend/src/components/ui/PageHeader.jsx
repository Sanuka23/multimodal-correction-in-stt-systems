/**
 * Sticky compact page header used by every screen.
 *
 * Props:
 *   eyebrow      — small uppercase label above the title (e.g. "System Intelligence")
 *   title        — main page title (string or ReactNode)
 *   description  — short subtitle paragraph
 *   icon         — optional Lucide icon component
 *   actions      — ReactNode rendered right-aligned (buttons, toggles, etc.)
 *   meta         — optional small node rendered below the title row (e.g. status badges)
 *   sticky       — set false to disable the sticky behaviour (default true)
 */
export default function PageHeader({
  eyebrow,
  title,
  description,
  icon: Icon,
  actions,
  meta,
  sticky = true,
}) {
  return (
    <header
      className={`
        ${sticky ? 'sticky top-0 z-20' : ''}
        -mx-6 px-6 pt-6 pb-5 mb-6
        bg-background/85 backdrop-blur-md
        border-b border-outline-variant/10
      `}
    >
      <div className="flex items-start justify-between gap-6 flex-wrap">
        <div className="min-w-0 flex-1">
          {eyebrow && (
            <p className="font-label text-[10px] text-primary uppercase tracking-[0.32em] mb-1.5">
              {eyebrow}
            </p>
          )}
          <div className="flex items-center gap-3">
            {Icon && (
              <span className="w-9 h-9 rounded-2xl bg-primary/10 text-primary flex items-center justify-center flex-shrink-0">
                <Icon size={18} strokeWidth={2} />
              </span>
            )}
            <h1 className="font-headline text-3xl md:text-4xl font-extrabold text-on-surface tracking-tight">
              {title}
            </h1>
          </div>
          {description && (
            <p className="text-on-surface-variant text-sm mt-2 max-w-2xl leading-relaxed">
              {description}
            </p>
          )}
          {meta && <div className="mt-3">{meta}</div>}
        </div>

        {actions && (
          <div className="flex items-center gap-2 flex-wrap flex-shrink-0">
            {actions}
          </div>
        )}
      </div>
    </header>
  )
}
