const colorMap = {
  primary: 'bg-primary/15 text-primary',
  secondary: 'bg-secondary/15 text-secondary',
  tertiary: 'bg-tertiary/15 text-tertiary',
  error: 'bg-error/15 text-error',
  cyan: 'bg-primary/15 text-primary',
  green: 'bg-secondary/15 text-secondary',
  amber: 'bg-tertiary/15 text-tertiary',
  red: 'bg-error/15 text-error',
  gray: 'bg-surface-container-high text-on-surface-variant',
}

export default function Badge({ children, color = 'gray' }) {
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-label font-medium uppercase tracking-wider ${colorMap[color] || colorMap.gray}`}>
      {children}
    </span>
  )
}
