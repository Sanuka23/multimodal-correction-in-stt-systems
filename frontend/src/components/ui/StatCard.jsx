export default function StatCard({ icon: Icon, label, value, valueColor = 'text-on-surface', subtitle }) {
  return (
    <div className="bg-surface-container rounded-3xl p-4 flex items-start gap-3">
      {Icon && (
        <div className="w-10 h-10 rounded-2xl bg-surface-container-high flex items-center justify-center">
          <Icon size={18} className="text-primary" />
        </div>
      )}
      <div>
        <p className="text-xs text-on-surface-variant font-label uppercase tracking-wider">{label}</p>
        <p className={`text-2xl font-headline font-bold ${valueColor}`}>{value}</p>
        {subtitle && <p className="text-xs text-on-surface-variant mt-0.5">{subtitle}</p>}
      </div>
    </div>
  )
}
