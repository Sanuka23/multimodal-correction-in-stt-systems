import { NavLink } from 'react-router-dom'
import { LayoutDashboard, Briefcase, GitCompare, Brain, Settings2, BarChart3, Rocket } from 'lucide-react'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/jobs', icon: Briefcase, label: 'Jobs' },
  { to: '/compare', icon: GitCompare, label: 'Compare' },
  { to: '/training', icon: Brain, label: 'Training' },
  { to: '/pipeline-control', icon: Settings2, label: 'Pipeline' },
  { to: '/eval', icon: BarChart3, label: 'Evaluation' },
]

export default function Sidebar() {
  return (
    <aside className="w-[220px] flex-shrink-0 bg-surface-container-low border-r border-outline-variant/20 flex flex-col h-full">
      {/* Brand */}
      <div className="p-5 pb-6">
        <h1 className="font-headline text-lg font-bold text-primary">Kinetic Console</h1>
        <p className="text-[10px] text-on-surface-variant font-label tracking-wider uppercase mt-0.5">AI Correction Engine</p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 space-y-1">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-3xl text-sm font-medium transition-all ${
                isActive
                  ? 'bg-primary/10 text-primary border-l-2 border-primary'
                  : 'text-on-surface-variant hover:bg-surface-container hover:text-on-surface'
              }`
            }
          >
            <Icon size={18} strokeWidth={1.5} />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Bottom */}
      <div className="p-4 border-t border-outline-variant/20">
        <NavLink
          to="/pipeline-control"
          className="flex items-center justify-center gap-2 w-full py-2.5 rounded-3xl bg-primary text-on-primary font-label font-semibold text-sm hover:bg-primary/90 transition-colors"
        >
          <Rocket size={16} />
          Deploy Model
        </NavLink>
      </div>
    </aside>
  )
}
