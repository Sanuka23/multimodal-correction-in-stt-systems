import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, Briefcase, GitCompare, Brain, Database, BarChart3, Pencil, Rocket, Sparkles,
} from 'lucide-react'

const navItems = [
  { to: '/',              icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/jobs',          icon: Briefcase,       label: 'Jobs' },
  { to: '/compare',       icon: GitCompare,      label: 'Compare' },
  { to: '/annotate',      icon: Pencil,          label: 'Annotate' },
  { to: '/training',      icon: Brain,           label: 'Training' },
  { to: '/training-data', icon: Database,        label: 'Training Data' },
  { to: '/eval',          icon: BarChart3,       label: 'Evaluation' },
]

export default function Sidebar() {
  return (
    <aside className="w-[224px] flex-shrink-0 bg-surface-container-low/90 backdrop-blur-md border-r border-outline-variant/15 flex flex-col h-full relative">
      {/* Vertical accent line on right edge */}
      <div className="absolute right-0 top-12 bottom-12 w-px bg-gradient-to-b from-transparent via-primary/20 to-transparent" />

      {/* Brand */}
      <div className="px-5 pt-6 pb-7">
        <div className="flex items-center gap-2.5">
          <span className="w-8 h-8 rounded-2xl bg-gradient-to-br from-primary to-secondary text-on-primary flex items-center justify-center shadow-lg shadow-primary/20">
            <Sparkles size={16} strokeWidth={2.4} />
          </span>
          <div className="min-w-0">
            <h1 className="font-headline text-[15px] font-bold text-on-surface leading-tight">
              Kinetic Console
            </h1>
            <p className="text-[9px] text-on-surface-variant/70 font-label tracking-[0.18em] uppercase mt-0.5">
              AI Correction Engine
            </p>
          </div>
        </div>
      </div>

      {/* Section heading */}
      <p className="px-5 pb-2 text-[9px] font-label uppercase tracking-[0.28em] text-on-surface-variant/60">
        Workspace
      </p>

      {/* Navigation */}
      <nav className="flex-1 px-3 space-y-0.5 overflow-y-auto">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `relative flex items-center gap-3 px-3 py-2.5 rounded-xl text-[13px] font-medium transition-all group ${
                isActive
                  ? 'bg-primary/10 text-primary'
                  : 'text-on-surface-variant hover:bg-surface-container/60 hover:text-on-surface'
              }`
            }
          >
            {({ isActive }) => (
              <>
                {/* Active indicator bar */}
                <span
                  className={`absolute left-0 top-1/2 -translate-y-1/2 w-[3px] rounded-r-full transition-all ${
                    isActive ? 'h-6 bg-primary shadow-[0_0_10px_rgba(123,208,255,0.6)]' : 'h-0 bg-transparent'
                  }`}
                />
                <Icon
                  size={17}
                  strokeWidth={isActive ? 2.2 : 1.7}
                  className={isActive ? '' : 'group-hover:text-primary/80 transition-colors'}
                />
                <span className="truncate">{label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer — Deploy Model CTA */}
      <div className="p-4 border-t border-outline-variant/15">
        <NavLink
          to="/pipeline-control"
          className={({ isActive }) =>
            `flex items-center justify-center gap-2 w-full py-2.5 rounded-xl font-label font-bold text-[12px] tracking-wide uppercase transition-all
             ${isActive
               ? 'bg-primary text-on-primary shadow-lg shadow-primary/30'
               : 'bg-gradient-to-br from-primary to-[#5cb8e8] text-on-primary hover:brightness-110 active:scale-[0.98]'
             }`
          }
        >
          <Rocket size={14} />
          Pipeline Control
        </NavLink>
        <p className="text-center text-[9px] text-on-surface-variant/50 font-label tracking-widest mt-3 uppercase">
          v4.1 · Whisper Reconciliation
        </p>
      </div>
    </aside>
  )
}
