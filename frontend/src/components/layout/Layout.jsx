import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'
import TopNav from './TopNav'

export default function Layout() {
  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar />
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden relative">
        {/* Subtle radial backdrop — gives every page a sense of depth */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-[0.55]"
          style={{
            background:
              'radial-gradient(80rem 50rem at 100% -10%, rgba(123, 208, 255, 0.06), transparent 60%),' +
              'radial-gradient(60rem 40rem at -10% 110%, rgba(78, 222, 163, 0.05), transparent 60%)',
          }}
        />

        <TopNav />

        <main className="relative flex-1 overflow-y-auto px-6 py-2">
          <div className="max-w-[1480px] mx-auto">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}
