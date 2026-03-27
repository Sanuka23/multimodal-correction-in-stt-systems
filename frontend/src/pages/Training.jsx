import { useState, useCallback } from 'react'
import { SlidersHorizontal, Database, CheckCircle } from 'lucide-react'
import { usePolling } from '../hooks/usePolling'
import api from '../api/client'

export default function Training() {
  const [iterations, setIterations] = useState('50,000')
  const [batchSize, setBatchSize] = useState('128')
  const [lr, setLr] = useState('1e-4')
  const [isTraining, setIsTraining] = useState(false)
  const [trainingStatus, setTrainingStatus] = useState(null)
  const [error, setError] = useState(null)

  /* ── Poll training status when active ── */
  const fetchStatus = useCallback(async () => {
    if (!isTraining) return
    try {
      const res = await api.get('/api/training/status')
      setTrainingStatus(res.data)
      if (res.data?.status === 'completed' || res.data?.status === 'failed') {
        setIsTraining(false)
      }
    } catch {
      /* ignore polling errors */
    }
  }, [isTraining])

  usePolling(fetchStatus, 3000, [isTraining])

  /* ── Start training ── */
  const handleStartTraining = async (e) => {
    e.preventDefault()
    setError(null)
    try {
      const payload = {
        iterations: parseInt(iterations.replace(/,/g, ''), 10),
        batch_size: parseInt(batchSize, 10),
        learning_rate: parseFloat(lr),
      }
      await api.post('/api/train', payload)
      setIsTraining(true)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to start training')
    }
  }

  /* ── Derived values ── */
  const currentLoss = trainingStatus?.loss ?? 0.0421
  const accuracy = trainingStatus?.accuracy ?? 98.4
  const step = trainingStatus?.step ?? 32540
  const totalSteps = trainingStatus?.total_steps ?? 50000
  const progressPct = trainingStatus ? Math.round((step / totalSteps) * 100) : 65

  return (
    <div>
      {/* ── Header Editorial ── */}
      <header className="mb-10 flex justify-between items-end">
        <div>
          <nav className="flex gap-2 mb-2">
            <span className="bg-surface-container-high text-on-surface-variant px-2 py-0.5 rounded-md font-label text-[10px]">
              /ENGINE/TRAINING/KINETIC_V3
            </span>
          </nav>
          <h2 className="text-4xl font-headline font-extrabold tracking-tight text-on-background">
            Model Training <span className="text-primary">Orchestration</span>
          </h2>
          <p className="text-on-surface-variant mt-2 max-w-xl font-body">
            Manage hyper-parameters, monitor convergence in real-time, and version adaptive layers for the Kinetic inference engine.
          </p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <span className="font-label text-sm text-secondary uppercase tracking-tighter">System Status: Nominal</span>
          <span className="font-label text-2xl font-bold">Apple M3 Max / MLX backend</span>
        </div>
      </header>

      {/* ── Main Bento Grid ── */}
      <div className="grid grid-cols-12 gap-6">

        {/* ── Left Column: Parameters & Dataset ── */}
        <div className="col-span-12 lg:col-span-4 space-y-6">

          {/* Section 1: Hyperparameters */}
          <section className="bg-surface-container-low p-6 rounded-xl border border-outline-variant/10 relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
              <SlidersHorizontal size={64} />
            </div>
            <h3 className="font-headline font-bold text-lg mb-6 flex items-center gap-2">
              <span className="w-1.5 h-4 bg-primary rounded-full" /> Hyperparameters
            </h3>
            <form className="space-y-6" onSubmit={handleStartTraining}>
              <div>
                <label className="block font-label text-xs text-on-surface-variant uppercase tracking-widest mb-2">
                  Iterations
                </label>
                <input
                  className="w-full bg-surface-container-highest border-none focus:ring-1 focus:ring-primary rounded-lg font-label text-lg p-3 text-primary"
                  type="text"
                  value={iterations}
                  onChange={(e) => setIterations(e.target.value)}
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block font-label text-xs text-on-surface-variant uppercase tracking-widest mb-2">
                    Batch Size
                  </label>
                  <input
                    className="w-full bg-surface-container-highest border-none focus:ring-1 focus:ring-primary rounded-lg font-label text-lg p-3"
                    type="text"
                    value={batchSize}
                    onChange={(e) => setBatchSize(e.target.value)}
                  />
                </div>
                <div>
                  <label className="block font-label text-xs text-on-surface-variant uppercase tracking-widest mb-2">
                    LR (&alpha;)
                  </label>
                  <input
                    className="w-full bg-surface-container-highest border-none focus:ring-1 focus:ring-primary rounded-lg font-label text-lg p-3"
                    type="text"
                    value={lr}
                    onChange={(e) => setLr(e.target.value)}
                  />
                </div>
              </div>
              {error && (
                <p className="text-error text-xs font-label">{error}</p>
              )}
              <button
                type="submit"
                disabled={isTraining}
                className="w-full py-4 bg-primary text-on-primary font-bold rounded-xl font-headline uppercase tracking-widest hover:brightness-110 active:scale-[0.98] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isTraining ? 'Training in Progress...' : 'Initialize Training Run'}
              </button>
            </form>
          </section>

          {/* Section 4: Dataset Synthesis */}
          <section className="bg-surface-container-low p-6 rounded-xl border border-outline-variant/10">
            <h3 className="font-headline font-bold text-lg mb-6 flex items-center gap-2">
              <span className="w-1.5 h-4 bg-tertiary rounded-full" /> Dataset Synthesis
            </h3>
            <div className="space-y-6">
              <div className="flex justify-between items-center bg-surface-container-high p-4 rounded-xl">
                <div>
                  <p className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest">Total Examples</p>
                  <p className="font-label text-2xl font-bold">18,920</p>
                </div>
                <Database className="text-tertiary" size={24} />
              </div>
              <div className="pt-4">
                <p className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest mb-4">ScreenApp Source Mix</p>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-[10px] font-label uppercase mb-1">
                      <span>aws_migration</span>
                      <span>42%</span>
                    </div>
                    <div className="h-1 bg-surface-container-highest rounded-full overflow-hidden">
                      <div className="h-full bg-primary" style={{ width: '42%' }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-[10px] font-label uppercase mb-1">
                      <span>onboarding_andre</span>
                      <span>28%</span>
                    </div>
                    <div className="h-1 bg-surface-container-highest rounded-full overflow-hidden">
                      <div className="h-full bg-secondary" style={{ width: '28%' }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-[10px] font-label uppercase mb-1">
                      <span>business_discussion</span>
                      <span>30%</span>
                    </div>
                    <div className="h-1 bg-surface-container-highest rounded-full overflow-hidden">
                      <div className="h-full bg-tertiary" style={{ width: '30%' }} />
                    </div>
                  </div>
                </div>
              </div>
              <div className="pt-6 border-t border-outline-variant/10">
                <p className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest mb-4">Accent Distribution</p>
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-primary" />
                    <div className="text-[10px] font-label">
                      <p className="text-on-surface font-bold">South Asian</p>
                      <p className="text-on-surface-variant">34.2%</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-secondary" />
                    <div className="text-[10px] font-label">
                      <p className="text-on-surface font-bold">American</p>
                      <p className="text-on-surface-variant">29.8%</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-tertiary" />
                    <div className="text-[10px] font-label">
                      <p className="text-on-surface font-bold">European</p>
                      <p className="text-on-surface-variant">21.5%</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-outline" />
                    <div className="text-[10px] font-label">
                      <p className="text-on-surface font-bold">Unknown</p>
                      <p className="text-on-surface-variant">14.5%</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* ── Right Column: Progress & Versioning ── */}
        <div className="col-span-12 lg:col-span-8 space-y-6">

          {/* Section 2: Convergence Metrics */}
          <section className="bg-surface-container-low p-6 rounded-xl border border-outline-variant/10">
            <div className="flex justify-between items-center mb-8">
              <div>
                <h3 className="font-headline font-bold text-lg flex items-center gap-2">
                  <span className="w-1.5 h-4 bg-emerald-400 rounded-full" /> Convergence Metrics
                </h3>
                <p className="text-[10px] font-label text-on-surface-variant uppercase tracking-widest mt-1">
                  Live Telemetry &bull; ID: RUN_772_DELTA
                </p>
              </div>
              <div className="flex gap-4">
                <div className="text-right">
                  <p className="text-[10px] font-label text-on-surface-variant uppercase tracking-widest">Current Loss</p>
                  <p className="font-label text-xl font-bold text-primary">{currentLoss.toFixed(4)}</p>
                </div>
                <div className="text-right">
                  <p className="text-[10px] font-label text-on-surface-variant uppercase tracking-widest">Accuracy</p>
                  <p className="font-label text-xl font-bold text-secondary">{accuracy.toFixed(1)}%</p>
                </div>
              </div>
            </div>

            {/* Loss Curve SVG — going DOWN (converging) */}
            <div className="h-64 bg-surface-container-high rounded-xl relative overflow-hidden mb-6 p-4 border border-outline-variant/10">
              {/* Grid background */}
              <div className="absolute inset-0 grid grid-cols-12 grid-rows-6 pointer-events-none opacity-5">
                {Array.from({ length: 36 }).map((_, i) => (
                  <div key={i} className="border-r border-b border-outline" />
                ))}
              </div>
              {/* SVG loss curve */}
              <svg className="absolute bottom-0 left-0 w-full h-full" viewBox="0 0 1000 200">
                <defs>
                  <linearGradient id="curve-fill" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="#7bd0ff" stopOpacity="0.1" />
                    <stop offset="100%" stopColor="#7bd0ff" stopOpacity="0" />
                  </linearGradient>
                </defs>
                {/* Converging line (Going down) */}
                <path
                  d="M0,40 Q150,50 300,100 T600,150 T850,165 T1000,170"
                  fill="none"
                  stroke="#7bd0ff"
                  strokeLinecap="round"
                  strokeWidth="3"
                />
                <path
                  d="M0,40 Q150,50 300,100 T600,150 T850,165 T1000,170 V200 H0 Z"
                  fill="url(#curve-fill)"
                />
              </svg>
              {/* Pulsing indicator dot */}
              <div className="absolute bottom-[13%] right-[5%] w-3 h-3 bg-primary rounded-full shadow-[0_0_15px_#7bd0ff] animate-pulse" />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Training Progress */}
              <div className="bg-surface-container-high p-4 rounded-xl border border-outline-variant/10">
                <div className="flex justify-between items-end mb-3">
                  <div>
                    <p className="text-[10px] font-label text-on-surface-variant uppercase tracking-widest">Training Progress</p>
                    <p className="font-label text-lg font-bold">
                      Step {step.toLocaleString()} <span className="text-on-surface-variant font-normal">/ 50k</span>
                    </p>
                  </div>
                  <p className="font-label text-sm text-secondary">{progressPct}.0%</p>
                </div>
                <div className="h-2 bg-surface-container-highest rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-primary to-secondary transition-all duration-500"
                    style={{ width: `${progressPct}%` }}
                  />
                </div>
              </div>

              {/* Quality Filter Log */}
              <div className="bg-surface-container-high p-4 rounded-xl border border-outline-variant/10 font-label">
                <p className="text-[10px] text-on-surface-variant uppercase tracking-widest mb-3">Quality Filter Log</p>
                <div className="space-y-2 text-[10px]">
                  <div className="flex justify-between items-center text-error">
                    <span>REMOVED: Stop Words Pattern</span>
                    <span className="font-bold">-12,402</span>
                  </div>
                  <div className="flex justify-between items-center text-error">
                    <span>REMOVED: Short Context (&lt;4s)</span>
                    <span className="font-bold">-48,912</span>
                  </div>
                  <div className="flex justify-between items-center text-secondary">
                    <span>PASSED: High Confidence (OCR)</span>
                    <span className="font-bold">+1,179,178</span>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Section 3: Adapter Management */}
          <section className="bg-surface-container-low p-6 rounded-xl border border-outline-variant/10">
            <div className="flex justify-between items-center mb-6">
              <h3 className="font-headline font-bold text-lg flex items-center gap-2">
                <span className="w-1.5 h-4 bg-primary rounded-full" /> Adapter Management
              </h3>
              <div className="flex gap-2">
                <button className="bg-surface-container-high hover:bg-surface-bright text-xs px-3 py-1.5 rounded-lg font-label uppercase tracking-widest transition-colors border border-outline-variant/20">
                  Compare All
                </button>
              </div>
            </div>
            <div className="overflow-hidden">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest border-b border-outline-variant/10">
                    <th className="pb-4 px-2">Identifier</th>
                    <th className="pb-4 px-2">Val Loss</th>
                    <th className="pb-4 px-2">Timestamp</th>
                    <th className="pb-4 px-2 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="font-body text-sm">
                  <tr className="group hover:bg-surface-bright/20 transition-colors border-b border-outline-variant/5">
                    <td className="py-4 px-2">
                      <div className="flex items-center gap-3">
                        <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                        <div>
                          <p className="font-label font-bold text-on-surface uppercase">adapters_v2/</p>
                          <p className="text-[10px] text-primary uppercase">v2, training</p>
                        </div>
                      </div>
                    </td>
                    <td className="py-4 px-2 font-label text-primary">{currentLoss.toFixed(4)}</td>
                    <td className="py-4 px-2 text-on-surface-variant">Active Run</td>
                    <td className="py-4 px-2 text-right">
                      <span className="text-[10px] font-label uppercase text-primary px-3 py-1 bg-primary/10 rounded-full">
                        Current
                      </span>
                    </td>
                  </tr>
                  <tr className="group hover:bg-surface-bright/20 transition-colors border-b border-outline-variant/5">
                    <td className="py-4 px-2">
                      <div className="flex items-center gap-3">
                        <span className="w-2 h-2 rounded-full bg-on-surface-variant" />
                        <div>
                          <p className="font-label font-bold text-on-surface uppercase">adapters/</p>
                          <p className="text-[10px] text-on-surface-variant uppercase">v1, val loss 0.773</p>
                        </div>
                      </div>
                    </td>
                    <td className="py-4 px-2 font-label text-on-surface-variant">0.7730</td>
                    <td className="py-4 px-2 text-on-surface-variant">2023.10.20 09:12</td>
                    <td className="py-4 px-2 text-right">
                      <button className="text-xs font-label uppercase tracking-widest text-on-surface-variant hover:text-on-surface px-4 py-2 transition-colors">
                        Rollback
                      </button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </div>

      {/* ── Footer Metric Bar ── */}
      <footer className="mt-12 pt-8 border-t border-outline-variant/10 grid grid-cols-2 md:grid-cols-4 gap-8">
        <div>
          <p className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest mb-1">Compute Tier</p>
          <p className="font-headline font-extrabold text-xl text-on-surface">Apple M3 Max</p>
        </div>
        <div>
          <p className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest mb-1">ML Backend</p>
          <p className="font-headline font-extrabold text-xl text-on-surface">Apple MLX</p>
        </div>
        <div>
          <p className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest mb-1">Queue Priority</p>
          <p className="font-headline font-extrabold text-xl text-on-surface">High-Impact</p>
        </div>
        <div>
          <p className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest mb-1">Session ID</p>
          <p className="font-headline font-extrabold text-xl text-on-surface">KC-992-TX</p>
        </div>
      </footer>
    </div>
  )
}
