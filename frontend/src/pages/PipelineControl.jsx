import { useState } from 'react'
import {
  Settings2,
  CheckCircle,
  Zap,
  Eye,
  FileSearch,
  Ear,
  Layers,
  Database,
  Network,
  Save,
  RotateCcw,
  Upload,
  Terminal,
  Cloud,
  Ban,
  Info,
  Gauge,
  Shield,
  ShieldCheck,
  FolderOpen,
  RefreshCw,
} from 'lucide-react'
import api from '../api/client'

const TABS = ['OCR Settings', 'LLM Filter', 'AVSR Config', 'Sync Triggers']

const PIPELINE_NODES = [
  { id: 'llm_detector', label: 'LLM Detector', icon: Eye, color: 'primary' },
  { id: 'ocr_engine', label: 'OCR Engine', icon: FileSearch, color: 'secondary' },
  { id: 'avsr_sync', label: 'AVSR Sync', icon: Ear, color: 'tertiary' },
  { id: 'batch_corrector', label: 'Batch Corrector', icon: Layers, color: 'outline' },
  { id: 'data_collection', label: 'Data Collection', icon: Database, color: 'primary' },
]

const OCR_PROVIDERS = [
  { id: 'paddle', label: 'PaddleOCR (Local)', icon: Terminal },
  { id: 'google', label: 'Google Cloud', icon: Cloud },
  { id: 'disabled', label: 'Disabled', icon: Ban },
]

const AVSR_MODES = [
  { id: 'auto', label: 'Auto-Detect', disabled: true },
  { id: 'sync', label: 'Sync Only', active: true },
  { id: 'continuous', label: 'Continuous' },
]

export default function PipelineControl() {
  // Master switch
  const [masterEnabled, setMasterEnabled] = useState(true)

  // Pipeline node checkboxes
  const [nodeStates, setNodeStates] = useState({
    llm_detector: true,
    ocr_engine: true,
    avsr_sync: false,
    batch_corrector: true,
    data_collection: true,
  })

  // Active tab
  const [activeTab, setActiveTab] = useState(0)

  // Model orchestration
  const [baseArchitecture, setBaseArchitecture] = useState('Llama-3-70B-Instruct')
  const [adapterVersion, setAdapterVersion] = useState('v2.0')
  const [computeBackend, setComputeBackend] = useState('MLX')
  const [adapterPath, setAdapterPath] = useState(
    '/usr/bin/local/adapters/v2/lora-finetune-universal-ocr'
  )

  // OCR settings
  const [ocrProvider, setOcrProvider] = useState('paddle')
  const [confidence, setConfidence] = useState(85)
  const [frameInterval, setFrameInterval] = useState(250)
  const [maxFrames, setMaxFrames] = useState(64)
  const [avsrMode, setAvsrMode] = useState('sync')

  // Preset dropdown
  const [preset, setPreset] = useState('')

  const toggleNode = (id) => {
    setNodeStates((prev) => ({ ...prev, [id]: !prev[id] }))
  }

  const colorMap = {
    primary: {
      iconText: 'text-primary',
      bgIcon: 'bg-primary/10',
      bgIconHover: 'group-hover:bg-primary/20',
      checkbox: 'text-primary',
    },
    secondary: {
      iconText: 'text-secondary',
      bgIcon: 'bg-secondary/10',
      bgIconHover: 'group-hover:bg-secondary/20',
      checkbox: 'text-secondary',
    },
    tertiary: {
      iconText: 'text-tertiary',
      bgIcon: 'bg-tertiary/10',
      bgIconHover: 'group-hover:bg-tertiary/20',
      checkbox: 'text-tertiary',
    },
    outline: {
      iconText: 'text-outline',
      bgIcon: 'bg-outline/10',
      bgIconHover: 'group-hover:bg-outline/20',
      checkbox: 'text-primary',
    },
  }

  return (
    <>
      {/* Header Section */}
      <div className="flex justify-between items-end mb-10">
        <div>
          <h2 className="text-4xl font-extrabold font-headline tracking-tight text-on-background mb-2">
            Pipeline Configuration
          </h2>
          <p className="text-on-surface-variant font-body">
            Orchestrate multi-modal processing nodes and optimization logic.
          </p>
        </div>
        <div className="flex space-x-3">
          <div className="relative group">
            <select
              value={preset}
              onChange={(e) => setPreset(e.target.value)}
              className="pl-4 pr-10 py-2 bg-surface-container-high hover:bg-surface-bright text-on-surface-variant text-sm font-label rounded-lg transition-all appearance-none cursor-pointer border-none focus:ring-2 focus:ring-primary/20"
            >
              <option value="">Load Preset...</option>
              <option value="production-v4">Production-V4</option>
              <option value="fast-response">Fast-Response</option>
              <option value="high-accuracy-medical">High-Accuracy-Medical</option>
            </select>
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-on-surface-variant pointer-events-none">
              <Settings2 size={14} />
            </span>
          </div>
          <button className="px-4 py-2 bg-surface-container-high hover:bg-surface-bright text-on-surface-variant text-sm font-label rounded-lg transition-all flex items-center">
            <Save size={14} className="mr-2" />
            Save Preset
          </button>
          <button className="px-4 py-2 bg-surface-container-high hover:bg-surface-bright text-on-surface-variant text-sm font-label rounded-lg transition-all flex items-center">
            <RotateCcw size={14} className="mr-2" />
            Reset
          </button>
          <button className="px-6 py-2 bg-primary text-on-primary text-sm font-label font-bold rounded-lg shadow-lg shadow-primary/20 hover:scale-[1.02] active:scale-[0.98] transition-all flex items-center ml-2">
            <Upload size={14} className="mr-2" />
            Apply Changes
          </button>
        </div>
      </div>

      {/* Bento Layout */}
      <div className="grid grid-cols-12 gap-6">
        {/* Left Column — Global Status & Master Control */}
        <div className="col-span-12 lg:col-span-4 space-y-6">
          {/* Master Switch Card */}
          <div className="bg-surface-container-low p-6 rounded-xl relative overflow-hidden obsidian-glass">
            <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 blur-3xl -mr-16 -mt-16" />
            <div className="flex justify-between items-start mb-6">
              <div>
                <span className="font-label text-[10px] uppercase tracking-[0.2em] text-primary block mb-1">
                  Global Pipeline State
                </span>
                <h3 className="text-xl font-bold font-headline">Master Switch</h3>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={masterEnabled}
                  onChange={() => setMasterEnabled(!masterEnabled)}
                  className="sr-only peer"
                />
                <div className="w-14 h-7 bg-surface-container-highest rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:start-[4px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-6 after:w-6 after:transition-all peer-checked:bg-secondary" />
              </label>
            </div>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-surface-container rounded-lg">
                <div className="flex items-center">
                  <CheckCircle size={18} className="text-emerald-400 mr-3" />
                  <span className="text-sm font-medium">Pipeline Connectivity</span>
                </div>
                <span className="text-[10px] font-label text-emerald-400 bg-secondary/10 px-2 py-0.5 rounded-full">
                  OPTIMAL
                </span>
              </div>
              <div className="flex items-center justify-between p-3 bg-surface-container rounded-lg">
                <div className="flex items-center">
                  <Zap size={18} className="text-sky-400 mr-3" />
                  <span className="text-sm font-medium">Current Latency</span>
                </div>
                <span className="text-[10px] font-label text-on-surface">p50: 161s</span>
              </div>
            </div>
          </div>

          {/* Active Node Registry */}
          <div className="bg-surface-container-low p-6 rounded-xl obsidian-glass">
            <h3 className="font-label text-xs uppercase tracking-widest text-on-surface-variant mb-6">
              Active Node Registry
            </h3>
            <div className="space-y-3">
              {PIPELINE_NODES.map((node) => {
                const colors = colorMap[node.color]
                const Icon = node.icon
                return (
                  <div
                    key={node.id}
                    onClick={() => toggleNode(node.id)}
                    className="flex items-center justify-between p-4 bg-surface-container-high rounded-lg hover:bg-surface-bright transition-colors cursor-pointer group"
                  >
                    <div className="flex items-center">
                      <div
                        className={`w-8 h-8 rounded-lg ${colors.bgIcon} flex items-center justify-center mr-4 ${colors.bgIconHover}`}
                      >
                        <Icon size={18} className={colors.iconText} />
                      </div>
                      <span className="text-sm font-semibold">{node.label}</span>
                    </div>
                    <input
                      type="checkbox"
                      checked={nodeStates[node.id]}
                      onChange={() => toggleNode(node.id)}
                      onClick={(e) => e.stopPropagation()}
                      className={`rounded-full border-none bg-surface-container-highest ${colors.checkbox} focus:ring-0 w-5 h-5`}
                    />
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        {/* Right Column — Detailed Step Config */}
        <div className="col-span-12 lg:col-span-8 space-y-6">
          {/* Core Model Orchestration */}
          <div className="bg-surface-container-low p-8 rounded-xl border border-outline-variant/10 obsidian-glass">
            <div className="flex items-center mb-8">
              <Network size={20} className="text-primary mr-3" />
              <h3 className="text-xl font-bold font-headline">Core Model Orchestration</h3>
            </div>
            <div className="grid grid-cols-3 gap-6">
              {/* Base Architecture */}
              <div className="col-span-1 space-y-2">
                <label className="font-label text-xs uppercase tracking-widest text-on-surface-variant">
                  Base Architecture
                </label>
                <select
                  value={baseArchitecture}
                  onChange={(e) => setBaseArchitecture(e.target.value)}
                  className="w-full bg-surface-container-highest border-none rounded-lg text-sm p-3 focus:ring-2 focus:ring-primary/40 appearance-none"
                >
                  <option>Llama-3-70B-Instruct</option>
                  <option>Mistral-Nemo-12B</option>
                  <option>Kinetic-Large-Vision</option>
                </select>
              </div>

              {/* Adapter Version */}
              <div className="col-span-1 space-y-2">
                <label className="font-label text-xs uppercase tracking-widest text-on-surface-variant">
                  Adapter Version
                </label>
                <div className="flex p-1 bg-surface-container rounded-lg">
                  <button
                    onClick={() => setAdapterVersion('v1.2')}
                    className={`flex-1 py-2 text-xs font-bold font-label rounded-lg transition-all ${
                      adapterVersion === 'v1.2'
                        ? 'bg-primary text-on-primary shadow-lg shadow-primary/20'
                        : 'text-on-surface-variant hover:text-on-surface'
                    }`}
                  >
                    v1.2
                  </button>
                  <button
                    onClick={() => setAdapterVersion('v2.0')}
                    className={`flex-1 py-2 text-xs font-bold font-label rounded-lg transition-all ${
                      adapterVersion === 'v2.0'
                        ? 'bg-primary text-on-primary shadow-lg shadow-primary/20'
                        : 'text-on-surface-variant hover:text-on-surface'
                    }`}
                  >
                    v2.0 (Stable)
                  </button>
                </div>
              </div>

              {/* Compute Backend */}
              <div className="col-span-1 space-y-2">
                <label className="font-label text-xs uppercase tracking-widest text-on-surface-variant">
                  Compute Backend
                </label>
                <div className="flex p-1 bg-surface-container rounded-lg">
                  <button
                    onClick={() => setComputeBackend('MLX')}
                    className={`flex-1 py-2 text-xs font-bold font-label rounded-lg transition-all ${
                      computeBackend === 'MLX'
                        ? 'bg-primary text-on-primary shadow-lg shadow-primary/20'
                        : 'text-on-surface-variant hover:text-on-surface'
                    }`}
                  >
                    MLX
                  </button>
                  <button
                    onClick={() => setComputeBackend('Torch')}
                    className={`flex-1 py-2 text-xs font-bold font-label rounded-lg transition-all ${
                      computeBackend === 'Torch'
                        ? 'bg-primary text-on-primary shadow-lg shadow-primary/20'
                        : 'text-on-surface-variant hover:text-on-surface'
                    }`}
                  >
                    Torch
                  </button>
                </div>
              </div>

              {/* Active Adapter Path */}
              <div className="col-span-3 space-y-2">
                <label className="font-label text-xs uppercase tracking-widest text-on-surface-variant">
                  Active Adapter Path
                </label>
                <div className="flex space-x-2">
                  <input
                    type="text"
                    value={adapterPath}
                    onChange={(e) => setAdapterPath(e.target.value)}
                    className="flex-1 font-label bg-surface-container-highest border-none rounded-lg text-sm p-3 focus:ring-2 focus:ring-primary/40"
                  />
                  <button className="px-4 bg-surface-container-highest rounded-lg hover:bg-surface-bright transition-all">
                    <FolderOpen size={18} />
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Tabbed Config Panel */}
          <div className="bg-surface-container-low rounded-xl overflow-hidden border border-outline-variant/10 obsidian-glass">
            {/* Tab Header */}
            <div className="flex bg-surface-container px-6 border-b border-outline-variant/5">
              {TABS.map((tab, idx) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(idx)}
                  className={`px-6 py-4 text-sm tracking-tight transition-all ${
                    activeTab === idx
                      ? 'border-b-2 border-primary text-primary font-bold'
                      : 'text-on-surface-variant hover:text-on-surface'
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>

            {/* Config Panel Content */}
            <div className="p-8 space-y-8">
              {activeTab === 0 && (
                <div className="grid grid-cols-2 gap-x-12 gap-y-8">
                  {/* Engine Provider */}
                  <div className="space-y-3">
                    <h4 className="text-sm font-bold flex items-center">
                      <span className="w-1.5 h-1.5 bg-primary rounded-full mr-2" />
                      Engine Provider
                    </h4>
                    <div className="grid grid-cols-3 gap-3">
                      {OCR_PROVIDERS.map((provider) => {
                        const Icon = provider.icon
                        const isActive = ocrProvider === provider.id
                        return (
                          <button
                            key={provider.id}
                            onClick={() => setOcrProvider(provider.id)}
                            className={`p-3 rounded-xl bg-surface-container-highest border flex flex-col items-center justify-center space-y-2 group transition-all ${
                              isActive
                                ? 'border-primary/40'
                                : 'border-transparent hover:border-outline-variant/30'
                            }`}
                          >
                            <Icon
                              size={20}
                              className={isActive ? 'text-primary' : 'text-on-surface-variant'}
                            />
                            <span
                              className={`text-[9px] font-label font-bold uppercase tracking-widest text-center ${
                                isActive ? '' : 'text-on-surface-variant'
                              }`}
                            >
                              {provider.label}
                            </span>
                          </button>
                        )
                      })}
                    </div>
                  </div>

                  {/* Confidence Threshold */}
                  <div className="space-y-3">
                    <h4 className="text-sm font-bold flex items-center">
                      <span className="w-1.5 h-1.5 bg-primary rounded-full mr-2" />
                      Confidence Threshold
                    </h4>
                    <div className="pt-4">
                      <input
                        type="range"
                        min={0}
                        max={100}
                        value={confidence}
                        onChange={(e) => setConfidence(Number(e.target.value))}
                        className="w-full h-1.5 bg-surface-container-highest rounded-full appearance-none cursor-pointer accent-primary"
                      />
                      <div className="flex justify-between mt-2">
                        <span className="text-[10px] font-mono text-on-surface-variant">
                          Aggressive
                        </span>
                        <span className="text-[10px] font-mono text-primary font-bold">
                          {(confidence / 100).toFixed(2)} (Opt)
                        </span>
                        <span className="text-[10px] font-mono text-on-surface-variant">
                          Strict
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Frame Settings */}
                  <div className="col-span-2 grid grid-cols-2 gap-8 pt-4 border-t border-outline-variant/10">
                    {/* Frame Interval */}
                    <div className="space-y-3">
                      <h4 className="text-sm font-bold flex items-center">
                        <span className="w-1.5 h-1.5 bg-primary rounded-full mr-2" />
                        Frame Interval (ms)
                      </h4>
                      <input
                        type="range"
                        min={10}
                        max={1000}
                        value={frameInterval}
                        onChange={(e) => setFrameInterval(Number(e.target.value))}
                        className="w-full h-1.5 bg-surface-container-highest rounded-full appearance-none cursor-pointer accent-primary"
                      />
                      <div className="flex justify-between mt-2">
                        <span className="text-[10px] font-mono text-on-surface-variant">10ms</span>
                        <span className="text-[10px] font-mono text-primary font-bold">
                          {frameInterval}ms
                        </span>
                        <span className="text-[10px] font-mono text-on-surface-variant">1s</span>
                      </div>
                    </div>

                    {/* Max Buffered Frames */}
                    <div className="space-y-3">
                      <h4 className="text-sm font-bold flex items-center">
                        <span className="w-1.5 h-1.5 bg-primary rounded-full mr-2" />
                        Max Buffered Frames
                      </h4>
                      <div className="relative">
                        <input
                          type="number"
                          value={maxFrames}
                          onChange={(e) => setMaxFrames(Number(e.target.value))}
                          className="w-full bg-surface-container-highest border-none rounded-lg text-sm p-3 focus:ring-2 focus:ring-primary/40"
                        />
                        <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] font-label font-bold text-on-surface-variant uppercase">
                          Frames
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* AVSR Integration Mode */}
                  <div className="col-span-2 pt-6 border-t border-outline-variant/10">
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="text-sm font-bold flex items-center">
                        <span className="w-1.5 h-1.5 bg-tertiary rounded-full mr-2" />
                        AVSR Integration Mode
                      </h4>
                      <div className="relative group cursor-help">
                        <Info size={14} className="text-on-surface-variant" />
                        <div className="absolute bottom-full right-0 mb-2 w-48 p-2 bg-surface-bright text-[10px] leading-tight text-on-surface rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50 shadow-xl border border-outline-variant/20">
                          AVSR may be disabled if no audio stream is detected or if GPU memory
                          headroom is below 4GB.
                        </div>
                      </div>
                    </div>
                    <div className="flex p-1 bg-surface-container rounded-lg w-full">
                      <button
                        className="flex-1 py-2 text-xs font-bold font-label rounded-lg bg-surface-container-highest text-on-surface-variant opacity-50 cursor-not-allowed"
                        disabled
                      >
                        Auto-Detect
                      </button>
                      <button
                        onClick={() => setAvsrMode('sync')}
                        className={`flex-1 py-2 text-xs font-bold font-label rounded-lg transition-all ${
                          avsrMode === 'sync'
                            ? 'bg-tertiary text-on-tertiary shadow-lg shadow-tertiary/20'
                            : 'text-on-surface-variant hover:text-on-surface'
                        }`}
                      >
                        Sync Only
                      </button>
                      <button
                        onClick={() => setAvsrMode('continuous')}
                        className={`flex-1 py-2 text-xs font-bold font-label rounded-lg transition-all ${
                          avsrMode === 'continuous'
                            ? 'bg-tertiary text-on-tertiary shadow-lg shadow-tertiary/20'
                            : 'text-on-surface-variant hover:text-on-surface'
                        }`}
                      >
                        Continuous
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 1 && (
                <div className="text-on-surface-variant text-sm">
                  LLM Filter configuration panel. Configure language model filtering parameters.
                </div>
              )}

              {activeTab === 2 && (
                <div className="text-on-surface-variant text-sm">
                  AVSR configuration panel. Configure audio-visual speech recognition settings.
                </div>
              )}

              {activeTab === 3 && (
                <div className="text-on-surface-variant text-sm">
                  Sync Triggers configuration panel. Configure synchronization trigger rules.
                </div>
              )}
            </div>
          </div>

          {/* Presets & Quick Actions */}
          <div className="grid grid-cols-3 gap-6">
            <div className="col-span-1 bg-surface-container-low p-4 rounded-xl flex items-center space-x-4 border border-outline-variant/5 hover:border-primary/20 transition-all cursor-pointer obsidian-glass">
              <div className="w-12 h-12 rounded-lg bg-primary/5 flex items-center justify-center">
                <Gauge size={20} className="text-primary" />
              </div>
              <div>
                <p className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant">
                  Performance
                </p>
                <h4 className="text-sm font-bold">Fast Inference</h4>
              </div>
            </div>
            <div className="col-span-1 bg-surface-container-low p-4 rounded-xl flex items-center space-x-4 border border-outline-variant/5 hover:border-secondary/20 transition-all cursor-pointer obsidian-glass">
              <div className="w-12 h-12 rounded-lg bg-secondary/5 flex items-center justify-center">
                <Shield size={20} className="text-secondary" />
              </div>
              <div>
                <p className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant">
                  Accuracy
                </p>
                <h4 className="text-sm font-bold">Deep Scan Pro</h4>
              </div>
            </div>
            <div className="col-span-1 bg-surface-container-low p-4 rounded-xl flex items-center space-x-4 border border-outline-variant/5 hover:border-tertiary/20 transition-all cursor-pointer obsidian-glass">
              <div className="w-12 h-12 rounded-lg bg-tertiary/5 flex items-center justify-center">
                <ShieldCheck size={20} className="text-tertiary" />
              </div>
              <div>
                <p className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant">
                  Validation
                </p>
                <h4 className="text-sm font-bold">Safe Mode</h4>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* System Log Footer */}
      <div className="mt-8 bg-surface-container-lowest p-4 rounded-lg border border-outline-variant/10 obsidian-glass">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant">
              Live System Output
            </span>
          </div>
          <button className="text-[10px] font-label text-primary hover:underline">
            OPEN CONSOLE
          </button>
        </div>
        <div className="font-label text-xs text-on-surface-variant space-y-1">
          <p>
            <span className="text-primary/60">[14:22:01]</span> NODE_INIT: Starting OCR Provider
            cluster-A1...
          </p>
          <p>
            <span className="text-primary/60">[14:22:03]</span> CONFIG_SYNC: Configuration version
            2.0.0 applied globally.
          </p>
          <p>
            <span className="text-emerald-500/60">[14:22:05]</span> PIPELINE_READY: Awaiting stream
            input on :8080.
          </p>
        </div>
      </div>

      {/* Floating Action */}
      <div className="fixed bottom-6 right-6 z-30">
        <div className="bg-surface-container-highest shadow-2xl p-4 rounded-2xl border border-primary/20 flex items-center space-x-4 obsidian-glass">
          <div className="flex -space-x-2">
            <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-[10px] font-bold text-on-primary">
              ML
            </div>
            <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center text-[10px] font-bold text-on-secondary">
              OC
            </div>
          </div>
          <div className="pr-4 border-r border-outline-variant/20">
            <p className="text-[10px] font-label uppercase tracking-widest text-on-surface-variant">
              Active Threads
            </p>
            <p className="text-xs font-bold font-mono">14 / 20</p>
          </div>
          <button className="text-primary hover:scale-110 transition-transform">
            <RefreshCw size={20} />
          </button>
        </div>
      </div>
    </>
  )
}
