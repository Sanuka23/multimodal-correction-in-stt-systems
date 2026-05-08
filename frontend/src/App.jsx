import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/layout/Layout'

const Dashboard       = lazy(() => import('./pages/Dashboard'))
const Compare         = lazy(() => import('./pages/Compare'))
const Jobs            = lazy(() => import('./pages/Jobs'))
const Training        = lazy(() => import('./pages/Training'))
const TrainingData    = lazy(() => import('./pages/TrainingData'))
const PipelineControl = lazy(() => import('./pages/PipelineControl'))
const Pipeline        = lazy(() => import('./pages/Pipeline'))
const Eval            = lazy(() => import('./pages/Eval'))
const Annotate        = lazy(() => import('./pages/Annotate'))

function PageFallback() {
  return (
    <div className="flex items-center justify-center py-24">
      <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
    </div>
  )
}

function withSuspense(Component) {
  return (
    <Suspense fallback={<PageFallback />}>
      <Component />
    </Suspense>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index                              element={withSuspense(Dashboard)} />
          <Route path="compare"                     element={withSuspense(Compare)} />
          <Route path="jobs"                        element={withSuspense(Jobs)} />
          <Route path="training"                    element={withSuspense(Training)} />
          <Route path="training-data"               element={withSuspense(TrainingData)} />
          <Route path="pipeline-control"            element={withSuspense(PipelineControl)} />
          <Route path="pipeline/:jobId"             element={withSuspense(Pipeline)} />
          <Route path="eval"                        element={withSuspense(Eval)} />
          <Route path="annotate"                    element={withSuspense(Annotate)} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
