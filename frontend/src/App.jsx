import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/layout/Layout'
import Dashboard from './pages/Dashboard'
import Compare from './pages/Compare'
import Jobs from './pages/Jobs'
import Training from './pages/Training'
import PipelineControl from './pages/PipelineControl'
import Pipeline from './pages/Pipeline'
import Eval from './pages/Eval'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="compare" element={<Compare />} />
          <Route path="jobs" element={<Jobs />} />
          <Route path="training" element={<Training />} />
          <Route path="pipeline-control" element={<PipelineControl />} />
          <Route path="pipeline/:jobId" element={<Pipeline />} />
          <Route path="eval" element={<Eval />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
