import { useQuery } from '@tanstack/react-query'
import api from './client'

export function useStats() {
  return useQuery({
    queryKey: ['stats'],
    queryFn: () => api.get('/api/stats').then((r) => r.data),
  })
}

export function useJobs(params = {}) {
  return useQuery({
    queryKey: ['jobs', params],
    queryFn: () => api.get('/api/jobs', { params }).then((r) => r.data),
  })
}

export function useJobSteps(jobId, enabled = true) {
  return useQuery({
    queryKey: ['jobSteps', jobId],
    queryFn: () => api.get(`/api/jobs/${jobId}/steps`).then((r) => r.data),
    enabled: !!jobId && enabled,
    refetchInterval: (query) => {
      const data = query.state.data
      if (!data) return 2000
      const steps = data.pipeline_steps || data.steps || data
      const isRunning = Array.isArray(steps) && steps.some((s) => s.status === 'running')
      return isRunning ? 2000 : false
    },
  })
}

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: () => api.get('/api/health').then((r) => r.data),
  })
}

export function useCorrections(limit = 50) {
  return useQuery({
    queryKey: ['corrections', limit],
    queryFn: () => api.get('/api/corrections', { params: { limit } }).then((r) => r.data),
  })
}

export function useJobsStats() {
  return useQuery({
    queryKey: ['jobsStats'],
    queryFn: () => api.get('/api/jobs/stats').then((r) => r.data),
  })
}

export function useTrainingDatasets() {
  return useQuery({
    queryKey: ['trainingDatasets'],
    queryFn: () => api.get('/api/training/datasets').then((r) => r.data),
  })
}

export function useTrainingData(params = {}) {
  return useQuery({
    queryKey: ['trainingData', params],
    queryFn: () => api.get('/api/training/data', { params }).then((r) => r.data),
    enabled: !!params.dataset && !!params.file,
  })
}

export function useTrainingStatus() {
  return useQuery({
    queryKey: ['trainingStatus'],
    queryFn: () => api.get('/api/training/status').then((r) => r.data),
  })
}

export function useEvalResults(version) {
  return useQuery({
    queryKey: ['evalResults', version],
    queryFn: () => api.get('/api/eval/results', { params: { version } }).then((r) => r.data),
    enabled: !!version,
  })
}
