import { useMutation, useQueryClient } from '@tanstack/react-query'
import api from './client'

export function useEvaluate() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (payload) => api.post('/api/evaluate', payload).then((r) => r.data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['evalResults'] })
      qc.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}

export function useTrain() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (payload) => api.post('/api/train', payload).then((r) => r.data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['trainingStatus'] })
      qc.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}
