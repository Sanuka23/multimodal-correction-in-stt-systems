import { useState, useEffect, useCallback } from 'react'
import api from '../api/client'

export function useApi(url, defaultValue = null) {
  const [data, setData] = useState(defaultValue)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetch = useCallback(async () => {
    try {
      setLoading(true)
      const res = await api.get(url)
      setData(res.data)
      setError(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [url])

  useEffect(() => { fetch() }, [fetch])
  return { data, loading, error, refetch: fetch }
}
