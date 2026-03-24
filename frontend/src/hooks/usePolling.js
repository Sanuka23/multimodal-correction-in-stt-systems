import { useEffect, useRef } from 'react'

export function usePolling(fn, intervalMs, deps = []) {
  const savedFn = useRef(fn)
  useEffect(() => { savedFn.current = fn }, [fn])

  useEffect(() => {
    savedFn.current()
    const id = setInterval(() => savedFn.current(), intervalMs)
    return () => clearInterval(id)
  }, [intervalMs, ...deps])
}
