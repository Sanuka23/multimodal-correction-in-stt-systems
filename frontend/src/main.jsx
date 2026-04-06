import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import QueryProvider from './contexts/QueryProvider'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <QueryProvider>
      <App />
    </QueryProvider>
  </StrictMode>,
)
