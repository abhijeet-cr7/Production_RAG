import { useState, useEffect } from 'react'
import FileUpload from './components/FileUpload'
import QueryPanel from './components/QueryPanel'

export default function App() {
  const [apiStatus, setApiStatus] = useState('checking') // 'checking' | 'ok' | 'down'

  useEffect(() => {
    fetch('/health')
      .then(r => r.ok ? setApiStatus('ok') : setApiStatus('down'))
      .catch(() => setApiStatus('down'))
  }, [])

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">🧠</span>
            <span className="logo-text">Production RAG</span>
          </div>
          <div className={`api-badge ${apiStatus}`}>
            <span className="badge-dot" />
            {apiStatus === 'checking' ? 'Checking API…' : apiStatus === 'ok' ? 'API connected' : 'API offline'}
          </div>
        </div>
      </header>

      <main className="main">
        {apiStatus === 'down' && (
          <div className="banner-warn">
            ⚠️ API is unreachable. Start it with:{' '}
            <code>uvicorn online_pipeline.api.gateway:app --reload</code>
          </div>
        )}
        <div className="grid">
          <FileUpload />
          <QueryPanel />
        </div>
      </main>

      <footer className="footer">
        <span>Production RAG — Qdrant · Redis · Kafka · Groq</span>
      </footer>
    </div>
  )
}
