import { useState, useRef } from 'react'

const ALLOWED = ['.pdf', '.docx', '.txt']

export default function FileUpload() {
  const [dragging, setDragging] = useState(false)
  const [files, setFiles] = useState([])
  const [status, setStatus] = useState(null)   // 'uploading' | 'success' | 'error'
  const [result, setResult] = useState(null)
  const [errorMsg, setErrorMsg] = useState('')
  const [url, setUrl] = useState('')
  const [urlStatus, setUrlStatus] = useState(null) // 'loading' | 'success' | 'error'
  const [urlResult, setUrlResult] = useState(null)
  const [urlError, setUrlError] = useState('')
  const inputRef = useRef()

  function validateFile(f) {
    const ext = '.' + f.name.split('.').pop().toLowerCase()
    return ALLOWED.includes(ext)
  }

  function addFiles(list) {
    const incoming = Array.from(list)
    const rejected = incoming.filter(f => !validateFile(f))
    const accepted = incoming.filter(validateFile)

    setErrorMsg(rejected.length
      ? `Skipped unsupported file(s): ${rejected.map(f => f.name).join(', ')}. Allowed: ${ALLOWED.join(', ')}`
      : '')

    if (accepted.length) {
      setFiles(prev => {
        const names = new Set(prev.map(f => f.name))
        return [...prev, ...accepted.filter(f => !names.has(f.name))]
      })
      setResult(null)
      setStatus(null)
    }
  }

  function removeFile(name) {
    setFiles(prev => prev.filter(f => f.name !== name))
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files)
  }

  async function upload() {
    if (!files.length) return
    setStatus('uploading')
    setResult(null)
    setErrorMsg('')

    const form = new FormData()
    files.forEach(f => form.append('files', f))

    try {
      const res = await fetch('/ingest', { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) {
        setStatus('error')
        setErrorMsg(data.detail || 'Upload failed.')
        return
      }
      setStatus('success')
      setResult(data)
      setFiles([])
    } catch {
      setStatus('error')
      setErrorMsg('Could not reach the API. Is the server running?')
    }
  }

  async function ingestUrl() {
    if (!url.trim()) return
    setUrlStatus('loading')
    setUrlResult(null)
    setUrlError('')

    try {
      const res = await fetch('/ingest/url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim() }),
      })
      const data = await res.json()
      if (!res.ok) {
        setUrlStatus('error')
        setUrlError(data.detail || 'URL ingestion failed.')
        return
      }
      setUrlStatus('success')
      setUrlResult(data)
      setUrl('')
    } catch {
      setUrlStatus('error')
      setUrlError('Could not reach the API. Is the server running?')
    }
  }

  return (
    <section className="panel">
      <h2 className="panel-title">
        <span className="icon">📄</span> Ingest Documents
      </h2>

      <div
        className={`drop-zone ${dragging ? 'drag-over' : ''} ${files.length ? 'has-file' : ''}`}
        onClick={() => inputRef.current.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt"
          multiple
          style={{ display: 'none' }}
          onChange={e => { if (e.target.files.length) addFiles(e.target.files); e.target.value = '' }}
        />
        <div className="drop-hint">
          <span className="drop-icon">⬆️</span>
          <p>Drag & drop files here or <strong>click to browse</strong></p>
          <p className="drop-sub">Supports PDF, DOCX, TXT — multiple files allowed</p>
        </div>
      </div>

      {files.length > 0 && (
        <ul className="file-list">
          {files.map(f => (
            <li key={f.name} className="file-list-item">
              <span className="file-icon">{fileIcon(f.name)}</span>
              <span className="file-name">{f.name}</span>
              <span className="file-size">{formatSize(f.size)}</span>
              <button
                type="button"
                className="file-remove"
                aria-label={`Remove ${f.name}`}
                onClick={() => removeFile(f.name)}
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}

      {errorMsg && <p className="msg error">{errorMsg}</p>}

      <button
        className="btn btn-primary"
        onClick={upload}
        disabled={!files.length || status === 'uploading'}
      >
        {status === 'uploading'
          ? <><span className="spinner" /> Indexing…</>
          : `Upload & Index${files.length > 1 ? ` (${files.length} files)` : ''}`}
      </button>

      {status === 'success' && result && (
        <div className="result-card success">
          <p className="result-title">✅ {result.total_chunks_indexed} chunks indexed</p>
          <table className="meta-table">
            <tbody>
              {result.files.map(f => (
                <tr key={f.filename}>
                  <td>{f.status === 'indexed' ? '✅' : '⚠️'}</td>
                  <td>
                    <code>{f.filename}</code>{' '}
                    {f.status === 'indexed'
                      ? <span className="source-tag">{f.chunks_indexed} chunks</span>
                      : <span className="source-tag">{f.detail}</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="url-ingest">
        <p className="section-label">Or index a web page</p>
        <div className="url-row">
          <input
            type="url"
            className="url-input"
            placeholder="https://example.com/article"
            value={url}
            onChange={e => setUrl(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') ingestUrl() }}
            disabled={urlStatus === 'loading'}
          />
          <button
            className="btn btn-secondary"
            onClick={ingestUrl}
            disabled={!url.trim() || urlStatus === 'loading'}
          >
            {urlStatus === 'loading' ? <><span className="spinner" /> Fetching…</> : 'Index URL'}
          </button>
        </div>
        {urlError && <p className="msg error">{urlError}</p>}
        {urlStatus === 'success' && urlResult && (
          <div className="result-card success">
            <p className="result-title">✅ Indexed {urlResult.chunks_indexed} chunks from <code>{urlResult.source}</code></p>
          </div>
        )}
      </div>
    </section>
  )
}

function fileIcon(name) {
  const ext = name.split('.').pop().toLowerCase()
  if (ext === 'pdf') return '📕'
  if (ext === 'docx') return '📘'
  return '📄'
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
