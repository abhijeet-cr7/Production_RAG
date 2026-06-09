import { useState, useRef } from 'react'

const ALLOWED = ['.pdf', '.docx', '.txt']

export default function FileUpload() {
  const [dragging, setDragging] = useState(false)
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState(null)   // 'uploading' | 'success' | 'error'
  const [result, setResult] = useState(null)
  const [errorMsg, setErrorMsg] = useState('')
  const inputRef = useRef()

  function validateFile(f) {
    const ext = '.' + f.name.split('.').pop().toLowerCase()
    return ALLOWED.includes(ext)
  }

  function pickFile(f) {
    if (!validateFile(f)) {
      setErrorMsg(`Unsupported file type. Allowed: ${ALLOWED.join(', ')}`)
      setFile(null)
      return
    }
    setErrorMsg('')
    setFile(f)
    setResult(null)
    setStatus(null)
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) pickFile(f)
  }

  async function upload() {
    if (!file) return
    setStatus('uploading')
    setResult(null)
    setErrorMsg('')

    const form = new FormData()
    form.append('file', file)

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
    } catch (err) {
      setStatus('error')
      setErrorMsg('Could not reach the API. Is the server running?')
    }
  }

  return (
    <section className="panel">
      <h2 className="panel-title">
        <span className="icon">📄</span> Ingest Document
      </h2>

      <div
        className={`drop-zone ${dragging ? 'drag-over' : ''} ${file ? 'has-file' : ''}`}
        onClick={() => inputRef.current.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt"
          style={{ display: 'none' }}
          onChange={e => e.target.files[0] && pickFile(e.target.files[0])}
        />
        {file ? (
          <div className="file-info">
            <span className="file-icon">{fileIcon(file.name)}</span>
            <span className="file-name">{file.name}</span>
            <span className="file-size">{formatSize(file.size)}</span>
          </div>
        ) : (
          <div className="drop-hint">
            <span className="drop-icon">⬆️</span>
            <p>Drag & drop a file here or <strong>click to browse</strong></p>
            <p className="drop-sub">Supports PDF, DOCX, TXT</p>
          </div>
        )}
      </div>

      {errorMsg && <p className="msg error">{errorMsg}</p>}

      <button
        className="btn btn-primary"
        onClick={upload}
        disabled={!file || status === 'uploading'}
      >
        {status === 'uploading' ? <><span className="spinner" /> Indexing…</> : 'Upload & Index'}
      </button>

      {status === 'success' && result && (
        <div className="result-card success">
          <p className="result-title">✅ Indexed successfully</p>
          <table className="meta-table">
            <tbody>
              <tr><td>File</td><td><code>{result.filename}</code></td></tr>
              <tr><td>Chunks</td><td><code>{result.chunks_indexed}</code></td></tr>
              <tr><td>Doc ID</td><td><code className="truncate">{result.doc_id}</code></td></tr>
            </tbody>
          </table>
        </div>
      )}
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
