import { useState, useRef } from 'react'

const ALLOWED = ['.pdf', '.docx', '.txt']

export default function QueryPanel() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [attaching, setAttaching] = useState(false)
  const [errorMsg, setErrorMsg] = useState('')
  const [messages, setMessages] = useState([])
  const textareaRef = useRef()
  const fileInputRef = useRef()

  async function attachFiles(fileList) {
    const incoming = Array.from(fileList)
    const accepted = incoming.filter(f => ALLOWED.includes('.' + f.name.split('.').pop().toLowerCase()))
    const rejected = incoming.filter(f => !accepted.includes(f))

    if (rejected.length) {
      setErrorMsg(`Skipped unsupported file(s): ${rejected.map(f => f.name).join(', ')}`)
    } else {
      setErrorMsg('')
    }
    if (!accepted.length) return

    setAttaching(true)
    const form = new FormData()
    accepted.forEach(f => form.append('files', f))

    try {
      const res = await fetch('/ingest', { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) {
        setErrorMsg(data.detail || 'File indexing failed.')
        return
      }
      const summary = data.files
        .map(f => f.status === 'indexed'
          ? `${f.filename} (${f.chunks_indexed} chunks)`
          : `${f.filename} — failed: ${f.detail}`)
        .join(', ')
      setMessages(prev => [
        ...prev,
        { role: 'system', content: `📎 Added to knowledge base: ${summary}` },
      ])
    } catch {
      setErrorMsg('Could not reach the API. Is the server running?')
    } finally {
      setAttaching(false)
    }
  }

  async function submit(e) {
    e.preventDefault()
    if (!query.trim()) return

    const userText = query.trim()
    const nextMessages = [...messages, { role: 'user', content: userText }]

    setLoading(true)
    setErrorMsg('')
    setMessages(nextMessages)
    setQuery('')

    try {
      const res = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userText,
          chat_history: nextMessages
            .filter(({ role }) => role !== 'system')
            .map(({ role, content }) => ({ role, content })),
        }),
      })
      const data = await res.json()
      if (!res.ok) {
        setErrorMsg(data.detail || 'Query failed.')
        setMessages(messages)
        setLoading(false)
        return
      }
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer || '',
          sources: data.sources || [],
        },
      ])
    } catch {
      setMessages(messages)
      setErrorMsg('Could not reach the API. Is the server running?')
    } finally {
      setLoading(false)
    }
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit(e)
    }
  }

  function startNewChat() {
    setMessages([])
    setErrorMsg('')
    setQuery('')
    textareaRef.current?.focus()
  }

  return (
    <section className="panel query-panel">
      <h2 className="panel-title">
        <span className="icon">💬</span> Chat with your documents
      </h2>

      <div className="chat-toolbar">
        <button
          type="button"
          className="btn btn-secondary"
          onClick={startNewChat}
          disabled={loading || messages.length === 0}
        >
          New chat
        </button>
      </div>

      <div className="chat-thread" aria-live="polite">
        {messages.length === 0 && (
          <p className="chat-empty">
            Start a conversation. Follow-up questions will automatically include prior turns.
          </p>
        )}
        {messages.map((message, i) => (
          message.role === 'system' ? (
            <p key={i} className="chat-notice">{message.content}</p>
          ) : (
          <div key={i} className={`chat-bubble ${message.role}`}>
            <p className="chat-role">{message.role === 'user' ? 'You' : 'Assistant'}</p>
            <p className="chat-text">{message.content}</p>

            {message.role === 'assistant' && message.sources?.length > 0 && (
              <details className="sources-details">
                <summary className="sources-summary">Sources ({message.sources.length})</summary>
                <div className="sources-list">
                  {message.sources.map((s, idx) => (
                    <div key={idx} className="source-item">
                      <span className="source-num">[{idx + 1}]</span>
                      <div className="source-meta">
                        {s.source && <span className="source-file">{s.source}</span>}
                        {s.chunk_index !== undefined && (
                          <span className="source-tag">chunk {s.chunk_index}</span>
                        )}
                        {s.ingested_at && (
                          <span className="source-tag">{new Date(s.ingested_at).toLocaleDateString()}</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </details>
            )}
          </div>
          )
        ))}
        {(loading || attaching) && (
          <p className="chat-thinking">{attaching ? 'Indexing attached files…' : 'Assistant is thinking…'}</p>
        )}
      </div>

      <form onSubmit={submit} className="query-form">
        <textarea
          ref={textareaRef}
          className="query-input"
          placeholder="Send a message… (Enter to send, Shift+Enter for newline)"
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={onKeyDown}
          rows={3}
          disabled={loading}
        />
        <div className="query-actions">
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.txt"
            multiple
            style={{ display: 'none' }}
            onChange={e => { if (e.target.files.length) attachFiles(e.target.files); e.target.value = '' }}
          />
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => fileInputRef.current.click()}
            disabled={attaching || loading}
            title="Attach and index documents"
          >
            📎 Add files
          </button>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={!query.trim() || loading}
          >
            {loading ? <><span className="spinner" /> Thinking…</> : 'Ask'}
          </button>
        </div>
      </form>

      {errorMsg && <p className="msg error">{errorMsg}</p>}
    </section>
  )
}
