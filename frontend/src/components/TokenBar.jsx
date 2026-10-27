import { useState } from 'react'
import { describeToken, getToken, setToken } from '../auth'

export default function TokenBar() {
  const [token, setLocal] = useState(getToken())
  const [draft, setDraft] = useState('')
  const info = token ? describeToken(token) : null

  function save() {
    setToken(draft)
    setLocal(draft.trim())
    setDraft('')
  }

  function clear() {
    setToken('')
    setLocal('')
  }

  return (
    <div className="token-bar">
      {token ? (
        <>
          <span className={`token-state ${info && !info.expired ? 'ok' : 'bad'}`}>
            {info
              ? info.expired
                ? 'Token expired'
                : `${info.user} · tenant ${info.tenant}`
              : 'Token set (unreadable)'}
          </span>
          {info && !info.expired && info.scopes.length > 0 && (
            <span className="token-scopes">{info.scopes.join(' ')}</span>
          )}
          <button type="button" className="btn btn-secondary btn-small" onClick={clear}>Sign out</button>
        </>
      ) : (
        <>
          <input
            type="password"
            className="url-input token-input"
            placeholder="Paste access token (JWT)"
            value={draft}
            onChange={e => setDraft(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && draft.trim()) save() }}
            aria-label="Access token"
          />
          <button type="button" className="btn btn-secondary btn-small" onClick={save} disabled={!draft.trim()}>
            Use token
          </button>
        </>
      )}
    </div>
  )
}
