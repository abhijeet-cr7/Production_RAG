// Access-token handling for the API.
//
// Production: the token comes from Okta (sign-in flow not included here yet: see
// README "Authentication"). Development: paste a token from
// `scripts/make_dev_token.py` into the token bar, or set VITE_DEV_ACCESS_TOKEN.
// The token lives in sessionStorage only (cleared when the tab closes).

const KEY = 'rag_access_token'

export function getToken() {
  try {
    return sessionStorage.getItem(KEY) || import.meta.env.VITE_DEV_ACCESS_TOKEN || ''
  } catch {
    return import.meta.env.VITE_DEV_ACCESS_TOKEN || ''
  }
}

export function setToken(token) {
  try {
    if (token) sessionStorage.setItem(KEY, token.trim())
    else sessionStorage.removeItem(KEY)
  } catch {
    /* storage unavailable: the token simply is not remembered */
  }
}

/** Headers for an API call: `Authorization` is added when a token is set. */
export function authHeaders(extra = {}) {
  const token = getToken()
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra
}

/** Read the claims of a JWT for display only. This does NOT verify anything. */
export function describeToken(token) {
  try {
    const payload = JSON.parse(atob(token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/')))
    const scopes = payload.scp || (payload.scope ? String(payload.scope).split(' ') : [])
    return {
      user: payload.uid || payload.sub || 'unknown',
      tenant: payload.tenant_id || '—',
      scopes,
      expired: payload.exp ? payload.exp * 1000 < Date.now() : false,
    }
  } catch {
    return null
  }
}
