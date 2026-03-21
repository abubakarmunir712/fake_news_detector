import { useCallback, useEffect, useState } from 'react'
import './index.css'
import Badge from './components/ui/badge'
import Button from './components/ui/button'
import Card from './components/ui/card'
import Input from './components/ui/input'
import ThemeToggle from './components/ui/theme-toggle'
import { API_URL } from './config'

const api = (path) => `${API_URL}/api${path}`

const useTheme = () => {
  const getInitialTheme = () => {
    const canAccess =
      typeof window !== 'undefined' && typeof localStorage !== 'undefined'
    const stored = canAccess ? localStorage.getItem('theme') : null
    if (stored === 'light' || stored === 'dark') return stored
    const prefersDark =
      canAccess &&
      window.matchMedia &&
      window.matchMedia('(prefers-color-scheme: dark)').matches
    return prefersDark ? 'dark' : 'light'
  }

  const [theme, setTheme] = useState(getInitialTheme)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  return { theme, toggle: () => setTheme((t) => (t === 'dark' ? 'light' : 'dark')) }
}

function App() {
  const { theme, toggle } = useTheme()

  const [mode, setMode] = useState('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [claim, setClaim] = useState('')
  const [user, setUser] = useState(null)
  const [result, setResult] = useState(null)
  const [authMessage, setAuthMessage] = useState('')
  const [authError, setAuthError] = useState(false)
  const [detectMessage, setDetectMessage] = useState('')
  const [authLoading, setAuthLoading] = useState(false)
  const [detectLoading, setDetectLoading] = useState(false)

  const readJson = async (res) => {
    try {
      return await res.json()
    } catch {
      return {}
    }
  }

  const refreshSession = useCallback(async () => {
    try {
      const res = await fetch(api('/me'), { credentials: 'include' })
      if (!res.ok) {
        setUser(null)
        return
      }
      setUser(await res.json())
    } catch {
      setUser(null)
    }
  }, [])

  useEffect(() => {
    refreshSession()
    const intervalId = setInterval(refreshSession, 10000)
    return () => clearInterval(intervalId)
  }, [refreshSession])

  const handleAuth = async () => {
    setAuthLoading(true)
    setAuthMessage('')
    setAuthError(false)
    try {
      const res = await fetch(api(mode === 'login' ? '/login' : '/signup'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ username, password }),
      })
      const data = await readJson(res)
      if (!res.ok) {
        setAuthError(true)
        setAuthMessage(data.error || 'Something went wrong')
        setUser(null)
        return
      }
      setUser(data)
      setAuthMessage(mode === 'login' ? 'Welcome back!' : 'Account created.')
      setUsername('')
      setPassword('')
    } catch (err) {
      setAuthError(true)
      setAuthMessage(err.message)
    } finally {
      setAuthLoading(false)
    }
  }

  const logout = async () => {
    setAuthLoading(true)
    try {
      await fetch(api('/logout'), { method: 'POST', credentials: 'include' })
      setUser(null)
      setResult(null)
      setAuthMessage('Logged out.')
      setAuthError(false)
    } finally {
      setAuthLoading(false)
    }
  }

  const runDetect = async () => {
    if (!claim.trim()) {
      setDetectMessage('Claim is required.')
      return
    }
    if (!user) {
      setDetectMessage('Login required before detecting.')
      return
    }
    setDetectLoading(true)
    setDetectMessage('Running detection...')
    setResult(null)
    try {
      const res = await fetch(api('/detect'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ claim }),
      })
      const data = await readJson(res)
      if (!res.ok) {
        setDetectMessage(data.error || 'Unable to complete detection.')
        return
      }
      setResult(data)
      setDetectMessage('')
    } catch (err) {
      setDetectMessage(err.message)
    } finally {
      setDetectLoading(false)
    }
  }

  const clearResult = () => {
    setResult(null)
    setDetectMessage('')
  }

  return (
    <div className="page">
      <header className="header">
        <div className="header-content">
          <h1 className="title">Fake News Detector</h1>
          <p className="subtitle">
            AI-powered verification using Tavily search & Gemini analysis
          </p>
        </div>
        <div className="stack">
          <Badge user={user} />
          <ThemeToggle theme={theme} onToggle={toggle} />
        </div>
      </header>

      <div className="grid">
        <Card>
          <div className="card-head">
            <div>
              <h2>🔐 Account</h2>
              <p className="muted">
                Sign in or create an account to verify claims
              </p>
            </div>
          </div>
          
          <div className="pill-toggle" style={{ marginBottom: '1.5rem' }}>
            <button
              className={mode === 'login' ? 'active' : ''}
              onClick={() => setMode('login')}
            >
              Login
            </button>
            <button
              className={mode === 'signup' ? 'active' : ''}
              onClick={() => setMode('signup')}
            >
              Sign up
            </button>
          </div>

          <div className="stack column">
            <Input
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
            />
            <Input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
            />
          </div>

          <div className="stack" style={{ marginTop: '1rem' }}>
            <Button
              variant="primary"
              onClick={handleAuth}
              disabled={authLoading}
            >
              {authLoading && <span className="loading-spinner"></span>}
              {authLoading
                ? 'Processing...'
                : mode === 'login'
                  ? 'Login'
                  : 'Create account'}
            </Button>
            {user && (
              <Button
                variant="secondary"
                onClick={logout}
                disabled={authLoading}
              >
                Logout
              </Button>
            )}
            <Button
              variant="ghost"
              onClick={refreshSession}
              disabled={authLoading}
            >
              Refresh
            </Button>
          </div>

          {authMessage && (
            <div className={`status ${authError ? 'error' : ''}`}>
              {authMessage}
            </div>
          )}
          {user && (
            <div className="status">
              ✓ Signed in as <strong>{user.username}</strong>
            </div>
          )}
        </Card>

        <Card>
          <div className="card-head">
            <div>
              <h2>🔍 Detect</h2>
              <p className="muted">
                Enter any claim to check its veracity
              </p>
            </div>
          </div>

          <textarea
            className="input"
            placeholder="Example: NASA has confirmed the existence of alien life..."
            value={claim}
            onChange={(e) => setClaim(e.target.value)}
          ></textarea>

          <div className="stack" style={{ marginTop: '1rem' }}>
            <Button
              variant="primary"
              onClick={runDetect}
              disabled={detectLoading || !user}
            >
              {detectLoading && <span className="loading-spinner"></span>}
              {detectLoading ? 'Analyzing...' : 'Run detection'}
            </Button>
            <Button
              variant="secondary"
              onClick={clearResult}
              disabled={detectLoading && !result}
            >
              Clear
            </Button>
          </div>
          
          {!user && (
            <div className="status error">⚠️ Please login to detect claims</div>
          )}
          {detectMessage && <div className="status">{detectMessage}</div>}

          {result && (
            <div className="result">
              <div className="result-header">
                <div className={`tag ${
                  result.verdict?.toLowerCase().includes('true') ? 'true' :
                  result.verdict?.toLowerCase().includes('fake') ? 'fake' : 
                  'unverifiable'
                }`}>
                  {result.verdict}
                </div>
                {result.search_query && (
                  <div className="fade">🔎 Query: {result.search_query}</div>
                )}
              </div>
              <div className="explanation">
                {result.explanation || result.raw}
              </div>
              {result.sources?.length > 0 && (
                <div>
                  <div className="muted" style={{ marginTop: '1.5rem', fontWeight: 600 }}>
                    📚 Sources ({result.sources.length})
                  </div>
                  <ul className="sources">
                    {result.sources.map((source, idx) => (
                      <li key={idx}>{source}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}

export default App
