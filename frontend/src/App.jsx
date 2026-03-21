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

  const getVerdictKey = (v) => {
    if (!v) return 'unverifiable'
    const s = v.toLowerCase()
    if (s.includes('true')) return 'likely true'
    if (s.includes('fake')) return 'likely fake'
    if (s.includes('insufficient')) return 'insufficient info'
    if (s.includes('scope')) return 'out of scope'
    return 'unverifiable'
  }

  return (
    <div className="page">
      <header className="header">
        <div className="header-content">
          <h1 className="title">TruthLens</h1>
          <p className="subtitle">
            AI-Powered Fact Verification System
          </p>
        </div>
        <div className="stack">
          {user && <Badge user={user} />}
          <ThemeToggle theme={theme} onToggle={toggle} />
        </div>
      </header>

      {!user ? (
        /* Login/Signup Screen - Full Width/Centered */
        <div className="login-container">
          <Card className="login-card">
            <div className="card-head center-text">
              <h2>Welcome to TruthLens</h2>
              <p className="muted">
                Please sign in to verify claims with AI precision
              </p>
            </div>

            <div className="pill-toggle center-toggle">
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
              <div className="full-width">
                <label className="label">Username</label>
                <Input
                  placeholder="johndoe"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  autoComplete="username"
                  required
                />
              </div>
              <div className="full-width">
                <label className="label">Password</label>
                <Input
                  type="password"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                  required
                />
              </div>
            </div>

            <div className="mt-6">
              <Button
                variant="primary"
                className="full-width"
                onClick={handleAuth}
                disabled={authLoading}
              >
                {authLoading && <span className="loading-spinner"></span>}
                {authLoading
                  ? 'Processing...'
                  : mode === 'login'
                    ? 'Sign In'
                    : 'Create Account'}
              </Button>
            </div>

            {authMessage && (
              <div className={`status ${authError ? 'error' : ''}`}>
                {authMessage}
              </div>
            )}
          </Card>
        </div>
      ) : (
        /* Main Application - Grid Layout */
        <div className="grid grid-2">
          <Card>
            <div className="card-head">
              <div>
                <h2>🔍 Verify Claim</h2>
                <p className="muted">
                  Enter a news headline or claim to analyze
                </p>
              </div>
              <Button
                variant="ghost"
                onClick={logout}
                className="small-btn"
              >
                Logout
              </Button>
            </div>

            <div className="mb-4">
              <label className="label">News Content / Claim</label>
              <textarea
                className="input"
                placeholder="Paste the news headline or article text here to verify..."
                value={claim}
                onChange={(e) => setClaim(e.target.value)}
                rows={5}
              ></textarea>
            </div>

            <div className="stack mt-4 flex-end">
              <Button
                variant="secondary"
                onClick={clearResult}
                disabled={detectLoading && !result}
              >
                Clear
              </Button>
              <Button
                variant="primary"
                onClick={runDetect}
                disabled={detectLoading}
              >
                {detectLoading && <span className="loading-spinner"></span>}
                {detectLoading ? 'Analyzing...' : 'Verify Claim'}
              </Button>
            </div>

            {detectMessage && <div className="status">{detectMessage}</div>}
          </Card>

          {result && (
            <Card
              className="result-card"
              data-verdict={getVerdictKey(result.verdict)}
            >
              <div className="result-header">
                <div className="verdict-container">
                  <div className={`verdict-badge ${result.verdict?.toLowerCase().includes('true') ? 'true' :
                    result.verdict?.toLowerCase().includes('fake') ? 'fake' :
                      result.verdict?.toLowerCase().includes('scope') ? 'scope' :
                        result.verdict?.toLowerCase().includes('insufficient') ? 'insufficient' :
                          'unverifiable'
                    }`}>
                    {result.verdict}
                  </div>
                </div>
                {result.confidence_score !== undefined && result.confidence_score > 0 && (
                  <div className="score-badge">
                    <span className="score-label">Confidence</span>
                    <span className="score">{result.confidence_score}%</span>
                  </div>
                )}
              </div>

              <div className="explanation">
                <h3>Analysis</h3>
                <p>{result.explanation || result.raw}</p>
              </div>

              {result.sources?.length > 0 && (
                <div className="sources-section">
                  <h3>📚 Referenced Sources</h3>
                  <div className="sources-grid">
                    {result.sources.map((source, idx) => (
                      <a
                        key={idx}
                        href={source.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="source-link"
                      >
                        <div className="source-title">{source.title}</div>
                        <div className="source-domain">{new URL(source.url).hostname.replace('www.', '')}</div>
                      </a>
                    ))}
                  </div>
                </div>
              )}
            </Card>
          )}
        </div>
      )}
    </div>
  )
}

export default App
