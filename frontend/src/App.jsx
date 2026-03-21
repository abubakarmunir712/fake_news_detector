import { useCallback, useEffect, useState } from 'react'
import './index.css'
import Badge from './components/ui/badge'
import ThemeToggle from './components/ui/theme-toggle'
import { API_URL } from './config'
import AuthScreen from './components/AuthScreen'
import Dashboard from './components/Dashboard'

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
        <AuthScreen
          mode={mode}
          setMode={setMode}
          username={username}
          setUsername={setUsername}
          password={password}
          setPassword={setPassword}
          handleAuth={handleAuth}
          authLoading={authLoading}
          authMessage={authMessage}
          authError={authError}
        />
      ) : (
        <Dashboard
          claim={claim}
          setClaim={setClaim}
          logout={logout}
          clearResult={clearResult}
          runDetect={runDetect}
          detectLoading={detectLoading}
          detectMessage={detectMessage}
          result={result}
          getVerdictKey={getVerdictKey}
        />
      )}
    </div>
  )
}

export default App
