import Card from './ui/card'
import Button from './ui/button'
import Input from './ui/input'

const AuthScreen = ({ 
  mode, 
  setMode, 
  username, 
  setUsername, 
  password, 
  setPassword, 
  handleAuth, 
  authLoading, 
  authMessage, 
  authError 
}) => {
  return (
    <div className="login-container">
      <Card className="login-card">
        <div className="card-head column center-text">
          <div className="brand-icon-large"></div>
          <h2>Welcome to TruthLens</h2>
          <p className="muted">
            Securely sign in to verify claims with AI precision
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
  )
}

export default AuthScreen
