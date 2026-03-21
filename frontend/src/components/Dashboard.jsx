import Card from './ui/card'
import Button from './ui/button'
import ResultCard from './ResultCard'

const Dashboard = ({ 
  claim, 
  setClaim, 
  logout, 
  clearResult, 
  runDetect, 
  detectLoading, 
  detectMessage, 
  result, 
  getVerdictKey 
}) => {
  return (
    <div className="grid grid-stack">
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

        <div className="stack sm mt-4 flex-end">
          <Button
            variant="secondary"
            onClick={clearResult}
            disabled={detectLoading && !result}
            className="flex-1"
          >
            Clear
          </Button>
          <Button
            variant="primary"
            onClick={runDetect}
            disabled={detectLoading}
            className="flex-1"
          >
            {detectLoading && <span className="loading-spinner"></span>}
            {detectLoading ? 'Analyzing...' : 'Verify Claim'}
          </Button>
        </div>

        {detectMessage && <div className="status">{detectMessage}</div>}
      </Card>

      <ResultCard result={result} getVerdictKey={getVerdictKey} />
    </div>
  )
}

export default Dashboard
