import Card from './ui/card'

const ResultCard = ({ result, getVerdictKey }) => {
  if (!result) return null

  return (
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
  )
}

export default ResultCard
