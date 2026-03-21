const ThemeToggle = ({ theme, onToggle }) => {
  const label = theme === 'dark' ? 'Dark' : 'Light'
  const icon = theme === 'dark' ? '🌙' : '☀️'
  return (
    <button className="theme-toggle" onClick={onToggle} aria-label="Toggle color theme">
      <span style={{ marginRight: '0.35rem' }}>{icon}</span>
      {label} mode
    </button>
  )
}

export default ThemeToggle
