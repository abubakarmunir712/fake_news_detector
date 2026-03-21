const ThemeToggle = ({ theme, onToggle, className, ...props }) => {
  const icon = theme === 'dark' ? '🌙' : '☀️'
  return (
    <button 
      className={['theme-toggle', className].filter(Boolean).join(' ')} 
      onClick={onToggle} 
      title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
      aria-label="Toggle color theme"
      {...props}
    >
      {icon}
    </button>
  )
}

export default ThemeToggle
