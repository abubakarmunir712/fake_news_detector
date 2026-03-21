const Badge = ({ user, className, ...props }) => {
  const classes = [
    'badge',
    user ? 'badge--user' : 'badge--anon',
    className
  ].filter(Boolean).join(' ')

  return (
    <span className={classes} {...props}>
      {user ? (user.username || user.id) : 'Guest'}
    </span>
  )
}

export default Badge
