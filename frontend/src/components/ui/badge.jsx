const Badge = ({ user, className, ...props }) => {
  const classes = [
    'badge',
    user ? 'badge--user' : 'badge--anon',
    className
  ].filter(Boolean).join(' ')

  return (
    <div className={classes} {...props}>
      {user ? user.username : 'Guest'}
    </div>
  )
}

export default Badge
