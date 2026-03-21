const Badge = ({ user }) => {
  return (
    <div className={`badge ${user ? 'badge--user' : 'badge--anon'}`}>
      {user ? user.username : 'Guest'}
    </div>
  )
}

export default Badge
