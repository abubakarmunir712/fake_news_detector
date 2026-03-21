const Card = ({ className, ...props }) => {
  return <div className={['card', className].filter(Boolean).join(' ')} {...props}>{props.children}</div>
}

export default Card
