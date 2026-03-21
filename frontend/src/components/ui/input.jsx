const Input = ({ className = '', ...props }) => {
  return <input className={['input', className].filter(Boolean).join(' ')} {...props} />
}

export default Input
