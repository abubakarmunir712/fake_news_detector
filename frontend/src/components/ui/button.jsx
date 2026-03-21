const Button = ({ variant = 'primary', className = '', ...props }) => {
  const classes = ['btn', variant, className].filter(Boolean).join(' ')
  return <button className={classes} {...props} />
}

export default Button
