laplace <- function(n, k , p){
  q <- 1-p
  square <- sqrt(2*pi*n*p*q)
  exponent <- -(k-n*p)^2/(2*n*p*q)
  e <- exp(1)
  res <- (1/square)*e^(exponent)
  return (res)
}
