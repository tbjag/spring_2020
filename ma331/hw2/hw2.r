n <<- 20
eq1 <- c()
eq2 <- c()
data1 <<- data.frame(eq1,eq2)
for (I in 0:99){
  d <<- rnorm(n, 2, 9)
  equ1 <- (mean(d)-2)/(sqrt(9/n))
  equ2 <- ((n-1)* sd(d))/9
  data1.df <- rbind(I,c(equ1, equ2))
}

