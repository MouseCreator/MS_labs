library(Deriv)



file_path <- "D:\\tempRepo\\sem5\\MS\\Lab1\\f10.txt"
values <- scan(file_path, what = numeric())


x <- seq(0.00, length.out = length(values), by = 0.01)


f <- function(x) {
  if (x >= 0.01 && x <= length(values) * 0.01) {
    return(values[ceiling(x / 0.01)])
  } else {
    return(NA)
  }
}

DiscreteFourierTransform <- function(x) {
  N = length(x)
  c <- vector("numeric", length = N)
  for (n in 1:N) {
    sum <- 0
    for (m in 1:N) {
      s <- x[m] * exp(-1i*2*pi*n*m/N)
      sum <- sum + s
    }
    sum <- abs(sum)
    c[n] <- sum / N
  }
  return (c)
  
}
fft0 <- function(z, inverse=FALSE) {
  n <- length(z)
  if(n == 0) return(z)
  k <- 0:(n-1)
  ff <- (if(inverse) 1 else -1) * 2*pi * 1i * k/n
  vapply(1:n, function(h) sum(z * exp(ff*(h-1)))/n, complex(1))
}

ExtremumFinder <- function(seq) {
  extremums = c()
  md = 0.05
  n = length(seq)
  for (i in 1:n) {
    na = if(i-2<1) 0 else seq[i-2]
    nb = if(i-1<1) 0 else seq[i-1]
    nc = seq[i]
    nd = if(i+1>n) 0 else seq[i+1]
    ne = if(i+2>n) 0 else seq[i+2]
    
    if (nc - na > md && nc - nb > md && nc - nd > md && nc - ne > md) {
      extremums = c(extremums, i)
    }
  }
  return (extremums)
}

y <- sapply(x, f)
##CUSTOM FFT
i_sequence <- 1:501

dft <- DiscreteFourierTransform(y)
plot(i_sequence, dft, type = "l", xlab = "x", ylab = "f(x)", main = "Plot of y = dft(x)")

###FFT
plot(x, y, type = "l", xlab = "x", ylab = "f(x)", main = "Plot of y = f(x)")

fft_result <- fft(sapply(x, f))

fft_magnitude <- abs(fft_result)

plot(fft_magnitude, type = "l", xlab = "Frequency", ylab = "Magnitude",
     main = "FFT Magnitude of f(x)")

##Subvector
x_sub <- x[1:250]
d_sub <- dft[1:250]
i_sub = 1:250
plot(i_sub, d_sub, type = "l", xlab = "x", ylab = "y", main = "Plot of DFT (first half)")

extremums <- ExtremumFinder(d_sub)
cat("Extremums: ", extremums)

## 25 ## 75 ##


expression <- quote(0) 
for (i in i_sequence) {
  expression <- expression + expression(a1^2)
}


# Use the deriv() function to compute the derivative dF/da1
dF_da1 <- deriv(expression, "a1")
print(dF_da1)

### DER
a1 <- as.name("a1")
a2 <- as.name("a2")
a3 <- as.name("a3")
a4 <- as.name("a4")
a5 <- as.name("a5")
a6 <- as.name("a6")

# Create an expression for the sum
sum_expression <- expression(sum((a1 * x^3 + a2 * x^2 + a3 * x + a4 * sin(pi*x*10) + a5*sin(pi*x*30) + a6 - y)^2) / 2)
# Parse the expression string into an expression
F <- parse(text = sum_expression)
###

print(F)

# Calculate the derivatives
dfFa <- Deriv(F, "a1")
dfFb <- Deriv(F, "a2")
dfFc <- Deriv(F, "a3")
dfFd <- Deriv(F, "a4")
dfFe <- Deriv(F, "a5")
dfFf <- Deriv(F, "a6")
print(dfFa)
print(dfFb)
print(dfFc)
print(dfFd)
print(dfFe)
print(dfFf)
matrXX <- matrix(c(
  sum(x^6), sum(x^5)/2, sum(x^4)/2, sum(sin(10*pi*x)*x^3)/2, sum(sin(30*pi*x)*x^3)/2, sum(x^3)/2,
  sum(x^5)/2, sum(x^4), sum(x^3)/2, sum(sin(10*pi*x)*x^2)/2, sum(sin(30*pi*x)*x^2)/2, sum(x^2)/2,
  sum(x^4)/2, sum(x^3)/2, sum(x^2), sum(sin(10*pi*x)*x)/2, sum(sin(30*pi*x)*x)/2, sum(x)/2,
  sum(x^3*sin(10*x*pi))/2, sum(x^2*sin(10*pi*x))/2, sum(x*sin(10*pi*x))/2, sum(sin(10*pi*x)*sin(10*pi*x)), sum(sin(30*pi*x)*sin(10*pi*x))/2, sum(sin(10*pi*x))/2,
  sum(x^3*sin(30*x*pi))/2, sum(x^2*sin(30*pi*x))/2, sum(x*sin(30*pi*x))/2, sum(sin(10*pi*x)*sin(30*pi*x))/2, sum(sin(30*pi*x)*sin(30*pi*x)), sum(sin(30*pi*x))/2,
  sum(x^3)/2, sum(x^2)/2, sum(x)/2, sum(sin(10*pi*x))/2, sum(sin(30*pi*x)/2), 500
), nrow = 6, byrow = TRUE)

b = c(sum(x^3*y)/2, sum(x^2*y)/2, sum(x*y)/2, sum(sin(50*pi*x)*y)/2, sum(sin(150*pi*x)*y)/2, sum(y)/2)

solutions <- solve(matrXX, b)

print(solutions)

yNew <- 0.1802208976 * x^3 + 0.6130210998 * x^2 + 1.3420222390 * x + 0.1650650717 * sin(pi*x*50) + 0.0315861829*sin(pi*x*150) -0.0002841654
plot(x, yNew, type = "l", xlab = "x", ylab = "f(x)", main = "Plot of y = dft(x)")


read_numeric_vector_from_file <- function(file_path) {
  numeric_vector <- scan(file_path, what = numeric(), sep = " ")
  
  return(numeric_vector)
}

file_path <- "D:\\tempRepo\\sem5\\MS\\Lab1\\f10.txt"
values <- read_numeric_vector_from_file(file_path)


y <- values
x <- seq(0.00, by = 0.01, length.out = length(values))

data <- data.frame(x = x, y = y)
linear_model <- lm(y ~ x + I(x*x) + I(x*x*x) + sin(10 * pi * x) + sin(30 * pi * x), data)

# Print the model summary
summary(linear_model)

est <- predict(linear_model)
plot(x, est, type = "l", xlab = "x", ylab = "f(x)", main = "Plot of Estimation", col="blue")
lines(x,y,col="green", lty=2)
# Print or use the coefficient values as needed
coefficients <- coef(linear_model)
print(coefficients)
