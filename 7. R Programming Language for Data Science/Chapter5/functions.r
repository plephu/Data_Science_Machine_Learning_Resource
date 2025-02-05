# All my function


sum_even.function1 <- function(start, end) {
  sum_even <- 0
  for (i in start:end) {
    if (i %% 2 == 0) {
      sum_even <- sum_even + i
    }
  }
  return(sum_even)
}