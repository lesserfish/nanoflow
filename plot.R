fp <- "~/Documents/Code/nanoflow/output.csv"
data <- read.csv(fp)
x <- data[1]$x
y <- data[2]$y
z <- data[3]$z

colors <- c()
for(v in z)
{
  if(v > 0.52)
  {
    colors <- c(colors, "red")
  }
  else if(v < 0.48)
  {
    colors <- c(colors, "blue")
  }
  else
  {
    colors <- c(colors, "green")
  }
}
plot(x,y, col=colors, type='p')
