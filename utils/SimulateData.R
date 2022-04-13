sigma <- function(x){
  return(1/(1+exp(-x)))
}

simulation_cla <- function(n, E){
  
  Age <- rpois(n,65+0.5*E)
  
  Eth <- sample(c(0,1),n,replace=T,prob=c(0.7+0.025*E,0.3-0.025*E))
  
  MI <- sample(c(0,1),n,replace=T,prob=c(0.8,0.2))
  
  Ang <- sapply(0.2*E-0.5+1.3*MI, function(x) 
    sample(c(1,0),1,replace=T,prob=c(sigma(x),1-sigma(x))))
  
  ACE <- sapply(0.3*E-1+0.015*Age+0.001*Eth+1.5*MI, function(x) 
    sample(c(1,0),1,replace=T,prob=c(sigma(x),1-sigma(x))))
  
  NYHA1 <- sample(c(0,1),n,replace=T,prob=c(1-0.175+0.015*E,0.175-0.015*E))
  
  NYHA2 <- sapply(NYHA1, function(x) 
    {ifelse(x==1,0,sample(c(1,0),1,replace=T,prob=c(0.3,0.7)))})
  
  NYHA3 <- sapply(NYHA1+NYHA2, function(x) 
    {ifelse(x==1,0,sample(c(1,0),1,replace=T,prob=c(0.6,0.4)))})
  
  Surv <- sapply(0.4*E+1.5-0.1*(Age-65)-0.05*Eth-1.75*MI-2.5*Ang+0.6*ACE
                 +0.25*NYHA1-0.75*NYHA2-2*NYHA3, 
                 function(x) rlnorm(1, meanlog=x, sdlog=1.5))
  
  Surv <- sapply(Surv, function(x) ifelse(x<=5, 0, 1))
  
  data <- data.frame(cbind(Age,Eth,Ang,MI,ACE,NYHA1,NYHA2,
                           NYHA3,Surv))
  return(data)
}

simulation_reg <- function(n, E, var){
  
  X1 <- rnorm(n,0.8*E,var)
  
  X2 <- rnorm(n,0.4*X1*X1,var)
  
  X3 <- rnorm(n,0.3*E+0.1*X2*X2,var)
  
  Y <- rnorm(n,-0.5*E*E+0.3*X1+0.7*X2,var)
  
  X4 <- rnorm(n,0.1*E*X1,var)
  
  X5 <- rnorm(n,-0.25*E*X4+0.6*Y,var)
  
  X6 <- rnorm(n,0.2*X3*Y-1,var)
  
  X7 <- rnorm(n,-0.6*E+X6,var)
  
  data <- data.frame(cbind(X1,X2,X3,X4,X5,X6,X7,Y))
  return(data)
}

set.seed(2021)
M = 9 # domain numbers
N1 = rpois(M,350)
N2 = rpois(M,350)
E = list()

## generate domain data for classification and regression task
for (i in 1:M) {
  E[[i]] = rnorm(1,mean = 0,sd = 1)
  cla <- simulation_cla(N1[i],E[[i]])
  reg <- simulation_reg(N2[i],E[[i]],var=0.5)
  
  ## output data
  write.csv(cla,file=paste("simulate300_cla",i-1,".csv",sep=""),row.names=FALSE)
  write.csv(reg,file=paste("simulate300_reg",i-1,".csv",sep=""),row.names=FALSE)
}


M = 10 # domain numbers
N = 500
E = c(-3.01,-2.33,-1.72,-1.11,-0.39,0.34,1.01,1.64,2.41,3.29)

## generate domain data for classification and regression task 
# with 500 for each selected domain
for (i in 1:M) {
  cla <- simulation_cla(N,E[i])
  reg <- simulation_reg(N,E[i],var=0.5)
  
  ## output data
  write.csv(cla,file=paste("simulate500_cla",i-1,".csv",sep=""),row.names=FALSE)
  write.csv(reg,file=paste("simulate500_reg",i-1,".csv",sep=""),row.names=FALSE)
}

