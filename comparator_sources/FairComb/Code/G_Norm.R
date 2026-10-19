
# Gaussian scaling METHOD FOR NORMALIZATION

G_Norm <- function(A)
{
  library(pracma)
  
  points_g = dim(A)[1]
  members_g = dim(A)[2]
  
  G_Mat = matrix(data=0,nrow =points_g, ncol=members_g);
  
  for(j in 1:members_g)
  {
    M_A = mean(A[,j]);
    STD_A = sd(A[,j]);
    
    for(i in 1:points_g)
    {
      temp1_g = A[i, j]-M_A;
      temp2_g = STD_A*(sqrt(2));
      temp2_g = erf(temp1_g/temp2_g);
      G_Mat[i, j] = max(0,temp2_g);
    }
  }
  return(G_Mat);
} # end of the function
