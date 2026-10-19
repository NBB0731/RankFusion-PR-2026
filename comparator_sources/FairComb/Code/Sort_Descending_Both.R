
Sort_Descending_Both <- function(B)
{
  points_sd = dim(B)[1]
  members_sd = dim(B)[2]
  
  B_SORT = matrix(data=0, nrow = points_sd, ncol= members_sd);
  B_IND = matrix(data=0, nrow = points_sd, ncol= members_sd);
  B_COMB = matrix(data=0, nrow = points_sd, ncol= (2*members_sd));
  
  for(i in 1:members_sd)
  {
    temp_sd = sort(B[,i], decreasing= TRUE, index.return = TRUE);
    B_SORT[,i]= as.matrix(temp_sd$x);
    B_IND[,i] = as.matrix(temp_sd$ix);
  }
  B_COMB[,1:members_sd] = B_IND[,]
  B_COMB[,(members_sd+1):(2*members_sd)] = B_SORT[,]
  
  return(B_COMB);
} # end of the function 
