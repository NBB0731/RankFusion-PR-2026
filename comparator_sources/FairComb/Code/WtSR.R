
# WtSR file  ...........................................................

WtSR <- function(sc_list_org, sc_list_norm, wt_alpha)
{

n= dim(sc_list_org)[1];
members2 = dim(sc_list_norm)[2]

sc_rnk = matrix(data=0, nrow =points, ncol=members2);
rnk_list = matrix(data=0, nrow = points, ncol=members2); 

for(j in 1:members2)
{
  temp = sort(sc_list_org[,j], decreasing = TRUE, index.return = TRUE);
  TEMP_RNK = as.matrix(temp$ix);
  for(i in 1:points)
  {
    rnk_list[TEMP_RNK[i,1],j] = i;
  }
}

prop_rnk = matrix(data=0, nrow = points, ncol= members2);
sc_rnk = matrix(data=0, nrow = points, ncol= members2);
wt_alpha_rev = 1-wt_alpha

for(i in 1:members2)
{
  for(j in 1:points)
  {
    prop_rnk[j,i] = (n-rnk_list[j,i]+1)/n;
    sc_rnk[j,i] = (sc_list_norm[j,i]*wt_alpha)+ (prop_rnk[j,i]*wt_alpha_rev);
  }
}

return(sc_rnk); # SP-reduced SCORES (using WtSR)
}

