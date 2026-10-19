
En_Comb_Parity_EP <- function(EN_RES_IND, MEMBERS_IND)
{
  point1 = dim(EN_RES_IND)[1]
  mem1 = dim(MEMBERS_IND)[2]
  topk_1 = point1+1
  topk_2 = point1+2
  
  TK_EN_LOC = matrix(data=0, nrow = topk_2, ncol=mem1)
  all_mem_out_count = 0
  mem_out =0
  
  for(j in 1:mem1)
  {
    all_mem_out_count = all_mem_out_count + sum(MEMBERS_IND[,j])
  }
  
  u_mem = -1
  for(j in 1:mem1)
  {
    for(i in 1:point1)
    {
      if(MEMBERS_IND[i,j]==1)
      {
        u_mem <- union(u_mem, i) # for union 
        
        TK_EN_LOC[i,j] = match(i, EN_RES_IND[,1])
        mem_out = max(MEMBERS_IND[topk_1,]) # max of both mem_out_count
      }
    }
  }
  u_mem <- as.matrix(setdiff(u_mem, -1))
  
  for(j in 1:mem1)
  {
    mem_k = MEMBERS_IND[topk_1,j]   # last row (points+1)th
    Ideal_En_Agg_Rank = (mem_k*(mem_k+1))/2
    
    TK_EN_LOC[topk_1, j] = sum(TK_EN_LOC[,j])
    TK_EN_LOC[topk_2, j] = Ideal_En_Agg_Rank/TK_EN_LOC[topk_1, j]
  }
 
  ep = max(TK_EN_LOC[topk_2, ])-min(TK_EN_LOC[topk_2, ])
  
  return(ep);
}
