
Convert_Binary_Gnorm_Variable <- function(SC_LIST_GNORM_SORT_ENPTH, SC_LIST_IND_ENPTH, EN_RES_IND_ENPTH, SEL_MEM_IND_01)
{
  points_enp = dim(SC_LIST_GNORM_SORT_ENPTH)[1] 
  mem_total = dim(SC_LIST_GNORM_SORT_ENPTH)[2] 
  
  mem_enp = sum(SEL_MEM_IND_01[,1]) 
  
  points_enp1 = points_enp+1
  mem_out_count_th = matrix(data=0, nrow = mem_enp, ncol=1)
  MEMBERS_IND_01_ENP = matrix(data=0, nrow = points_enp1, ncol=mem_enp)
  
  sel_mem_ind = matrix(data=0, nrow = mem_enp, ncol=1)
  
  for(j in 1:mem_total)
  {
    if(SEL_MEM_IND_01[j,1]==1)
    {
      sel_mem_ind = union(sel_mem_ind,j)
    }
  }
  sel_mem_ind = as.matrix(setdiff(sel_mem_ind,0))
  
  for(j in 1:mem_enp)
  {
    mem_index = sel_mem_ind[j,1]
    i=1;
    while(SC_LIST_GNORM_SORT_ENPTH[i,mem_index]>0.1)
    {
      i=i+1;
    }
    mem_out_count_th[j,1] = i-1;
   
    MEMBERS_IND_01_ENP[SC_LIST_IND_ENPTH[1:(mem_out_count_th[j,1]), mem_index],j] =1
    MEMBERS_IND_01_ENP[points_enp1,j] = sum(MEMBERS_IND_01_ENP[,j])
  }
  
  return(MEMBERS_IND_01_ENP); 
}