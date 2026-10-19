
# The AnD select function for Ensemble Construction


AnD_SELECT <- function(Final_Scores_Norm_1, ht, k_p, v_or_b, div_num, th_m, l_or_z, th_eqn)
{
 
  members = dim(Final_Scores_Norm_1)[1]  
  temp_pruned = matrix(data=1, nrow = members, ncol=1)
  
  hm = dim(k_p)[2]


th_acc_2 = as.numeric(Th_Acc(Final_Scores_Norm_1, v_or_b, l_or_z, th_eqn))

for(i in 1:members) # members
{
  if(Final_Scores_Norm_1[i,v_or_b]<th_acc_2)
  {
    temp_pruned[i,1] =0
  }
}

#k_p = as.matrix(cbind(5,10,20,30,40,50,60,70,80,90,100))

E_ht <- c(-1)

for(i in 1:ht)
{
  E_hm <- c(-1)
  
  x_lb = ((i-1)*hm)+1
  x_ub = i*hm
  
  temp_mono = floor(hm*th_m)
  
  xlb_mono_percent = x_lb+(temp_mono)-1
  xub_mono_percent = x_ub-(temp_mono)+1
  
  corr_test_3 = cor(Final_Scores_Norm1[x_lb:xlb_mono_percent,v_or_b],k_p[1,1:temp_mono], method = "spearman");
  corr_test_4 = cor(Final_Scores_Norm1[xub_mono_percent:x_ub,v_or_b],k_p[1,(hm-temp_mono+1):hm], method = "spearman");
  
  corr_test_5 = cor(Final_Scores_Norm1[x_lb:(xlb_mono_percent+1),v_or_b],k_p[1,1:(temp_mono+1)], method = "spearman");
  corr_test_6 = cor(Final_Scores_Norm1[(xub_mono_percent-1):x_ub,v_or_b],k_p[1,(hm-temp_mono):hm], method = "spearman");
  
  
  corr_test3_abs = abs(corr_test_3)
  corr_test4_abs = abs(corr_test_4)
  corr_test5_abs = abs(corr_test_5)
  corr_test6_abs = abs(corr_test_6)
  
  case=0

  corr_test1 = cor(Final_Scores_Norm1[x_lb:x_ub,v_or_b],k_p[1,1:hm], method = "spearman");
  abs(corr_test1)
  
  if(isTRUE(all.equal(abs(corr_test1),1))) #Case 1
  {
    case = 1
    if(Final_Scores_Norm_1[x_lb,v_or_b] > Final_Scores_Norm_1[x_ub,v_or_b])
    {
      p_sel = x_lb
    }else{
      p_sel = x_ub
    }
    if(temp_pruned[p_sel,1]==1)
    {
      E_hm = union(E_hm, p_sel)
    }
  }else
    if( ( (isTRUE(all.equal(corr_test3_abs,1))) || ((isTRUE(all.equal(round(corr_test3_abs,1),1))) && (isTRUE(all.equal(round(corr_test5_abs,1),1))))) ||
        ( (isTRUE(all.equal(corr_test4_abs,1))) || ((isTRUE(all.equal(round(corr_test4_abs,1),1))) && (isTRUE(all.equal(round(corr_test6_abs,1),1))))) )
    {   
        case = 2
        if((isTRUE(all.equal(corr_test3_abs,1)))||(isTRUE(all.equal(corr_test4_abs,1))))
        {
         if((isTRUE(all.equal(corr_test3_abs,1))))
         {  
         if(corr_test_3>0) # == +1
         {
          temp_ub = x_ub
          l_count=1
          while(l_count<=hm)
          {
            if((Final_Scores_Norm_1[temp_ub, v_or_b]<th_acc_2))
            {
             temp_ub = temp_ub-1
              l_count = l_count+1
            }else{break}
          }
          if(temp_ub>(x_lb-1)) {E_hm = union(E_hm, temp_ub)}
        }else{
          temp_lb = x_lb
          l_count=1
          while(l_count<=hm)
          {
            if((Final_Scores_Norm_1[temp_lb, v_or_b]<th_acc_2))
            {
             temp_lb = temp_lb+1
              l_count = l_count+1
            }else{break}
          }
          if(temp_lb<(x_ub+1)) {E_hm = union(E_hm, temp_lb)}
        } # end else
      }else{            
        if(corr_test_4>0) # == +1
        {
          temp_ub = x_ub
          l_count=1
          while(l_count<=hm) 
          {
            if((Final_Scores_Norm_1[temp_ub, v_or_b]<th_acc_2))
            {
              temp_ub = temp_ub-1
              l_count = l_count+1
            }else{break}
          }
          if(temp_ub>(x_lb-1)){E_hm = union(E_hm, temp_ub)}
        }else{
          temp_lb = x_lb
          l_count =1
          while(l_count<=hm)
          {
            if((Final_Scores_Norm_1[temp_lb, v_or_b]<th_acc_2))
            {
              temp_lb = temp_lb+1
              l_count = l_count+1
            }else{break}
          }
          if(temp_lb<(x_ub+1)) {E_hm = union(E_hm, temp_lb)}
        } #end inner else
      } # end 2nd outer else
    } else{ # end 2nd inner 
            if((isTRUE(all.equal(round(corr_test3_abs,1),1)))&&(isTRUE(all.equal(round(corr_test5_abs,1),1))))
            {
              if(corr_test_3>0) # == +1
              {
                temp_ub = x_ub
                l_count=1
                while(l_count<=hm)
                {
                  if((Final_Scores_Norm_1[temp_ub, v_or_b]<th_acc_2))
                  {
                    temp_ub = temp_ub-1
                    l_count = l_count+1
                  }else{break}
                }
                if(temp_ub>(x_lb-1)) {E_hm = union(E_hm, temp_ub)}
              }else{
                temp_lb = x_lb
                l_count=1
                while(l_count<=hm)
                {
                  if((Final_Scores_Norm_1[temp_lb, v_or_b]<th_acc_2))
                  {
                    temp_lb = temp_lb+1
                    l_count = l_count+1
                  }else{break}
                }
                if(temp_lb<(x_ub+1)) {E_hm = union(E_hm, temp_lb)}
              } # end else
            }else{ # end-if
              if((isTRUE(all.equal(round(corr_test4_abs,1),1)))&&(isTRUE(all.equal(round(corr_test6_abs,1),1))))
              {
              if(corr_test_4>0) # == +1
              {
                temp_ub = x_ub
                l_count=1
                while(l_count<=hm) 
                {
                  if((Final_Scores_Norm_1[temp_ub, v_or_b]<th_acc_2))
                  {
                    temp_ub = temp_ub-1
                    l_count = l_count+1
                  }else{break}
                }
                if(temp_ub>(x_lb-1)){E_hm = union(E_hm, temp_ub)}
              }else{
                temp_lb = x_lb
                l_count =1
                while(l_count<=hm)
                {
                  if((Final_Scores_Norm_1[temp_lb, v_or_b]<th_acc_2))
                  {
                    temp_lb = temp_lb+1
                    l_count = l_count+1
                  }else{break}
                }
                if(temp_lb<(x_ub+1)) {E_hm = union(E_hm, temp_lb)}
              } #end inner else
              }# end if
          } # end 3rd outer else
      } # end 2nd outer else - 
        
    } else{       # Case 3
      case = 3
      prune_set <- c(-1)
      for(ki in 1:hm)
      {    
        if(temp_pruned[ki+x_lb-1, 1]==1)
        {
          prune_set <- union(prune_set, ki+x_lb-1)
        }
      }
      prune_set2 <- as.matrix(setdiff(prune_set,-1))
      px = dim(prune_set2)[1]
      fval_px = matrix(data=0, nrow =px, ncol=2)
      
      if(px) # if prune_set2 is non zero set
      {
        for(ki in 1:px)
        {
          fval_px[ki,1] = Final_Scores_Norm1[prune_set2[ki,1],v_or_b]
          fval_px[ki,2] = Final_Scores_Norm1[prune_set2[ki,1],div_num] # 4 for diversity
        }
      }
      
      fval_max_acc = which.max(fval_px[,1]) # max acc detector b/w x_lb and x_ub
      E_hm = union(E_hm, prune_set2[fval_max_acc,1])
      fval_max_div = which.max(fval_px[,2]) # max div detector b/w x_lb and x_ub
      E_hm = union(E_hm, prune_set2[fval_max_div,1])
      
    } # end of else 
  
  E_hm2 <- setdiff(E_hm, -1) # removing dummy set element -1 from selected homo ens
  E_ht <- union(E_ht, E_hm2)
} # end-for of hetero Ht ......

E_ht2 <- setdiff(E_ht, -1)
#E_ht2
sel_en = as.matrix(E_ht2)

return(E_ht2);

} # end of the function

