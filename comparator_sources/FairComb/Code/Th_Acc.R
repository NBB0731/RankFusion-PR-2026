
# Getting the required threshold from th_1 - th_4 (as given in the paper), using 
# any of the 3 equation numbers, and linear(1)/Z-score normalization(2)

Th_Acc <- function(Final_Scores_Norm_2, v_or_b, l_or_z, th_eq)
{
  th_acc_1 =0
  
  x = mean(Final_Scores_Norm_2[,v_or_b])-sd(Final_Scores_Norm_2[,v_or_b])
  y = quantile(Final_Scores_Norm_2[,v_or_b],0.25)
  
  #Eq.-1 .............................................................
  if(x >0)
  {
    th_acc = x
    }else{
    th_acc = mean(Final_Scores_Norm_2[,v_or_b]) 
  }

  # Eq.-2 .............................................................
  
  if(x>=y)
  {
    th_acc_l = x
  }else{
    th_acc_l = mean(Final_Scores_Norm_2[,v_or_b])
  }
  
  
  # Eq.-3 .............................................................
  
  if(x>=0)
  {
    th_acc_z = max(x, y)
  }else{
    th_acc_z = mean(Final_Scores_Norm_2[,v_or_b])
  } 
  
  # ....................................................................
  
  if(l_or_z==1)                     # Linear-scaling
  {
    if(th_eq==1)
    {
      th_acc_1 = th_acc             # th_1
    }else {
      th_acc_1 = th_acc_l           # th_2
    }
  }else{                            #Z-score normalization
    if(th_eq==1)
    {
      th_acc_1 = th_acc             # th_3
    }else{
      th_acc_1 = th_acc_z           # th_4
    }
  }
  
 return(th_acc_1);
} # end of the function


