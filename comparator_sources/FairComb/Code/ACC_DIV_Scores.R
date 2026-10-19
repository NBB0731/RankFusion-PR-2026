
ACC_DIV_Scores <- function(SCORE_LIST_SET_NORM, SCORE_LIST_SET_IND, k, t)
{
  library(weights)
  
  points = dim(SCORE_LIST_SET_NORM)[1]
  members = dim(SCORE_LIST_SET_NORM)[2]
  
  ACC_Scores_Mat = matrix(data=0, nrow = members, ncol=3)
  
  acc_values = matrix(data=0, nrow = members, ncol=1)
  acc2_values = matrix(data=0, nrow = members, ncol=1)
  acc3_values = matrix(data=0, nrow = members, ncol=1)
  
  # ACC_G-select .................................................
  
  u1 = matrix(data=0, nrow =k, ncol=1)
  for(i in 1:members)
  {
    u1 = union(u1, SCORE_LIST_SET_IND[1:k,i]);
  }
  u1 <- as.matrix(u1);
  
  target = matrix(data =0, nrow = points, ncol = 1);
  
  for(i in 1:points)
  {
    if((is.element(i,u1)))
    { 
      target[i,1] = 1;      
    }
  }
  
  uni_k <- dim(u1)[1];
  uni_k_inv <- points - uni_k;
  wout = 1/(2*uni_k);
  win = 1/(2*uni_k_inv);
  
  W = matrix(data =0, nrow = points, ncol = 1);
  
  for(i in 1:points)
  {
    if(target[i,1] == 1)
    { 
      W[i,1] = wout;    
    }
    else
    { 
      W[i,1] = win;  
    }
  }
  corr = matrix(, nrow = members, ncol = 4);
  for(i in 1:members)
  {
    corr[i,] = wtd.cor(SCORE_LIST_SET_NORM[,i],target, W[,1]);
    acc_values[i,] = corr[i,1]
  }
  
  # ACC_V-select .........................................................
  
  target_v = matrix(data=0, nrow = points, ncol=1)
  for(i in 1: points)
  {
    target_v[i,1] = mean(SCORE_LIST_SET_NORM[i,])
  }
  
  rnk_list = matrix(data=0, nrow = points, ncol=1); 
  
  temp = sort(target_v[,1], decreasing = TRUE, index.return = TRUE);
  TEMP_RNK = as.matrix(temp$ix);
  for(i in 1:points)
  {
    rnk_list[TEMP_RNK[i,1],1] = i;
  }
  W_V = matrix(data =0, nrow = points, ncol = 1);
  
  for(i in 1:points)
  {
    W_V[i,1] = 1/(rnk_list[i,1])
  }
  
  corr_v = matrix(data=0, nrow = members, ncol =4);
  
  for(i in 1:members)
  {
    corr_v[i,]  <- wtd.cor(SCORE_LIST_SET_NORM[,i], target_v[,1],W_V[,1]);
  }
  acc2_values[,1] = corr_v[,1]
  
  # ACC_Boost_Select  ..................................................
  
  temp_target = matrix(data=0, nrow = points, ncol=1)
  temp_target_SORT = matrix(data=0, nrow = points, ncol=1)
  temp_target_IND = matrix(data=0, nrow = points, ncol=1)
  
  for(i in 1:points)
  {
    temp_target[i,1] = mean(SCORE_LIST_SET_NORM[i,])  
  }
  
  top_nt = floor(points*t)
  
  tempp = sort(temp_target[,1], decreasing= TRUE, index.return = TRUE);
  temp_target_SORT[,1]= as.matrix(tempp$x);
  temp_target_IND[,1] = as.matrix(tempp$ix);
  
  target3 = Convert_binary(temp_target, t)
  
  W3 = matrix(data =0, nrow = points, ncol = 1);
  
  wout = 1/(2*top_nt);
  win = 1/(2*(points-top_nt));
  
  for(i in 1:points)
  {
    if(target3[i,1] == 1)
    { 
      W3[i,1] = wout;    
    }
    else
    { 
      W3[i,1] = win;  
    }
  }
  corr_bs = matrix(, nrow = members, ncol = 4);
  
  for(i in 1:members)
  {
    corr_bs[i,] = wtd.cor(SCORE_LIST_SET_NORM[,i],target3, W3[,1]);
  } 
  acc3_values[,1] = corr_bs[,1]
  
  ACC_Scores_Mat[,1] = acc_values[,1]
  ACC_Scores_Mat[,2] = acc2_values[,1]
  ACC_Scores_Mat[,3] = acc3_values[,1]
  
  DIV_Scores_Mat = matrix(data=0, nrow = members, ncol=4)
  library(weights);
  
  # DIVERSITY_GREEDY  ...............................................................................
  
  corr1 = matrix(data=0, nrow = 1, ncol = 4);
  temp1 = matrix(data=0, nrow = members, ncol=members)
  temp1_sum = matrix(data=0, nrow = members, ncol=1)
  temp1_hetero_sum = matrix(data=0, nrow = members, ncol=1)
  
  for(i in 1:members)
  {
    for(j in 1:members)
    {
      if(i!=j)
      {
        corr1[1,] = wtd.cor(SCORE_LIST_SET_NORM[,i],SCORE_LIST_SET_NORM[,j], W[,1]);
        temp1[i,j] = corr1[1,1]
        temp1_sum[i,1] = temp1[i,j] + temp1_sum[i,1]
      }
    }
  }
  # DIVERSITY_GENERAL  ...............................................................................
  
  corr1 = matrix(data=0, nrow = 1, ncol = 4);
  temp_gen = matrix(data=0, nrow = members, ncol=members)
  temp_gen_sum = matrix(data=0, nrow = members, ncol=1)
  temp1_gen_hetero_sum = matrix(data=0, nrow = members, ncol=1)
  
  for(i in 1:members)
  {
    for(j in 1:members)
    {
      if(i!=j)
      {
        corr1[1,] = cor(SCORE_LIST_SET_NORM[,i],SCORE_LIST_SET_NORM[,j], method = "pearson");
        temp_gen[i,j] = corr1[1,1]
        temp_gen_sum[i,1] = temp_gen[i,j] + temp_gen_sum[i,1]
      }
    }
  }
  
  # DIVERSITY_VERTICAL  ...............................................................................
  
  corr3 = matrix(data=0, nrow = 1, ncol = 4);
  temp3 = matrix(data=0, nrow = members, ncol=members)
  temp3_sum = matrix(data=0, nrow = members, ncol=1)
  
  for(i in 1:members)
  {
    for(j in 1:members)
    {
      if(i!=j)
      {
        corr3[1,] = wtd.cor(SCORE_LIST_SET_NORM[,i],SCORE_LIST_SET_NORM[,j], W_V[,1]);
        temp3[i,j] = corr3[1,1]
        temp3_sum[i,1] = temp3[i,j] + temp3_sum[i,1]
      }
    }
  }
  
  # DIVERSITY_BOOST_SEL  ...............................................................................
  
  corr4 = matrix(data=0, nrow = 1, ncol = 4);
  temp4 = matrix(data=0, nrow = members, ncol=members)
  temp4_sum = matrix(data=0, nrow = members, ncol=1)
  
  for(i in 1:members)
  {
    for(j in 1:members)
    {
      if(i!=j)
      {
        corr4[1,] = wtd.cor(SCORE_LIST_SET_NORM[,i], SCORE_LIST_SET_NORM[,j], W3[,1]);
        temp4[i,j] = corr4[1,1]
        temp4_sum[i,1] = temp4[i,j] + temp4_sum[i,1]
      }
    }
  }
  
  tm = members-1
  for(i in 1:members)
  {
    DIV_Scores_Mat[i,1] = 1-(temp1_sum[i,1]/tm)
    DIV_Scores_Mat[i,2] = 1-(temp3_sum[i,1]/tm)
    DIV_Scores_Mat[i,3] = 1-(temp4_sum[i,1]/tm)
    DIV_Scores_Mat[i,4] = 1-(temp_gen_sum[i,1]/tm)
  }
  #return(DIV_Scores_Mat);
  
  ACC_DIV_Scores_Mat = matrix(data=0, nrow = members, ncol=7)
  
  ACC_DIV_Scores_Mat[,1:3] = ACC_Scores_Mat[,]
  ACC_DIV_Scores_Mat[,4:7] = DIV_Scores_Mat[,]
  
  
  return(ACC_DIV_Scores_Mat);
} # end of the function
