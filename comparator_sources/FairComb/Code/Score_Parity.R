
Score_Parity<- function(En_Sel_Ind, MEMBERS_IND, En_Sel_Score_Norm_Sort)
{
sel_c = sum(En_Sel_Ind[,1])
points = dim(En_Sel_Score_Norm_Sorted)[1]
points_1 = points+1

Avg_Topk_SC_NORM_Sel = matrix(data=0, nrow = sel_c, ncol=1) 
Avg_SD_Topk_SC_NORM_Sel = matrix(data=0, nrow = sel_c, ncol=1) 
for(i in 1:sel_c)
{
  Avg_Topk_SC_NORM_Sel[i,1] = mean(En_Sel_Score_Norm_Sort[1:(MEMBERS_IND[points_1,i]), i])
  Avg_SD_Topk_SC_NORM_Sel[i,1] = mean(En_Sel_Score_Norm_Sort[1:(MEMBERS_IND[points_1,i]), i])+sd(En_Sel_Score_Norm_Sorted[1:(MEMBERS_IND[points_1,i]), i])
}

P1 = max(Avg_Topk_SC_NORM_Sel[,1])-min(Avg_Topk_SC_NORM_Sel[,1])
P2 = max(Avg_SD_Topk_SC_NORM_Sel[,1])-min(Avg_SD_Topk_SC_NORM_Sel[,1])
SP = max(P1, P2)

return(SP);
}