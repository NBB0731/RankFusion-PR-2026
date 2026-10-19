
# Author: Akanksha Mukhriya and Rajeev Kumar

# Fairness-aware score-based Combination in Outlier Detection Ensembles (ODE)


# Input: For FairComb .....................................................
# Score_List_SET - matrix of outlier scorelists of the candidate base detectors
# DATA_LABELS - Actual groud truth
# th_1 - threshold for checking significance of Score Parity across normalized scores in ODE
# selector_choice - 1/2/3/4, for opting of any of four ODE selector, used in the paper
# norm_choice - 1/2, for opting any of the two used score scaking methods, Z-norm(1)/G-normO(2)
# ep_max_avg - 1/2, for opting Max/Avg score combination for EP-calculation, Max(1)/ Avg(2)


# Input: For ODE Selectors and others ..................................................
# d - drop rate
# t - threshold percentage for deciding |top-k| by each detector
# Combination Technique - In this work, we use Average for combination of scores
# ht - method_types
# k_p - parameter range
# v_or_b  - ACC: V*-Select(2) / Boost*-Select(3)
# div_num - DIV: V*-Select(5) / Boost*-Select(6)
# th_m ((as given in the paper))
# l_or_z - Normalization: Linear-scaling(1) / Z-score normalization(2)
# th_eq - Equation no (as given in the paper)


# Output:
# En_Final_Score_Avg - Final Combined Outlier score list of an ODE using FairComb
# En_Final_Rocauc - ROC AUC performance of final combined ensemble 



source("Current_prediction.R")
source("Convert_binary.R")
source("Sort_Descending.R")
source("Sort_Descending_Both.R")
source("Selected_en.R")
source("Sel_en2.R")
source("Roc_Auc.R")
source("G_Norm.R")          # Gaussian-Scaling (G-norm)
source("Z_Norm.R")          # Z-score normalization (Z-norm)
source("L_Norm.R")          # Linear-scaling, only used in AnD-SELECT for accuracy threshold 
source("Greedy_Select.R")  # G-Select
source("Vertical_Select.R") # V-Select
source("Boosting.R")
source("Boost_Select.R")
source("ACC_DIV_Scores.R")
source("Th_Acc.R")
source("AnD_SELECT.R")  # AnD-Select
source("Convert_Binary_Gnorm_Variable.R")
source("Convert_Binary_Gnorm_FixedTopk.R")
source("Score_Parity.R")
source("WtSR.R")
source("En_Comb_Parity_EP.R")



# ..........Rather than executing base detectors using ELKI-framework, you can use our scorelist as given in Score_Lists_Set.Zip in csv form.
# ...... Importing SCORE_LISTS for all datasets in the current R-workspace
# ..........Naming format for all SCORE_LIST_SET.csv -- Dataset_name(as per paper)_SCORE_LIST_SET.csv
# ..........Example for importing all 22 scorelists of SpamBase dataset named as - "SpamBase_SCORE_LIST_SET.csv" in Data.zip

# SpamBase_SCORE_LIST_SET <- as.matrix(read.csv("D:/R_bin/SpamBase_SCORE_LIST_SET.csv"), header = TRUE)
# as we use total 22 candidate detectors: 1-11 --- Average k-NN and 12-22 --- LOF
# SpamBase_DATA_LABELS <- as.matrix(read.csv("D:/R_bin/SpamBase_DATA_LABELS.csv"), header = TRUE)

#...........................................................................
#...........................................................................


# Load SCORE_LIST_SET of all candidate detectors from Score_List_Set.zip, 
# or run LOF, SOD, and Avg k-NN detectors separately using ELKI-framework

# Load "DATA_LABLES" from Data_Labels.zip. Although the full data files (.csv) also contains labels 
# in their last columns, but these CSVs only contain Lables, in the form of: outlier- 1, and inlier- 0

# These two are MANDATORY INPUTS to exeute code file from Main.R, 
# for other inputs, default values as used in Main.R (this file) will be used 
SCORE_LIST_SET = as.matrix(SPAM_SCORE_LIST_SET) # after loading SPAM_SCORE_LIST_SET from Score_Lists_Set.zip
DATA_LABELS = as.matrix(SPAM_DATA_LABELS) # from Data_Labels.zip

points = dim(SCORE_LIST_SET)[1]
members = dim(SCORE_LIST_SET)[2]
out_count2 = sum(DATA_LABELS[,1])

SCORE_LIST_SET_IND = as.matrix(Sort_Descending(SCORE_LIST_SET))
SCORE_LIST_SET_NORM = as.matrix(Z_Norm(SCORE_LIST_SET))

SCORE_LIST_SET_NORM_G = as.matrix(G_Norm(SCORE_LIST_SET))
SCORE_LIST_SET_GNORM_SORT = (as.matrix(Sort_Descending_Both(SCORE_LIST_SET_NORM_G)))[,(members+1):(2*members)]

k= out_count2
selector_choice = 1 # 1/2/3/4 for each of the four ensemble selectors
norm_choice = 1 # 1/2, 1: Z_Norm scaling, 2: G_Norm scaling
ep_max_avg = 1 # 1/2, choice for EP-calculations through Max(1)/Avg(2) combination
th_1 = 0.04 # threshold for SP-significance, we use it as 4%


if(selector_choice==1)
{
# Greedy_Select ...................................................

Selected_Greedy = as.matrix(Greedy_Select(SCORE_LIST_SET, SCORE_LIST_SET_NORM, SCORE_LIST_SET_IND, k))
EN_Greedy_ORG = as.matrix(Selected_en(Selected_Greedy, SCORE_LIST_SET))
EN_Sel_Scores = EN_Greedy_ORG
En_Sel_Ind = Selected_Greedy
}else if(selector_choice==2){
# Vertical_Select ...................................................

Selected_Vertical = as.matrix(Vertical_Select(SCORE_LIST_SET, SCORE_LIST_SET_NORM, SCORE_LIST_SET_IND))
EN_Vertical_ORG = as.matrix(Selected_en(Selected_Vertical, SCORE_LIST_SET))
EN_Sel_Scores = EN_Vertical_ORG
En_Sel_Ind = Selected_Vertical
}else if(selector_choice==3){
# Boost_Select ....................................................

t = 0.05    # for Boost-select
d =  0.25   # for Boost-select

Selected_Boost = as.matrix(Boost_Select(SCORE_LIST_SET, SCORE_LIST_SET_NORM, SCORE_LIST_SET_IND, t, d))
EN_Boost_ORG = as.matrix(Selected_en(Selected_Boost, SCORE_LIST_SET))
EN_Sel_Scores = EN_Boost_ORG
En_Sel_Ind = Selected_Boost
}else{

# AnD_SELECT ........................................................................

k = out_count2

ACC_DIV_temp = as.matrix(ACC_DIV_Scores(SCORE_LIST_SET_NORM, SCORE_LIST_SET_IND, k, t))
ACC_temp = as.matrix(ACC_DIV_temp[,1:3])
DIV_temp = as.matrix(ACC_DIV_temp[,4:7])

acc_count = dim(ACC_temp)[2]
div_count = dim(DIV_temp)[2]
total_scores = acc_count + div_count

Final_Scores = matrix(data=0, nrow = members, ncol=total_scores)
Final_Scores[,1:acc_count] = ACC_temp[,]
Final_Scores[,(acc_count+1):total_scores] = DIV_temp[,]

k_p = as.matrix(cbind(5,10,20,30,40,50,60,70,80,90,100))
hm = dim(k_p)[2]
ht = members/hm

if(l_or_z==1)
{
  Final_Scores_Norm1 = as.matrix(L_Norm(Final_Scores))  
}else{
  Final_Scores_Norm1 = as.matrix(Z_Norm(Final_Scores))   
}

v_or_b = 3        # ACC: Boost*-Select(3)
div_num = 6       # DIV: Boost*-Select(6)
th_m = 0.7        # (as given in the paper) - 0.7 (70%) / 0.8(80%)
l_or_z = 1        # Normalization: Linear-scaling(1) / Z-score normalization(2)
th_eq = 2         # th_2: Equation no (as given in the paper)


Selected_AnD = as.matrix(AnD_SELECT(Final_Scores_Norm1, ht, k_p, v_or_b, div_num, th_m, l_or_z, th_eq))
Selected_AnD_mat = matrix(data=0, nrow = members, ncol=1)
sel_and_count = dim(Selected_AnD)[1]
for(i in 1:sel_and_count)
{
  Selected_AnD_mat[Selected_AnD[i,1],1] = 1
}
EN_AnD_Boost_ORG = as.matrix(Selected_en(Selected_AnD_mat, SCORE_LIST_SET))
EN_Sel_Scores = EN_AnD_Boost_ORG
EN_Sel_Ind = Selected_AnD_mat
} # end if-else for selector_choice .................................


# Score-normalization/scaling for selected set of ensemble detectors

if(norm_choice==1) # Z_norm scores
{
 EN_Score_Norm = as.matrix(Z_Norm(EN_Sel_Scores))
}else{
  EN_Score_Norm = as.matrix(G_Norm(EN_Sel_Scores))  # Gaussian-scaled scores (G_Norm)
}
sel_count = dim(EN_Score_Norm)[2]
En_Score_Norm_SORT = (as.matrix(Sort_Descending_Both(EN_Score_Norm)))[,(sel_count+1):(2*sel_count)]
SELECTED_IND_01 = as.matrix(En_Sel_Ind)
MEMBERS_IND = as.matrix(Convert_Binary_Gnorm_Variable(SCORE_LIST_SET_GNORM_SORT, SCORE_LIST_SET_IND, EN_RES_IND, SELECTED_IND_01))

# Code for checking Score-Parity (SP) ..................................................

SP = Score_Parity(En_Sel_Ind, MEMBERS_IND, En_Score_Norm_SORT)

wt_alpha=1  # original scaled scores for ensemble without any SP-reduction
EN_Score_List_Org = WtSR(EN_Sel_Scores, En_Score_Norm, wt_alpha)

wt_alpha=0.5  # scaled-scores with WtSR for ensemble with SP-reduction
EN_Score_List_WtSR = WtSR(EN_Sel_Scores, En_Score_Norm, wt_alpha)

if(SP>=th_1)
{
  if(ep_max_avg==1) # Max-EP: EP-calculations through Max Combination: Recommended
  {
  for(i in 1:points)
  {
    EN_Score_MAX[i,1] = max(EN_Score_List_Org[i,]);
    EN_Score_MAX_WtSR[i,1] = max(EN_Score_List_WtSR[i,]);
  }
  EN_Score_MAX_score = sort(EN_Score_MAX[,1], decreasing = TRUE, index.return = TRUE);
  EN_Score_MAX_ind <- as.matrix(EN_Score_MAX_score$ix);
  
  EN_Score_MAX_WtSR_score = sort(EN_Score_MAX_WtSR[,1], decreasing = TRUE, index.return = TRUE);
  EN_Score_MAX_WtSR_ind <- as.matrix(EN_Score_MAX_score_WtSR$ix);
  
  EP_Old = EN_Comb_Parity_EP(EN_Score_MAX_ind, MEMBERS_IND)
  EP_New = EN_Comb_Parity_EP(EN_Score_MAX_WtSR_ind, MEMBERS_IND)
  
  if(EP_New<EP_Old) # Combination-Fairness gets improved
  {
    En_Final_Score_Avg = as.matrix(Current_prediction(EN_Score_List_WtSR))
  }else{ # checking with fixed thresholds for top-k% - 5%, 10%, and 15%
    
    EP_FixedTopk_Avg = 0
    # topk = 5% ....................................................
    MEMBERS_IND = as.matrix(Convert_Binary_Gnorm_FixedTopk(SCORE_LIST_SET_GNORM_SORT, SCORE_LIST_SET_IND, EN_RES_IND, SELECTED_IND_01, 0.05))
    SP = Score_Parity(En_Sel_Ind, MEMBERS_IND, En_Score_Norm_SORT)
    if(SP>=th_1)
    {
      EP_New = EN_Comb_Parity_EP(EN_Score_MAX_WtSR_ind, MEMBERS_IND)
      EP_FixedTopk_Avg = EP_FixedTopk_Avg+EP_New
    }
    # topk = 10% ....................................................
    MEMBERS_IND = as.matrix(Convert_Binary_Gnorm_FixedTopk(SCORE_LIST_SET_GNORM_SORT, SCORE_LIST_SET_IND, EN_RES_IND, SELECTED_IND_01, 0.1))
    SP = Score_Parity(En_Sel_Ind, MEMBERS_IND, En_Score_Norm_SORT)
    if(SP>=th_1)
    {
      EP_New = EN_Comb_Parity_EP(EN_Score_MAX_WtSR_ind, MEMBERS_IND)
      EP_FixedTopk_Avg = EP_FixedTopk_Avg+EP_New
    }
    # topk = 15% ....................................................
    MEMBERS_IND = as.matrix(Convert_Binary_Gnorm_FixedTopk(SCORE_LIST_SET_GNORM_SORT, SCORE_LIST_SET_IND, EN_RES_IND, SELECTED_IND_01, 0.15))
    SP = Score_Parity(En_Sel_Ind, MEMBERS_IND, En_Score_Norm_SORT)
    if(SP>=th_1)
    {
      EP_New = EN_Comb_Parity_EP(EN_Score_MAX_WtSR_ind, MEMBERS_IND)
      EP_FixedTopk_Avg = EP_FixedTopk_Avg+EP_New
    }
    
    if(EP_FixedTopk_Avg<EP_Old) # Combination-Fairness gets improved
    {
      En_Final_Score_Avg = as.matrix(Current_prediction(EN_Score_List_WtSR))
    }else{
      En_Final_Score_Avg = as.matrix(Current_prediction(EN_Score_List_Org))
    }
  }
  }else{ # Avg-EP: EP-calculations through Avg Combination
  for(i in 1:points)
  {
    EN_Score_AVG[i,1] = mean(EN_Score_List_Org[i,]);
    EN_Score_AVG_WtSR[i,1] = mean(EN_Score_List_WtSR[i,]);
  }
 
  EN_Score_AVG_score = sort(EN_Score_AVG[,1], decreasing = TRUE, index.return = TRUE);
  EN_Score_AVG_ind <- as.matrix(EN_Score_AVG_score$ix);
  
  EN_Score_AVG_WtSR_score = sort(EN_Score_AVG_WtSR[,1], decreasing = TRUE, index.return = TRUE);
  EN_Score_AVG_WtSR_ind <- as.matrix(EN_Score_AVG_score_WtSR$ix);
  
  EP_Old = EN_Comb_Parity_EP(EN_Score_AVG_ind, MEMBERS_IND)
  EP_New = EN_Comb_Parity_EP(EN_Score_AVG_WtSR_ind, MEMBERS_IND)
  
  if(EP_New<EP_Old) # Combination-Fairness gets improved
  {
    En_Final_Score_Avg = as.matrix(Current_prediction(EN_Score_List_WtSR))
  }else{ # checking with fixed thresholds for top-k% - 5%, 10%, and 15%
    
    EP_FixedTopk_Avg = 0
    # topk = 5% ....................................................
    MEMBERS_IND = as.matrix(Convert_Binary_Gnorm_FixedTopk(SCORE_LIST_SET_GNORM_SORT, SCORE_LIST_SET_IND, EN_RES_IND, SELECTED_IND_01, 0.05))
    SP = Score_Parity(En_Sel_Ind, MEMBERS_IND, En_Score_Norm_SORT)
    if(SP>=th_1)
    {
      EP_New = EN_Comb_Parity_EP(EN_Score_AVG_WtSR_ind, MEMBERS_IND)
      EP_FixedTopk_Avg = EP_FixedTopk_Avg+EP_New
    }
    # topk = 10% ....................................................
    MEMBERS_IND = as.matrix(Convert_Binary_Gnorm_FixedTopk(SCORE_LIST_SET_GNORM_SORT, SCORE_LIST_SET_IND, EN_RES_IND, SELECTED_IND_01, 0.1))
    SP = Score_Parity(En_Sel_Ind, MEMBERS_IND, En_Score_Norm_SORT)
    if(SP>=th_1)
    {
      EP_New = EN_Comb_Parity_EP(EN_Score_AVG_WtSR_ind, MEMBERS_IND)
      EP_FixedTopk_Avg = EP_FixedTopk_Avg+EP_New
    }
    # topk = 15% ....................................................
    MEMBERS_IND = as.matrix(Convert_Binary_Gnorm_FixedTopk(SCORE_LIST_SET_GNORM_SORT, SCORE_LIST_SET_IND, EN_RES_IND, SELECTED_IND_01, 0.15))
    SP = Score_Parity(En_Sel_Ind, MEMBERS_IND, En_Score_Norm_SORT)
    if(SP>=th_1)
    {
      EP_New = EN_Comb_Parity_EP(EN_Score_AVG_WtSR_ind, MEMBERS_IND)
      EP_FixedTopk_Avg = EP_FixedTopk_Avg+EP_New
    }
    if(EP_FixedTopk_Avg<EP_Old) # Combination-Fairness gets improved
    {
      En_Final_Score_Avg = as.matrix(Current_prediction(EN_Score_List_WtSR))
    }else{
      En_Final_Score_Avg = as.matrix(Current_prediction(EN_Score_List_Org))
    }
  }
  }
  
}else{# end of if-condition for checking significant Score-Parity acorss ensemble
  En_Final_Score_Avg = as.matrix(Current_prediction(EN_Score_List_Org))
}

# Final Ensemble Accuracy (ROC AUC) with FairComb .....................
En_Final_Rocauc = as.numeric(Roc_Auc(En_Final_Score_Avg, DATA_LABELS))
round(En_Final_Rocauc,4)












