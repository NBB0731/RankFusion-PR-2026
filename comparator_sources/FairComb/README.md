# Details of this work are under progress.

We here submit the Data and Code files, necessary to interpret, replicate, and build on the findings reported in the paper (under review for publication) titled as "Fairness-aware Score Combination in Outlier Ensemble".

Two zip files are in this repository, each of Data and Code.

Data File :
The Data.zip file contains tweleve benchmark datasets, with the pre-processed versions, as used in the paper (under review for publication). All these files are in form of Comma Separated Values (.csv). 
The benchmark datasets used in the paper (under review for publication) are: ALOI, Ann-thyroid (Thyroid), Arrhythmia 1,2 (Arrhyth1 and Arrhyth2), Cardiotocography (Cardio), HeartDisease (HeartD), InternetAds (IAds), PenDigits (PenD), Pima, SpamBase (Spam), Waveform (Wave), and Wilt. For some of these we use original versions given in UCI repository, while for others we use some versions given in DAMI repository: https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/. For all the versions, we present details in the paper (under review for publication), and the same versions are attached here.


Following are the list of .R files required to run the ensemble selection approach:

1. Base Outlier Detection Methods: 
In our work, we use three base detectors for experimentation: Average k-NN, LOF, and SOD detectors. Other than for Average k-NN, we use ELKI framework to execute rest five detectors on tweleve benchmark datasets, used in the manuscript (under review for publication). ELKI is an open source software, and the link, as provided in the paper (under review for publication), is: https://elki-project.github.io/algorithms/

2. Existing Member Selection Approaches for Outlier Detection Ensembels:
Following are the state-of-art member selection methods for outlier detection ensembles, which we use in this work to build a good ensemble which is accurate as well as diverse (atleast better than random), described as corresponding .R files

(1) Greedy_Select.R: Greedy Selection Method for Ensembles for Outlier Detection [Schubert et. al. 2012]

(2) Vertical_Select.R: Vertical Selection Method for Ensembles for Outlier Detection [Rayana et. al. 2015]

(3) Boost_Select.R:  Unsupervised Boosting-based Member Selection for Outlier Detection [Campos et. al. 2018]

(4) AnD_SELECT.R: Accurate-and-Diverse-Selector for Outlier Ensembles [Mukhriya and Kumar 2021]

3. How to run?
 - Load any of the dataset you want to evaluate for, from Data.zip, in your R-workspace.
 - Without using ELKI-framework: 
   - Rather than executing base detectors using ELKI-framework, you may directly use our scorelists-set (.csv), for all datasets used from Score_List_Set.Zip.
     - Load all Score_List_Sets and Data_Lables (from Data_Labels.zip), for all datasets in your current R-workspace. 
     - Naming format for all SCORE_LIST_SET CSVs are --      Dataset_name(as per paper)_SCORE_LIST_SET.csv
     - Example for importing all 33 scorelists of SpamBase dataset named as - "SpamBase_SCORE_LIST_SET.csv" 
          - SpamBase_SCORE_LIST_SET <- as.matrix(read.csv("D:/R_bin/SpamBase_SCORE_LIST_SET.csv"), header = TRUE). 
          - We use total 33 candidate detectors: 1-11 --- Average k-NN, 12-22 --- LOF, and 23-33 --- SOD
     - Example for importing all 33 scorelists of SpamBase dataset named as - "SpamBase_SCORE_LIST_SET.csv" 
          - SpamBase_DATA_LABELS <- as.matrix(read.csv("D:/R_bin/SpamBase_DATA_LABELS.csv"), header = TRUE)
 - Using ELKI-framework: 
      - For that dataset, which you want to evaluate on, Run base methods: Average k-NN (or Weighted k-NN in ELKI), LOF and SOD using ELKI-framework interface. 
        These csv data files are ready in the form to run as per ELKI-format.
      - Save their resultant scores (last column in their resultant Outlier Score Files ('method_name'-outlier_order.txt-file, e.g., lof-outlier_order.txt) as a matrix in 
        your R-workspace.
      - Add all the score lists in the ensemble input scorelist: SCORE_LIST_SET (as mentioned in Main.R). We use 1-11 lists numbers for 11 variants of Average k-NN, 12-22 for LOF varaints, and 23-33 for SOD variants. 
 - Now, execute the file: "Main.R". Other than 2-Mandatory-Inputs: SCORE_LIST_SET and DATA_LABELS, we in this file have set the default values for other input parameters. 
  Choose any one out of four ODE selecotrs at a time, using selector_choice (1/2/3/4): 1:G-Select, 2:V-select, 3:Boost-select, and 4:AnD-SELECT.
  Choose any one of the score scaling methods, using norm_choice (1/2): 1:Z-norm scores, 2:G-norm scores
  Choose any one of the way for EP-calculation (Default is MAX-EP, and recommended also): 1:MAX-EP, 2:AVG-EP
  Output will be Final Ensemble Score List (normalized) using FairComb (May be updated by FairComb or May be same as Input when no updation needed)
  
  
