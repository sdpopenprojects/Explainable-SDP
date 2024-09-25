# General Introduction

This repository provides the code and datasets used in this article - The impact of data sampling techniques on the interpretation of just-in-time software defect prediction.

### Environment Preparation

- Python	3. 6. 8

```
eli5           0. 13. 0

imblearn       0. 0 

lime           0. 2. 0. 1

numpy          1. 19. 5

optuna         3. 0. 6

pandas         1. 1. 5

shap           0. 41. 0

sklearn        0. 24. 2

statsmodels    0. 12. 2
```

### Repository Structure

- `./Explainable` : Code directory.
  
  - `datasets`：The dataset of Just-In-Time Software Defedct Prediction(JIT-SDP).
  - `demo-global.py`： Python code for conducting comparative experiments on three data sampling algorithms (under-sampling, over-sampling, and SMOTE) to interpret globally JIT-SDP models across 16 open-source software projects using a time-wise cross-validation setup. Two global interpretation techniques, Permutation and SHAP, were employed. Six classification models were utilized: Naive Bayes (NB), Random Forest (RF), K-Nearest Neighbors (KNN), Decision Tree (DT), Gradient Boosting Method (GBM), and Logistic Regression (LR).
  - `demo-local.py`：Python code was developed to conduct comparative experiments on three data sampling algorithms (under-sampling, over-sampling, and SMOTE) to  interpret locally JIT-SDP models across 16 open-source software projects using a time-wise cross-validation setup. Two local  interpretation techniques, SHAP and LIME, were employed. Six classification models were utilized: Naive Bayes (NB), Random Forest (RF), K-Nearest Neighbors (KNN), Decision Tree (DT), Gradient Boosting Method (GBM), and Logistic Regression (LR).
  - `utilities`:  
    - `tuneParameters.py`:  The Python code for parameter tuning of a model.
    - `timewiseCV.py`:  The Python code for dividing data sets based on time series.
    - `AutoSpearman.py`:  The Python code for eliminating metrics that are highly correlated and redundant.
  - `output`：
    - `gobal`
         - `permutation_scores`:  The scores of  using  *Permutation* to interpret globally the  JIT-SDP  models when using different sampling techniques.
         - `shap_scores`:   The scores of  using  *SHAP* to interpret globally the  JIT-SDP models when using different sampling techniques.
    - `local`
         - `lime_scores`:  The scores of  using  *LIME*  to interpret locally the  JIT-SDP models when using different sampling techniques.
         - `shap_scores`:  The scores of  using  *SHAP* to interpret locally the JIT-SDP models when using different sampling techniques.
  - `results`：
    - `RQ1` : The results of  *Kendall’s Tau coefficient*,  *Top-1 overlap*,  and *Top-3 overlap* value using permutation feature importance and SHAP methods to interpret globally each JIT-SDP model between NONE and sampling algorithms across the datasets used.
    - `RQ2` : The results of  features in  *Top 1 Rank* and *Top 3 Rank* using permutation and SHAP explanation methods to interpret globally each JIT-SDP model between NONE and sampling algorithms across the datasets used.
    - `RQ3` : The results of  *average value of contribution* (size) and *ratio of contradictions* (direction) using  *LIME*  and *SHAP* to locally interpret the  JIT defect prediction models when using NONE and different sampling techniques.
    - `Discuss` :The results of *Kendall’s Tau coefficient*, *Top-1 overlap*, and *Top-3 overlap* values between permutation feature importance and SHAP methods when interpreting globally each JIT-SDP model across the original datasets.  The results of  features in  *Top 1 Rank* and *Top 3 Rank* between permutation and SHAP explanation methods when interpreting globally each JIT-SDP model  across the original datasets. The results of  *average value of contribution* (size) and *ratio of contradictions* (direction) between  *LIME*  and *SHAP* when locally interpreting each JIT-SDP models across the original datasets.

### How to run

- Modify the line in the file `./demo-global.py` , and  `./demo-local.py` , the line are as follows:

  ```R
  # Specify the DIRECTORY path  of dataset
  outpath = "?"
  ```
  
- Run the commands in the terminal.
  
  ```cmd
  $cd your code path
  $python demo-global.py
  $python demo-local.py
  ```
