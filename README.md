# General Introduction

This repository provides the code and datasets used in this article - Zhiqiang Li, Qiannan Du, Hongyu Zhang, Xiao-Yuan Jing, Fei Wu. An empirical study of data sampling techniques for just-in-time software defect prediction, to appear Automated Software Engineering, doi: 10.1007/s10515-024-00455-8. 

### Environment Preparation

- Python	3.10.4

```
imblearn       0. 0 

numpy          1. 22. 3

optuna         3. 2. 0

pandas         1. 4. 3

sklearn        0.  0. post1
```

### Repository Structure

- `./software-defect_caiyang` : Code directory.
  - `datasets`：The dataset of Just-In-Time Software Defedct Prediction(JIT-SDP).
  
  - `main.py`： The main code to run all experiments 
  
  - `experiment.py`：The Python code for comparative experiments of constructing LApredict model using different sampling techniques  (time period( 2 months or 6 months ) and LApredict parameters setting ( optimized or not ) ). 
  
  - `experiment-metrics.py`： The Python code for comparative experiments of constructing Logistic Regression models on different metrics. 
  
  - `experiment-classifiers.py`： The Python code for comparative experiments of constructing different models(Naive Bayes(NB), Random Forest(RF)s using different sampling techniques.
  
  - `rankMeasure_c.py`: The Python code for calculating effort-aware performance measures.
  
  - `tunePara.py`: The Python code for parameter tuning of a classifier.
  
  - `output`:  The experimental result.
    - `output-metrics`:  The results of constructing Logistic Regression models on different metrics.
    - `output-2months-default`:  The results of constructing LApredict classifier with default parameters, and the time interval is 2 months when using different sampling techniques.
    - `output-2months-optimized`:  The results of constructing LApredict  classifier with optimized parameters, and the time interval is 2 months when using different sampling techniques.
    - `output-6months-default`:  The results of constructing LApredict  classifier with default parameters, and the time interval is 6 months when using different sampling techniques.
    - `output-classifiers`
           - `NB`:  The results of constructing Naive Bayes classifier (NB) with default parameters, and the time interval is 2 months when using different sampling techniques.
            - `RF`:  The results of constructing Random Forest classifier (RF) with default parameters, and the time interval is 2 months when using different sampling techniques.
    
       

### How to run

- Modify the line in the file `./software-defect_caiyang/experiment.py` , and  `./software-defect_caiyang/experiment_metrics.py` , the line are as follows:

  ```R
  # Specify the DIRECTORY path  of dataset
  file = "?" + fname
  ```
  
- Run the commands in the terminal.
  
  ```cmd
  $cd your code path
  $python main.py
  ```
