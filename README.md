# Computational-Data-Analysis

## Introduction/Data description

This project was done as a part of Computational Data Analysis course at DTU. At this course a dataset of unknown 
origin was given and the goals were to create a predictive model. The project was part of an in-class competition to 
achieve the smallest:

* root mean squared error(RMSE) on 1000 new observations
* difference between estimated root mean squared error and actual RMSE.
  
The data consist of 100 observations and Y responses with 100 features and 1000 additional X_new observations for prediction. 
Out of 100 features, 5 of them are categorical data. The rest is continuous and already normalized data.

## Model and method

The issue with this dataset is relatively small set of observation with responses compared to feature dimensions. 
To address this problem, appropriate preprocessing techniques and models were chosen to deal with high-dimensional data.  
Firstly, elastic net was picked to model the data, because as for a linear model it can handle well high dimensional data by 
combining L1 and L2 regularization penalties. Next, random forests were tried as their ensemble models - CARTs - can 
select optimal features for making a best split of bootstrapped data. By averaging predictions of each CART, 
random forest achieves reduced variance, while having the same bias as individual trees. Lastly, SVM with sigmoid 
kernel was used to try out a very flexible and non-linear regressor that can have potentially more capacity to model 
complex data.

## Missing values

In the X dataset, there were 58 missing values out of 10000 values. A single observation had at most 3 missing values. 
A single feature had also at most 3 missing values. Because of a small amount of missing values, less focus was 
given on them and they were replaced by a mean value of a column that they belong to.

## Factor handling
Last 5 features(X95 to X100) are categorical variables. In order to include them in all models, one-hot encoding and 
normalization of categorical data was done, extending number of features from 100 to 116. During model selection, 
models were evaluated on three datasets: original dataset, dataset with removed features that had high feature correlation 
and low output correlation and dataset whose dimensionality was reduced by PCA. The interest here was to analyze whether highly 
preprocessed datasets ease learning and improve results of models. Second dataset removed features that had above 0.9 feature 
correlation and below 0.02 output correlation. This approach removed 15 features. The third dataset was projected using PCA 
into 62 dimensions, which explained 0.95% of data's variance.

## Model selection

Dataset was randomly divided into 70/30 split between training and test data. All model selections used the same 
training data. Hyperparameters of models were tuned by grid search with 16-hold cross validation. In Fig.5.1 can be seen 
RMSE on training and validation set of the best found models for aforementioned three datasets. Those RMSE were calculated 
as mean RMSE of all folds in the cross validation. Models and hyperparameters:

Elastic net fits linear regression model with parameters: 

<p align="center">
  <img src="https://raw.githubusercontent.com/mikpat/Computational-Data-Analysis/master/eq.1.PNG">
</p>


* α - constant that multiplies the penalty terms. Linearly spread grid search was done with 20 values ranging from 0.1 to 1.5. For all models the best α=0.311
* l1_ratio- describes whether L1 or L2 penalty dominates regularization term. Linearly spread grid search was done with 10 values ranging from 0.01 to 0.99. For dataset #1 and #2 l1_ratio=0.99, which results in L1 penalty domination and shows that most features had small predictive power(only 21 parameters β were non-zero). For the dataset #3, l1_ratio=0.01 and  L2 penalty dominated regularization. Dataset #3 had heavily reduced feature dimensionality and therefore the best model didn’t need sparsity coming from L1 regularization.

Random forest hyperparameters:

* max_depth – regulates maximum depth of all the tree in a random forest. Values considered in grid search {10, 30, 50}. Best models: on dataset #1 max_depth=30, on dataset #2 and #3 max_depth=50. 
* n_estimators – number of trees in a random forest. Values considered in grid search {200, 400, 600}. On all datasets n_estimators=600 was the best hyperparameter

Support vector machine:

* C – penalty parameter of the error term. Linearly spread grid search was done with 5 values ranging from 2 to 10. Best models: on dataset #1 C=8.0, on dataset #2 C=6.0, 
on dataset #3 C=2.0. 

|Dataset          | Training Elastic Net      | CV Elastic Net   | Training Random Forest|CV Random Forest|Training SVM|CV SVM|
|-----------------|:--------------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:| 
|Dataset source #1| 1,866        | 3,041    |1,350|	3,528	|2,875	|3,690|
|Dataset #2       | 1,867       | 3,043|1,326	|3,506	|2,805	|3,543|
|Dataset #3 PCA   | 1,182          | 3,924|1,980	|5,202	|3,785	|4,160|


Generally, models performed worse with more data dimensionality reduction techniques. 
As can be seen from grid search, models in this case adjusted hyperparameters and performed better 
with all the information in the dataset. For the final model elastic net trained on the whole 
dataset #1 was chosen as it had the lowest mean cross-validation RMSE.

## Model validation
In order to validate model, it’s performance was evaluated on unseen test dataset, which consists of 30% of source dataset.
The chosen elastic net from model selection achieved (RMSE) = 2,938. Rest of the values are only for comparison, since model selection is based on cross-validation sets. One insight is that most of the (RMSE) from test sets are lower than cross-validation RMSE, which can suggest that test set was easier than mean evaluation in k-folds and is not representative of the whole distribution. However, choosing larger test set could lead to worse model selection as less data would be used in training.

|Dataset          | Test Elastic Net      | Test Random Forest   | Test SVM|
|-----------------|:--------------------:|:-------------:|:-------------:|
|Dataset source #1| 2,938        | 3,526    |3,450|	
|Dataset #2       | 2,944      | 3,522|3,551	|
|Dataset #3 PCA   | 3,509         | 4,217|3,695	|


## Results
To solve regression problem elastic nets, random forests and support vector machines were trained and validated 
on the original dataset and two augmented datasets. The model with lowest validation RMSE was an elastic net 
with α=0.311 and l1_ratio=0.99 used with original dataset. This model is predicted to have estimated (RMSE) = 2,938.
