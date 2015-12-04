# LinearClassifiersRace
============

In this project for the Machine Learning course I have implemented
a python code that generate a random binary classification problem
and then compare on it three linear classifiers methods using K-Fold cross validation.

### Support Vector Machine classifier:

* For this method I used another K-Fold validation for each iteration in order to estimate the best penalty coefficient in for the current sample

### Random Forest classifier:

* For this method I used another K-Fold validation for each iteration in order to estimate the best number of trees for the current sample

### Naive Bayes classifier:

* This method is simple and intuitive and don't need parameter estimation

### Evaluation:

* For the evaluation I calculated accuracy, F1 value and AucRoc value for each fold and for each method.
* Then I have compare the results of each round and of the average. 

### Results

* For the complete results read the report.pdf