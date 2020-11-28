# Stochastic-Gradient-Boosting-and-Adaboost

This repository is for CSCI 4961/6961 Project Details, Fall 2020. This code is an extension of the research provided in the paper Stochastic Gradient Boosting by Jerome H. Friedman (https://statweb.stanford.edu/~jhf/ftp/stobst.pdf) 

In this repository we evaluate the performance of Stochastic Boosting, Traditional Boosting, Gradient Descent, and Stochastic Gradient Descent methods in two ways. The first is through evaluting the amount of data needed for each method to effectively generalize the classification problem. The second is effect of increasing the complexity of Weak Learner. How does a Weak Learner perform as it becomes for complex. Is it still able to generalize the classifcation problem in the same number of epochs? 

## Classes

### WeakLearner

- Parameters:
  - model: A model to run predictions on (for us, either DoubleTree or SingleNN)
  - i: integer representing which iteration of boosting we are on
- miss_classify:
  - Parameters:
    - data: X_data to be classified
    - eval_data: Y_data to be compared to
  - Returns:
    - None. Appends missclassified datapoints to private array
- calc_error_rate;
  - Parameters:
    - w: float weight for assessing errors
  - Returns:
    - None. Sets self.error_rate
- calc_voting_power:
  - Parameters:
    - None
  - Returns:
    - None. Sets self.alpha
- name:
  - Parameters:
    - None
  - Returns:
    - Safe copy of self.__name
- model:
  - Parameters:
    - None
  - Returns:
    - Safe copy of self.__model

### Classify

- Parameters:
  - data: Y_data to reclassify
  - classification: Index of the positive classifier
- Return:
  - array of +1,-1 according to classification based on positive classifier
