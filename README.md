# Stochastic-Gradient-Boosting-and-Adaboost

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
