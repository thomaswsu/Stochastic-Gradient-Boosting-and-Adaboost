import numpy as np
from sklearn.tree import DecisionTreeClassifier

class WeakLearner:
    def __init__(self, model, i):
        self.__class = i
        self.__model = model
        self.miss_data = None
        self.error_rate = None
    
    def __sign(self, val):
        return 1 if val > 0 else -1
    
    def name(self):
        return self.__name
    
    def model(self):
        return self.__model
    
    def miss_classify(self, data, eval_data):
        self.miss_data = []
        y_pred = self.__model.predict(data)
        for i in range(len(y_pred)):
            if self.__sign(y_pred[i]) != self.__sign(eval_data[i]):
                self.miss_data.append(i)
        
    def calc_error_rate(self, w):
        self.error_rate = np.sum(w[self.miss_data])
    
    def calc_voting_power(self):
        self.__alpha = 1/2*np.log((1-self.error_rate)/self.error_rate)
        
def ShallowTree(d = 2):
    return DecisionTreeClassifier(max_depth=d)

def classify(data, classification):
    return [1 if np.where(d == 1)[0][0] == classification else -1 for d in data]

# Voting: returns +1 or -1
def eval_H(d,H):
    return np.sign(np.sum([h.alpha_*h.model().predict(d) for h in H]))

def H_accuracy(H,data):
    tot = len(data)
    c = 0
    for d in data:
        c += eval_H(d,H)
    return c/tot
