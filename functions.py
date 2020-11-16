import numpy as np
from sklearn.tree import DecisionTreeClassifier

class WeakLearner:
    def __init__(self, model, i):
        self.__class = i
        self.__model = model
        self.miss_data = None
        self.error_rate = None
    
    def sign(self, val):
        return 1 if val > 0 else -1
    
    def name(self):
        return self.__name
    
    def model(self):
        return self.__model
    
    def miss_classify(self, data, eval_data):
        self.miss_data = []
        y_pred = self.__model.predict(data)
        self.miss_data.extend(np.where(y_pred != eval_data)[0].tolist())
        # for i in range(len(y_pred)):
        #     if self.sign(y_pred[i]) != self.sign(eval_data[i]):
        #         self.miss_data.append(i)
        
    def calc_error_rate(self, w):
        self.error_rate = np.sum(w[self.miss_data])
    
    def calc_voting_power(self):
        self.alpha_ = 1/2*np.log((1-self.error_rate)/self.error_rate)
        
def ShallowTree(d = 2):
    return DecisionTreeClassifier(max_depth=d)

def classify(data, classification):
    return [1 if np.where(d == 1)[0][0] == classification else -1 for d in data]

# Voting: returns +1 or -1
def eval_P(P, eval_set, h):
    return [1 if h.sign(P[i]) == h.sign(eval_set[i]) else -1 for i in range(len(P))]

def H_accuracy(H, data, eval_set):
    c = []
    for h in H:
        results = eval_P(h.model().predict(data), eval_set, h)
        if not c:
            c.extend(results)
        else:
            for i in range(len(results)):
                c[i] += results[i]
    for i in range(len(c)):
        c[i] = c[i] / len(H)
    return sum(c)/len(c)
