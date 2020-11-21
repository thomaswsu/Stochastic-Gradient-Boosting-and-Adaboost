import numpy as np
from sklearn.tree import DecisionTreeClassifier

class WeakLearner:
    def __init__(self, model, i):
        self.__class = i
        self.__model = model
        self.miss_data = None
        self.error_rate = None
        self.y_pred = []
    
    def sign(self, val):
        return 1 if val > 0 else -1
    
    def name(self):
        return self.__name
    
    def model(self):
        return self.__model
    
    def miss_classify(self, data, eval_data):
        self.miss_data = []
        self.y_pred = self.__model.predict(data)
        self.miss_data.extend(np.where(self.y_pred != eval_data)[0].tolist())
        
    def calc_error_rate(self, w):
        self.error_rate = np.sum(w[self.miss_data])
    
    def calc_voting_power(self):
        self.alpha_ = 1/2*np.log((1-self.error_rate)/self.error_rate)

class GradientDescent:
    def __init__(self):
        pass

    def sgdmethod(X, y, regparam, w1, T, a, m):
        t = 0
        X_pos = 0
        X_batch = None
        y_batch = None
        w = [(w1[0],w1[1])]
        while t < T:
            # Shuffle X,y at epoch 
            if t%(len(y)//m) == 0:
                X_pos = 0
                state = np.random.get_state()
                np.random.shuffle(X)
                np.random.set_state(state)
                np.random.shuffle(y)
            X_batch = X[X_pos:X_pos+m]
            y_batch = y[X_pos:X_pos+m]
            X_pos += m
            
            alpha = (1 + a*t)**-1
            
            # Run descent on batch
            g = [0 for x in range(X_batch.shape[1])]
            g_bias = 0
            w_t = w[t]
            for i in range(m):
                bias,loss = self.subgradient_loss(X_batch[i], y_batch[i],[w_t[0],w_t[1]])
                g = np.add(g,loss)
                g_bias += bias
            reg = regparam * self.subgradient_regularizer(w_t[0])
            g = np.true_divide(g,m)
            g = np.add(g,reg)
            g_bias /= m
            w.append((np.add(w_t[0],-alpha*g),w_t[1]-g_bias*alpha))
            
            t += 1
        return w
        
    def subgradient_loss(x, y, w):
        v = [0,0]
        b = w[1]
        w = w[0]
        if 1-y*(np.dot(w,x)+b)>0:
            v[0] = -y
            v[1] = -y*x
        return v

    def subgradient_regularizer(w):
        return w[0]

    def risk(X,y,w,lam):
        b = w[1]
        w = w[0]
        hingesum = 0
        for i in range(X.shape[0]):
            hingesum += max(0, 1-y[i]*(np.dot(X[i],w) + b))
        
        regsum = sum(val ** 2 for val in w)
        return (1/X.shape[0])*hingesum + (lam/2)*regsum

    def loss(X,y,weight,lam):
        summand = 0
        for i in range(X.shape[0]):
            summand += max(1 - y[i] * (np.dot(weight[0],X[i]) + weight[1]),0)
        summand /= X.shape[0]
        return summand + lam/2 * sum(val ** 2 for val in weight[0]) 

    def accuracy(X,y,weight):
        total = 0
        for i in range(X.shape[0]):
            if (y[i]==0 and np.dot(weight,X[i])==0) or (y[i]*np.dot(weight,X[i]) > 0):
                total += 1
        return total/X.shape[0]

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
