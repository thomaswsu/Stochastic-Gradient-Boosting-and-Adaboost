import numpy as np
from sklearn.tree import DecisionTreeClassifier
import csv

def data_formating(losses, accuracies):
    f_data = []
    
    for i in range(len(losses[0])):
        row = []
        for j in range(len(losses)):
            row.append(losses[j][i])
            row.append(accuracies[j][i])
        f_data.append(row)
    
    return f_data

def csv_formater(file_name, lower, higher, losses, accuracies):
    file_name = file_name + ".csv"
    with open(file_name, 'w', newline='') as csvfile:
        fieldnames = []
        steps = [i/10 for i in range(lower, higher)]
        for i in range(len(steps)):
            fieldnames.append("Loss: {}".format(steps[i]))
            fieldnames.append("Training Accuracy: {}".format(steps[i]))

        writer = csv.writer(csvfile) 

        writer.writerow(fieldnames) 

        data = data_formating(losses, accuracies)
        writer.writerows(data)

def csv_reader(file_name):
    with open(file_name) as File:
        reader = csv.reader(File, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
        rows = []
        for row in reader:
            rows.append(row)
    
    num_row = len(rows)
    num_col = len(rows[0])

    data = []
    for i in range(num_col):
        column = []
        for j in range(1, num_row):
            column.append(float(rows[j][i]))
        data.append(column)
        
    return data
    
from copy import deepcopy
from time import time

class Boost:
    def __init__(self, n_estimators=50, learning_rate=1, random_state=None, base_learner=DecisionTreeClassifier(max_depth=2)):
        self.n_estimators_ = n_estimators
        self.learning_rate_ = learning_rate
        self.random_state_ = random_state
        self.estimators_ = []
        self.alphas_ = np.zeros(self.n_estimators_)
        self.accuracies = None
        self.base_learner_ = base_learner
    
    def plot(self):
        if not self.accuracies:
            raise ValueError("ERROR: self.accuracies not defined. Make sure self.fit() is called with visible=True")
        plt.plot([i+1 for i in range(self.n_estimators_)],self.accuracies)
        plt.xlabel("Number of estimators")
        plt.ylabel("Accuracy")
        plt.show()

    def fit(self, X, Y, verbose=False, visible=False, verbose_visibility=False):
        if visible:
            self.accuracies = []
            
        start = None
        if verbose:
            start = time()

        self.n_samples_ = X.shape[0]
        if type(Y) is list:
            self.classes_ = np.array(sorted(list(set(Y))))
        else:
            Y = np.ravel(Y)
            self.classes_ = np.array(sorted(set(Y.tolist())))
        self.n_classes_ = len(self.classes_)
        for i in range(self.n_estimators_):
            if i == 0:
                if verbose:
                    print(f"Iteration starting after {time()-start:.2f} seconds")
                local_alphas = np.ones(self.n_samples_) / self.n_samples_
            local_weights, estimator_alpha, estimator_error = self.boost(X, Y, local_alphas, visible)
            
            if estimator_error is None:
                break
            
            self.alphas_[i] = estimator_alpha
            
            if verbose and (i+1) % verbose == 0:
                print(f"Iteration {i+1} ended after {time()-start:.2f} seconds")
            
            if visible:
                acc_start = time()
                self.accuracies.append(self.accuracy(self.predict(X),Y))
                if verbose_visibility and (i+1) % verbose_visibility == 0:
                    print(f"Accuracy {i+1} ({self.accuracies[-1]:.2%}) took {time()-acc_start:.2f} seconds")
                
            if estimator_error <= 0:
                break
        
        if visible:
            self.plot()
        
        return self
    
    def boost(self, X, Y, weights, visible):
        estimator = deepcopy(self.base_learner_)
        if self.random_state_:
            estimator.set_params(random_state=1)

        estimator.fit(X, Y, sample_weight=weights)

        y_hat = estimator.predict(X)
        incorrect = y_hat != Y
        estimator_error = np.dot(incorrect, weights) / np.sum(weights, axis=0)

        if estimator_error >= 1 - 1 / self.n_classes_:
            return None, None, None

        estimator_weight = self.learning_rate_ * np.log((1 - estimator_error) / estimator_error) + np.log(self.n_classes_ - 1)

        if estimator_weight <= 0:
            return None, None, None

        weights *= np.exp(estimator_weight * incorrect)

        w_sum = np.sum(weights, axis=0)
        if w_sum <= 0:
            return None, None, None

        weights /= w_sum

        self.estimators_.append(estimator)

        return np.array(weights), estimator_weight, estimator_error

    def __predict(self, X, p):
        return p.predict(np.array(X))

    def predict(self, X):
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        trial = time()
        predictions = np.apply_along_axis(self.__predict, np.array(self.estimators_)[:, np.newaxis], 0, X=X)
        pred = sum((p == classes).T * a for p in e for e, a in zip(predicionts, self.alphas_))
        print(f"Optimization took {time()-trial:.2f} seconds")
        trial = time()
        pred = sum((estimator.predict(X) == classes).T * a for estimator, a in zip(self.estimators_, self.alphas_))
        print(f"Traditional took {time()-trial:.2f} seconds")
        pred /= self.alphas_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            pred = pred.sum(axis=1)
            return self.classes_.take(pred > 0, axis=0)
        
        return self.classes_.take(np.argmax(pred, axis=1), axis=0)
    
    def accuracy(self, y_hat, y_true):
        ones = np.where(y_hat == y_true, 1, 0)
        solid = np.ones((1,len(y_hat)))
        return ((ones @ ones.T) / (solid @ solid.T))[0][0]


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
