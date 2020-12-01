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
import matplotlib.pyplot as plt

class Boost:
    def __init__(self, n_estimators=50, learning_rate=1, random_state=None, base_learner=DecisionTreeClassifier(max_depth=2)):
        self.n_estimators_ = n_estimators
        self.learning_rate_ = learning_rate
        self.random_state_ = random_state
        self.estimators_ = []
        self.alphas_ = np.zeros(self.n_estimators_)
        self.accuracies = None
        self.losses = None
        self.base_learner_ = base_learner

    def plot(self):
        if not self.accuracies or not self.losses:
            raise ValueError("ERROR: self.accuracies or self.losses not defined. Make sure self.fit() is called with visible=True or that you called self.update_fit()")
        plt.plot([i+1 for i in range(len(self.accuracies))],self.accuracies)
        plt.xlabel("Number of estimators")
        plt.ylabel("Accuracy")
        plt.show()
        plt.plot([i+1 for i in range(len(self.losses))], self.losses)
        plt.xlabel("Number of estimators")
        plt.ylabel("Loss")
        plt.show()


    def fit(self, X, Y, verbose=False, batch_size = None):
        if not batch_size:
            batch_size = len(X)
            
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
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            indices = indices[:batch_size]
            if i == 0:
                if verbose:
                    print(f"Iteration starting after {time()-start:.2f} seconds")
                local_weights = np.ones(self.n_samples_) / self.n_samples_
            local_weights, estimator_alpha, estimator_error = self.boost(X, Y, local_weights, indices, visible)
            
            if estimator_error is None:
                break
            
            self.alphas_[i] = estimator_alpha
            
            if verbose and (i+1) % verbose == 0:
                print(f"Iteration {i+1} ended after {time()-start:.2f} seconds")
        
        return self

    def update_fit(self, X, Y, verbose=False, batch_size=None):
        if not batch_size:
            batch_size = len(X)
        start = time()
        self.accuracies = []
        self.losses = []
        predictions = None
        self.n_samples_ = X.shape[0]
        if type(Y) is list:
            self.classes_ = np.array(sorted(list(set(Y))))
        else:
            Y = np.ravel(Y)
            self.classes_ = np.array(sorted(set(Y.tolist())))
        self.n_classes_ = len(self.classes_)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        tracked_items = 0
        i = 0
        while i < self.n_estimators_:
            tracked_items += 600
            if tracked_items >= len(X):
                tracked_items = 0
                # if verbose:
                    # print("Reshuffling...")
                indices = np.arange(len(X))
                np.random.shuffle(indices)
            indices = indices[:batch_size]
            if i == 0:
                local_weights = np.ones(self.n_samples_) / self.n_samples_
            local_weights, estimator_alpha, estimator_error = self.boost(X, Y, local_weights, indices)
            
            if estimator_error is None:
                # print(f"WARNING: No estimator error after {tracked_items}/{len(X)} data points")
                # break
                local_weights = np.ones(self.n_samples_) / self.n_samples_
                continue
            
            if estimator_error <= 0:
                # print(f"WARNING: Estimator error of 0 after {tracked_items}/{len(X)} data points")
                # break
                local_weights = np.ones(self.n_samples_) / self.n_samples_
                continue

            acc_time = 0

            if predictions is None:
                acc_start = time()
                predictions = (self.estimators_[-1].predict(X) == self.classes_[:, np.newaxis]).T * estimator_alpha
                collapsed = self.reduce_prediction(predictions)
                self.accuracies.append(self.accuracy(collapsed, Y))
                self.losses.append(self.loss(Y, predictions, self.n_classes_))
                acc_time = time()-acc_start

            else:
                acc_start = time()
                predictions = self.update_prediction(predictions, X, self.estimators_[-1], estimator_alpha)
                collapsed = self.reduce_prediction(predictions)
                self.accuracies.append(self.accuracy(collapsed, Y))
                self.losses.append(self.loss(Y, predictions, self.n_classes_))
                acc_time = time()-acc_start

            self.alphas_[i] = estimator_alpha
            
            if verbose and (i+1) % verbose == 0:
                print(f"Accuracy/Loss {i+1} ({self.accuracies[-1]:.2%}/{self.losses[-1]:.3f} in {acc_time:.2f} seconds) after {time()-start:.2f} seconds")

            i += 1

        self.plot()
        return self
    
    def boost(self, X, Y, weights, indices, visible=False):
        estimator = deepcopy(self.base_learner_)
        if self.random_state_:
            estimator.set_params(random_state=1)

        estimator.fit(X[indices], Y[indices], sample_weight=weights[indices])

        y_hat = estimator.predict(X)
        incorrect = y_hat != Y
        estimator_error = np.dot(incorrect, weights) / np.sum(weights, axis=0)

        if estimator_error >= 1 - 1 / self.n_classes_:
            # print(f"WARNING: estimator_error {estimator_error} >= {1 - 1 / self.n_classes_}")
            return None, None, None

        estimator_weight = self.learning_rate_ * np.log((1 - estimator_error) / estimator_error) + np.log(self.n_classes_ - 1)

        if estimator_weight <= 0:
            # print(f"WARNING: estimator_weight {estimator_weight} <= 0")
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

    def update_prediction(self, p, X, t, a):
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = (t.predict(X) == classes).T * a
        p = (p*self.alphas_.sum())
        p += pred
        p /= (self.alphas_.sum() + a)
        if n_classes == 2:
            p[:, 0] *= -1
            p = p.sum(axis=1)
            return self.classes_.take(p > 0, axis=0)
        
        return p

    def reduce_prediction(self, p):
        return self.classes_.take(np.argmax(p, axis=1), axis=0)

    def predict(self, X):
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = sum((estimator.predict(X) == classes).T * a for estimator, a in zip(self.estimators_, self.alphas_))
        pred /= self.alphas_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            pred = pred.sum(axis=1)
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)
    
    def loss(self, Y, f, K=False):
        if not K:
            K = self.n_classes_
        y = np.ones(shape=f.shape) * (-1/(K-1))
        for i in range(f.shape[0]):
            if np.argmax(f[i]) == Y[i]:
                y[i][Y[i]] = 1

        return np.sum([np.exp((-1/K) * (y[:i].T @ f[:i])) for i in range(f.shape[1])])

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
