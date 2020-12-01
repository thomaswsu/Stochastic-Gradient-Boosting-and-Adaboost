import numpy as np
from sklearn.tree import DecisionTreeClassifier
from copy import deepcopy
from time import time
import matplotlib.pyplot as plt

class Boost:
"""
:param n_estimators: number of Weak Learners to be predicted over. Default=50
:type n_estimators: int > 0
:param learning_rate: Weight applied to gradients. Default=1
:type learning_rate: float
:param random_state: Random seed for replicating results. Default=None
:type random_state: int
:param base_learner: The Weak Learner type to be used and duplicated. Default=DecisionTreeClassifier(max_depth=2)
:type base_learner: sklearn Classifier object
"""

    def __init__(self, n_estimators=50, learning_rate=1, random_state=None, base_learner=DecisionTreeClassifier(max_depth=2)):
        self.n_estimators_ = n_estimators
        self.learning_rate_ = learning_rate
        self.random_state_ = random_state
        self.estimators_ = []
        self.alphas_ = np.zeros(self.n_estimators_)
        self.accuracies = None
        self.losses = None
        self.base_learner_ = base_learner

    def __plot(self):
        """
        Plot the accuracies and losses from an update_fit call
        :return: None
        """

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


    def fit(self, X, Y, verbose=False, batch_size=None):
        """
        :param X: X_data to be trained over
        :type X: np.array[np.array[float]]
        :param Y: Y_data to evaluate predictions
        :type Y: np.array[int]
        :param verbose: Controls how often updates are printed. Default=False
        :type verbose: int such that verbose > 0
        :param batch_size: Controls the size of minibatches. Default=None
        :type batch_size: int such that len(X) > batch_size > 0
        :raises: ValueError if invalid verbose value or batch_size is passed
        :return: updated model (self)
        :rtype: class<Boost>
        """

        if verbose and verbose < 1 or type(verbose) is not int:
            raise ValueError(f"ERROR: verbose value {verbose} is invalid")
        if batch_size and batch_size > len(X) or batch_size < 1 or type(batch_size) is not int:
            raise ValueError(f"ERROR: batch_size value {batch_size} is invalid")

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
            local_weights, estimator_alpha, estimator_error = self.boost(X, Y, local_weights, indices)
            
            if estimator_error is None:
                if verbose:
                    print(f"WARNING: No estimator error after {tracked_items}/{len(X)} data points")
                local_weights = np.ones(self.n_samples_) / self.n_samples_
                continue
            
            self.alphas_[i] = estimator_alpha
            
            if verbose and (i+1) % verbose == 0:
                print(f"Iteration {i+1} ended after {time()-start:.2f} seconds")
        
        return self

    def update_fit(self, X, Y, verbose=False, batch_size=None):
        """
        Similar behavior to Boost().fit(), but tracks and plots losses and accuracies with every iteration
        :param X: X_data to be trained over
        :type X: np.array[np.array[float]]
        :param Y: Y_data to evaluate predictions
        :type Y: np.array[int]
        :param verbose: Controls how often updates are printed. Default=False
        :type verbose: int such that verbose > 0
        :param batch_size: Controls the size of minibatches. Default=None
        :type batch_size: int such that len(X) > batch_size > 0
        :raises: ValueError if invalid verbose value or batch_size is passed
        :return: updated model (self)
        :rtype: class<Boost>
        """

        if verbose and verbose < 1 or type(verbose) is not int:
            raise ValueError(f"ERROR: verbose value {verbose} is invalid")
        if batch_size and batch_size > len(X) or batch_size < 1 or type(batch_size) is not int:
            raise ValueError(f"ERROR: batch_size value {batch_size} is invalid")    

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
                indices = np.arange(len(X))
                np.random.shuffle(indices)
            indices = indices[:batch_size]
            if i == 0:
                local_weights = np.ones(self.n_samples_) / self.n_samples_
            local_weights, estimator_alpha, estimator_error = self.boost(X, Y, local_weights, indices)
            
            if estimator_error is None:
                if verbose:
                    print(f"WARNING: No estimator error after {tracked_items}/{len(X)} data points")
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

        self.__plot()
        return self
    
    def boost(self, X, Y, weights, indices):
        """
        Boosts on given data
        :param X: X_data to be trained over
        :type X: np.array[np.array[float]]
        :param Y: Y_data to evaluate predictions
        :type Y: np.array[np.array[float]]
        :param weights: Weights used to select data points
        :type weights: List[float]
        :param indices: Indices from minibatch set
        :type indices: np.array[int]
        :return: new data weights, alpha for the new estimator, error for the new estimator
        :rytpe: np.array[float], float, float
        """

        estimator = deepcopy(self.base_learner_)
        if self.random_state_:
            estimator.set_params(random_state=1)

        estimator.fit(X[indices], Y[indices], sample_weight=weights[indices])

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
        """
        Internal prediction process
        :param X: X data to predict over
        :type X: 2-dimensional np.array[np.array[float]]
        :param p: Estimator to predict
        :type p: class<self.base_learner_> object
        :returns: Prediction from the estimator
        :rtype: np.array[np.array[float]]
        """
        
        return p.predict(np.array(X))

    def update_prediction(self, p, X, t, a):
        """
        Optimized prediction for Boost().update_fit
        :param p: Previous prdiction
        :type p: np.array[np.array[float]]
        :param X: X data to predict over
        :type X: np.array[np.array[float]]
        :param t: new estimator to predict
        :type t: class<self.base_learner_> object
        :param a: Alpha for the new estimator
        :type a: float
        :returns: New prediction
        :rtype: np.array[np.array[float]]
        """

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
        """
        Collapse predictions to single class
        :param p: Predictions
        :type p: np.array[np.array[float]]
        :returns: Class number predictions
        :rtype: np.array[int]
        """

        return self.classes_.take(np.argmax(p, axis=1), axis=0)

    def predict(self, X):
        """
        Run overall prediction
        :param X: X data to predict over
        :type X: np.array[np.array[float]]
        :returns: class predictions
        :rtype: np.array[int]
        """

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
        """
        :param Y: Known classifications
        :type Y: np.array[int]
        :param f: Predictions from previous forward pass
        :type f: np.array[np.array[float]]
        :param K: Number of classes in the data set. Default=False
        :type K: int
        :returns: Loss calculation for the given data set
        :rtype: float
        """

        if not K:
            K = self.n_classes_
        y = np.ones(shape=f.shape) * (-1/(K-1))
        for i in range(f.shape[0]):
            if np.argmax(f[i]) == Y[i]:
                y[i][Y[i]] = 1

        return np.sum([np.exp((-1/K) * (y[:i].T @ f[:i])) for i in range(f.shape[1])])

    def accuracy(self, y_hat, y_true):
        """
        :param y_hat: Predicted classes
        :type y_hat: np.array[int]
        :param y_true: Known classes
        :type y_true: np.array[int]
        :returns: Accuracy of predictions
        :rtype: float
        """

        ones = np.where(y_hat == y_true, 1, 0)
        solid = np.ones((1,len(y_hat)))
        return ((ones @ ones.T) / (solid @ solid.T))[0][0]


class GradientDescent:
    def __init__(self):
        pass

    def sgdmethod(X, y, regparam, w1, T, a, m):
        """
        :param X: X data to train over
        :type X: np.array[np.array[float]] or np.array[float]
        :param y: Y data to compare to
        :type y: np.array[int]
        :param regparam: Regularization parameter lambda
        :type regparam: float
        :param w1: Initial weights and biases
        :type w1: list[list[float], list[float]]
        :param T: Number of iterations to run
        :type T: int
        :param a: Weight of alpha modifier
        :type a: float
        :param m: Regularizer for gradients
        :type m: float
        :returns: Weights of iterative models
        :rtype: List[List[float],List[float]]
        """

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
        """
        :param x: X data
        :type X: np.array[float]
        :param y: Y data
        :type y: np.array[int]
        :param w: weights and biases
        :type w: List[List[float], List[float]]
        :returns: Subgradient loss
        :rtype: float
        """

        v = [0,0]
        b = w[1]
        w = w[0]
        if 1-y*(np.dot(w,x)+b)>0:
            v[0] = -y
            v[1] = -y*x
        return v

    def subgradient_regularizer(w):
        """
        :param w: Weights
        :type w: List[List[float],List[float]]
        :returns: Subgradient for the regularizer
        :rtype: List[float]
        """

        return w[0]

    def risk(X,y,w,lam):
        """
        :param X: X data
        :type X: List[float]
        :param y: Y data
        :type y: List[int]
        :param w: Weights
        :type w: List[List[float],List[float]]
        :param lam: Lambda regularizer
        :type lam: float
        :returns: Model risk
        :rtype: float
        """

        b = w[1]
        w = w[0]
        hingesum = 0
        for i in range(X.shape[0]):
            hingesum += max(0, 1-y[i]*(np.dot(X[i],w) + b))
        
        regsum = sum(val ** 2 for val in w)
        return (1/X.shape[0])*hingesum + (lam/2)*regsum

    def loss(X,y,weight,lam):
        """
        :param X: X data
        :type X: List[float]
        :param y: Y data
        :type y: List[int]
        :param weight: Weights to be applied
        :type weight: List[List[float],List[float]]
        :param lam: Lambda regularizer
        :type lam: float
        :returns: Model loss
        :rtype: float
        """

        summand = 0
        for i in range(X.shape[0]):
            summand += max(1 - y[i] * (np.dot(weight[0],X[i]) + weight[1]),0)
        summand /= X.shape[0]
        return summand + lam/2 * sum(val ** 2 for val in weight[0]) 

    def accuracy(X,y,weight):
        """
        :param X: X data
        :type X: List[float]
        :param y: Y data
        :type y: List[int]
        :param weight: Weights to be applied
        :type weight: List[List[float],List[float]]
        :returns: Model accuracy
        :rtype: float
        """

        total = 0
        for i in range(X.shape[0]):
            if (y[i]==0 and np.dot(weight,X[i])==0) or (y[i]*np.dot(weight,X[i]) > 0):
                total += 1
        return total/X.shape[0]
      
def ShallowTree(d = 2):
    """
    :param d: Depth of trees. Default=2
    :type d: int
    :returns: sklearn DecisionTreeClassifier with given depth
    :rtype: class<sklearn.tree.DecisionTreeClassifier> object
    """

    return DecisionTreeClassifier(max_depth=d)
