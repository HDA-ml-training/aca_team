
# coding: utf-8

# In[4]:

import numpy as np
from math import exp
from math import log

def gradient_descent(X, Y, epsilon=1e-6, l=1, step_size=1e-4, max_steps=1000):
    """
    Implement gradient descent using full value of the gradient.
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :param epsilon: approximation strength
    :param max_steps: maximum number of iterations before algorithm will
        terminate.
    :return: value of beta (1 dimensional np.array)
    """
    x_mean = np.mean(X,axis = 0)
    x_std = np.std(X,axis = 0)
    lam = np.zeros(X.shape[1])

    for i in range(1,X.shape[1]):
        X[:,i] = (X[:,i] - x_mean[i])/x_std[i]
        lam[i] = l/(x_std[i]**2)
    
    beta = np.zeros(X.shape[1])
    for s in range(max_steps):
        #if s % 1000 == 0:
        #    print(s, beta)
        beta_new = beta - step_size * normalized_gradient(X, Y, beta, lam)
        change = np.linalg.norm(beta_new - beta)/np.linalg.norm(beta_new)
        beta = beta_new
        if change < epsilon:
            break
    m = 0
    for i in range(1, len(beta)):
        m +=  (x_mean[i]/x_std[i])*beta[i]
    beta[0] = beta[0] - m
    for i in range(1, len(beta)):
        beta[i] =  beta[i]/x_std[i]
        
    return beta


# In[2]:

def normalized_gradient(X, Y, beta, l):
    """
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param beta: value of beta (1 dimensional np.array)
    :param l: regularization parameter lambda
    :return: normalized gradient, i.e. gradient normalized according to data
    """
    N = Y.shape[0]
    
    grad = np.zeros(len(beta))
    for i in range(N):
        grad -= Y[i] * X[i] * (1 - sigmoid(Y[i]*(beta.dot(X[i]))))
        
    #grad /= N
    #grad += l*beta
    #for i in range(0,len(beta)):
    #    grad[i] += l* beta[i]
    return grad + l*beta


# In[5]:

def sigmoid(s):
    return  1 / ( 1 + exp(-s))


# In[6]:

def predict(beta,X):
    return  [sigmoid(x.dot(beta)) for x in X]


# In[ ]:



