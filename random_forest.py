
# coding: utf-8

# In[3]:

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import decision_tree as dt

class RandomForest(object):
    """
    RandomForest a class, that represents Random Forests.

    :param num_trees: Number of trees in the random forest
    :param max_tree_depth: maximum depth for each of the trees in the forest.
    :param ratio_per_tree: ratio of points to use to train each of
        the trees.
    """
    def __init__(self, num_trees, max_tree_depth, ratio_per_tree=0.5):
        self.num_trees = num_trees
        self.max_tree_depth = max_tree_depth
        self.trees = None
        self.ratio_per_tree = ratio_per_tree

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        self.trees = []
        for i in range(self.num_trees):
            n = X.shape[0]
            idx = np.arange(n)
            np.random.seed(13)
            np.random.shuffle(idx)
            X = X[idx]
            Y = Y[idx]
            Xtrain = X[:int(n*self.ratio_per_tree), :]
            Ytrain = Y[:int(n*self.ratio_per_tree), :]
        
            classifier = dt.DecisionTree(self.max_tree_depth)
            classifier.fit(Xtrain, Ytrain)
            self.trees.append(classifier)
        

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: (Y, conf), tuple with Y being 1 dimension python
        list with labels, and conf being 1 dimensional list with
        confidences for each of the labels.
        """
        ans = []
        Y = []
        conf = []
        for tree in self.trees:
            ans.append(tree.predict(np.array(X)))
        for i in range(len(ans)):
            for j in range(len(ans[0])):
                ones = 0
                zeros = 0
                if(ans[i][j] == 1):
                    ones += 1
                else:
                    zeros += 1
                if(ones >= zeros):
                    Y.append(1)
                    conf.append(ones/(ones+zeros))
                else:
                    Y.append(0)
                    conf.append(zeros/(ones+zeros))

        return (Y, conf)


# In[ ]:



