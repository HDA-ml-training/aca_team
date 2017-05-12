
# coding: utf-8

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class DecisionNode(object):
    """
    README
    DecisionNode is a building block for Decision Trees.
    DecisionNode is a python class representing a  node in our decision tree
    node = DecisionNode()  is a simple usecase for the class
    you can also initialize the class like this:
    node = DecisionNode(column = 3, value = "Car")
    In python, when you initialize a class like this, its __init__ method is called 
    with the given arguments. __init__() creates a new object of the class type, and initializes its 
    instance attributes/variables.
    In python the first argument of any method in a class is 'self'
    Self points to the object which it is called from and corresponds to 'this' from Java

    """

    def __init__(self,
                 column=None,
                 value=None,
                 false_branch=None,
                 true_branch=None,
                 current_results=None,
                 is_leaf=False,
                 results=None):
        self.column = column
        self.value = value
        self.false_branch = false_branch
        self.true_branch = true_branch
        self.current_results = current_results
        self.is_leaf = is_leaf
        self.results = results


# In[10]:

class DecisionTree(object):
    """
    DecisionTree class, that represents one Decision Tree

    :param max_tree_depth: maximum depth for this tree.
    """
    def __init__(self, max_tree_depth):
        self.max_depth = max_tree_depth

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        data = np.column_stack((X,Y))
        self.tree = build_tree(data)

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: Y - 1 dimension python list with labels
        """
        Y = []
        for i in range(X.shape[0]):
            Y.append(predict_for_one_feature(self.tree,X[i]))
            
        return Y
       
def build_tree(data, current_depth=0, max_depth=1e10):
    """
    build_tree is a recursive function.
    What it does in the general case is:
    1: find the best feature and value of the feature to divide the data into
    two parts
    2: divide data into two parts with best feature, say data1 and data2
        recursively call build_tree on data1 and data2. this should give as two 
        trees say t1 and t2. Then the resulting tree should be 
        DecisionNode(...... true_branch=t1, false_branch=t2) 


    In case all the points in the data have same Y we should not split any more, and return that node
    For this function we will give you some of the code so its not too hard for you ;)
    
    param data: param data: A 2D python list
    param current_depth: an integer. This is used if we want to limit the numbr of layers in the
        tree
    param max_depth: an integer - the maximal depth of the representing
    return: an object of class DecisionNode

    """
    if len(data) == 0:
        return DecisionNode(is_leaf=True)

    if(current_depth == max_depth):
        return DecisionNode(current_results=dict_of_values(data))

    if(len(dict_of_values(data)) == 1):
        return DecisionNode(current_results=dict_of_values(data), is_leaf=True)

    #This calculates gini number for the data before dividing 
    self_gini = gini_impurity(data, [])
    
    
    best_gini = 1e10
    best_column = None
    best_value = None
    #best_split is tuple (data1,data2) which shows the two datas for the best divison so far
    best_split = None
    
    #Below are the attributes of the best division that you need to find. 
    #You need to update these when you find a division which is better
    for i in range(len(data[0]) - 1):
        results = defaultdict(bool)
        for row in data:
            r = row[i]
            results[r] = True
        for x in results.keys():
            data1,data2 = divide_data(data, i, x)
            if data1 == [] or data2 == []:
                continue
            gini_current = gini_impurity(data1, data2)
            if best_gini > gini_current:
                best_gini = gini_current
                best_column = i
                best_value = x
                best_split = data1, data2
                
    #You need to find the best feature to divide the data
    #For each feature and each possible value of the feature compute the 
    # gini number for that division. You need to find the feature that minimizes
    # gini number. Remember that last column of data is Y
    # Think how you can use the divide_data and gini_impurity functions you wrote 
    # above
    

    #if best_gini is no improvement from self_gini, we stop and return a node.
    if abs(self_gini - best_gini) < 1e-10:
        return DecisionNode(current_results=dict_of_values(data), is_leaf=True)
    else:
        #recursively call build tree, construct the correct return argument and return
        return DecisionNode(value = best_value,current_results = dict_of_values(data),column = best_column,
                            is_leaf = False, true_branch = build_tree(best_split[0],current_depth + 1),
                            false_branch = build_tree(best_split[1],current_depth + 1))
def gini_impurity(data1, data2):

    """
    Given two 2D lists of compute their gini_impurity index. 
    Remember that last column of the data lists is the Y
    Lets assume y1 is y of data1 and y2 is y of data2.
    gini_impurity shows how diverse the values in y1 and y2 are.
    gini impurity is given by 

    N1*sum(p_k1 * (1-p_k1)) + N2*sum(p_k2 * (1-p_k2))

    where N1 is number of points in data1
    p_k1 is fraction of points that have y value of k in data1
    same for N2 and p_k2
    

    param data1: A 2D python list
    param data2: A 2D python list
    return: a number - gini_impurity 
    """
    #TODO
    N1 = len(data1)
    res = 0
    dict1 = dict_of_values(data1)
    for x in dict1.values():
        res += N1* x / sum(dict1.values()) * (1 - x / sum(dict1.values())) 
    N2 = len(data2)
    dict2 = dict_of_values(data2)
    for x in dict2.values():
        res += N2* x / sum(dict2.values()) * (1 - x / sum(dict2.values())) 
    
    return res

def divide_data(data, feature_column, feature_val):
    """
    this function dakes the data and divides it in two parts by a line. A line
    is defined by the feature we are considering (feature_column) and the target 
    value. The function returns a tuple (data1, data2) which are the desired parts of the data.
    For int or float types of the value, data1 have all the data with values >= feature_val
    in the corresponding column and data2 should have rest.
    For string types, data1 should have all data with values == feature val and data2 should 
    have the rest.

    param data: a 2D Python list representing the data. Last column of data is Y.
    param feature_column: an integer index of the feature/column.
    param feature_val: can be int, float, or string
    return: a tuple of two 2D python lists
    """
    #TODO
    data1 = []
    data2 = []
    for x in data:
        if x[feature_column] >= feature_val:
            data1.append(x)
        else:
            data2.append(x)

    return data1, data2

def predict_for_one_feature(tree,x):
    y = -1
    ind = -1
    if(tree.is_leaf):
        for t in tree.current_results:
            if(tree.current_results[t] > y):
                y = tree.current_results[t]
                ind = t
        return ind
    else:
        if(x[tree.column] >= tree.value):
            return predict_for_one_feature(tree.true_branch,x)
        else:
            return predict_for_one_feature(tree.false_branch,x)
    
def dict_of_values(data):
    """
    param data: a 2D Python list representing the data. Last column of data is Y.
    return: returns a python dictionary showing how many times each value appears in Y

    for example 
    data = [[1,'yes'],[1,'no'],[1,'yes'],[1,'yes']]
    dict_of_values(data)
    should return {'yes' : 3, 'no' :1}
    """
    results = defaultdict(int)
    for row in data:
        r = row[len(row) - 1]
        results[r] += 1
    return dict(results)

def print_tree(tree, indent=' '):
    # Is this a leaf node?
    if tree.is_leaf:
        print(str(tree.current_results))
    else:
        # Print the criteria
        #         print (indent+'Current Results: ' + str(tree.current_results))
        print('Column ' + str(tree.column) + ' : ' + str(tree.value) + '? ')

        # Print the branches
        print(indent + 'True->', end="")
        print_tree(tree.true_branch, indent + '  ')
        print(indent + 'False->', end="")
        print_tree(tree.false_branch, indent + '  ')






