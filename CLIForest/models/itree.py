import numpy as np
import random
import sys
sys.path.append('..')
from util.utils import c

class IsolationTree:
    """
    Isolation Tree Algorithm from Liu et al. It is an Unsupervised Anomaly Detector
    that builds the tree by randomly selecting a feature and then randomly selecting
    a split value for the data until the data instances are isolated.
    """
    
    def __init__(self, height_limit, current_height):
        """ Initialize the tree parameters."""
        self.height_limit = height_limit
        self.current_height = current_height
        self.split_by = None
        self.split_value = None
        self.right = None
        self.left = None
        self.size = 0
        self.exnodes = 0
        self.n_nodes = 1
        

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.
        """

        self.size = X.shape[0]
        if self.size <= 1 or self.current_height >= self.height_limit:
            self.exnodes = 1
            return self
        
        # Improved split, as suggested by https://github.com/Divya-Bhargavi/isolation-forest/tree/master
        good_split = False
        while not good_split:
            self.split_by = random.choice(np.arange(X.shape[1]))
            X_col = X[:, self.split_by]
            min_x = X_col.min()
            max_x = X_col.max()
            split_value = min_x + random.betavariate(0.5, 0.5) * (max_x - min_x)
            
            if min_x == max_x:
                self.exnodes = 1
                return self
            
            good_split = not improved
            l, r = X[X_col < split_value], X[X_col >= split_value]
            if self.size < 10 or l.shape[0] < 0.25 * r.shape[0] or r.shape[0] < 0.25 * l.shape[0] or (l.shape[0] > 0 and r.shape[0] > 0):
                good_split = True
                
        self.split_value = split_value
        self.left  = IsolationTree(self.height_limit, self.current_height + 1).fit(l, improved)
        self.right = IsolationTree(self.height_limit, self.current_height + 1).fit(r, improved)
        self.n_nodes = self.left.n_nodes + self.right.n_nodes + 1

        return self
    
    
    def path_length(self, x: np.ndarray, depth=0):
        """
        Given a data instance x, compute the path length of x in the tree.  
        """
        if self.exnodes == 1:
            return depth + c(self.size)
        
        if x[self.split_by] < self.split_value:
            return self.left.path_length(x, depth + 1)
        else:
            return self.right.path_length(x, depth + 1)