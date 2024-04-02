import numpy as np
import pandas as pd
import random
from models.itree import IsolationTree

import sys
sys.path.append('..')
from util.utils import c

class IsolationForest:
    """
    An ensemble of Isolation Trees, each of which is built using different
    sub samples of the data. The Isolation Forest calculates the anomaly
    score for each observation in the data. The anomaly score is used to
    determine if an observation is an anomaly. If the score is greater than
    some threshold, the observation is an anomaly.
    """
    
    def __init__(self, sample_size, n_trees=10):
        """
        Set the sub sampling size and the number of trees in the forest.
        """
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.height_limit = np.log2(sample_size)
        self.trees = []

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees. 
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.trees = []

        for _ in range(self.n_trees):
            sample_idx = random.sample(range(len(X)), self.sample_size)
            temp_tree = IsolationTree(self.height_limit, 0).fit(X[sample_idx, :], improved)
            self.trees.append(temp_tree)

        return self

    def path_length(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X. Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i. Return an
        ndarray of shape (len(X), 1).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        pl_vector = []
        for x in (X):
            pl = np.array([t.path_length(x, 0) for t in self.trees])
            pl = pl.mean()

            pl_vector.append(pl)

        pl_vector = np.array(pl_vector).reshape(-1, 1)

        return pl_vector

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score for
        each x_i observation, returning an ndarray of shape (len(X), 1).
        """
        return 2.0 ** (-1.0 * self.path_length(X) / c(len(X)))

    def predict_from_anomaly_scores(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        predictions = [1 if p[0] >= threshold else 0 for p in scores]
        
        return predictions

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        A shorthand for calling anomaly_score() and predict_from_anomaly_scores().
        """
        scores = 2.0 ** (-1.0 * self.path_length(X) / c(len(X)))
        predictions = [1 if p[0] >= threshold else 0 for p in scores]

        return predictions