import numpy as np
import pandas as pd
from collections import deque
from models.itree import IsolationTree
from models.iforest import IsolationForest
import random

class IsolationForestPCB(IsolationForest):
    """
    An ensemble of Isolation Trees for streaming data, based on Michael Heigl et al.'s
    (2021) paper. Keeps a performance counter for each tree in the ensemble, and updates
    the trees based on the performance counter.
    """
    
    def __init__(self, sample_size, contamination_rate=0.1, n_trees=30, window_size=1000):
        """
        Set the sub sampling size, window and the number of trees in the forest.
        """
        super().__init__(sample_size, contamination_rate, n_trees)
        self.window_size = window_size
        self.performance_counter = np.zeros(n_trees)
        
        
    def update_forest(self, X: np.ndarray, improved=False):
        """
        Update the forest, based on the performance counter of each tree, 
        discarding the ones with negative performance counter.
        Returns the predictions, scores for the current window.
        """
        # Retrain the trees with performance counter <= 0
        for i in range(self.n_trees):
            if self.performance_counter[i] > 0:
                continue
            sample_idx = random.sample(range(len(X)), self.sample_size)
            self.trees[i] = IsolationTree(self.height_limit).fit(X[sample_idx, :], improved)
        
        # Reset the performance counters
        self.performance_counter = np.zeros(self.n_trees)
        
        # Get the predictions and scores
        scores = self.anomaly_score(X)
        self.threshold = np.percentile(scores, 100 * (1 - self.contamination))
        preds = [1 if score >= self.threshold else 0 for score in scores]
        
        return preds, scores
        
        
    def stream(self, X: np.ndarray, drift_detector, improved=False):
        """
        Assumes that the X is being streamed. Continuously updates the forest 
        while giving predictions on the data. (Discards the current forest at
        the start of the stream)
        Returns the predictions, scores for the streamed data.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        window = deque(maxlen=self.window_size)
        for i in range(self.window_size):
            window.append(X[i])
        
        # Build the initial forest
        preds, scores = self.update_forest(np.array(window), improved)
        # cur_anamoly_count = sum(preds)
            
        for (i, x) in enumerate(X, start=self.window_size):
            # Predict for new data point
            window.append(x)
            score = self.anomaly_score(np.array([x]))[0][0]
            pred = 1 if score >= self.threshold else 0
            scores.append(score)
            preds.append(pred)
            # cur_anamoly_count += pred
            # if i >= self.window_size:
            #     cur_anamoly_count -= preds[i - self.window_size]
                
            # Update the performance counters
            for i in range(self.n_trees):
                tree_pred = self.trees[i].anamoly_score(np.array([x]))[0][0] >= self.threshold
                if tree_pred == pred:
                    self.performance_counter[i] += 1
                else:
                    self.performance_counter[i] -= 1
            
            # If concept drift happens, update the forest
            if drift_detector(scores, self.window_size):
                preds_cur, scores_cur = self.update_forest(np.array(window), improved)
                scores = scores[:i - self.window_size] + scores_cur
                preds = preds[:i - self.window_size] + preds_cur
                
        return preds, scores