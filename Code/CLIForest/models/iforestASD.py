import numpy as np
import pandas as pd
from collections import deque
from models.iforest import IsolationForest

class IsolationForestASD(IsolationForest):
    """
    An ensemble of Isolation Trees for streaming data, based on (Ding & Fei, 2013) paper.
    If the anamoly rate in the current sliding window is more than the threshold, the 
    current forest is discarded and a new forest is built using the current window. 
    """
    
    def __init__(self, sample_size, contamination_rate=0.1, n_trees=30, window_size=1000):
        """
        Set the sub sampling size, window size and the number of trees in the forest.
        """
        super().__init__(sample_size, contamination_rate, n_trees)
        self.window_size = window_size
        
        
    def update_forest(self, X: np.ndarray, improved=False):
        """
        Discard the current forest and build a new one using the current window.
        Returns the predictions, scores for the current window.
        """
        self.trees = [None] * self.n_trees
        self.fit(X, improved)
        scores = self.anomaly_score(X)
        self.threshold = np.percentile(scores, 100 * (1 - self.contamination))
        preds = [1 if s[0] >= self.threshold else 0 for s in scores]
        
        return preds, scores
        
        
    def stream(self, X: np.ndarray, improved=False):
        """
        Assumes that the X is being streamed. Continuously updates the
        forest while giving predictions on the data. (Discards the current
        forest at the start of the stream)
        Returns the predictions, scores for the streamed data.
        """
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        window = deque(maxlen=self.window_size)
        for i in range(self.window_size):
            window.append(X[i])
        X = X[self.window_size:]
        
        # Build the initial forest
        preds, scores = self.update_forest(np.array(window), improved)
        cur_anamoly_count = sum(preds)
            
        for (i, x) in enumerate(X, start=self.window_size):
            # Predict for new data point
            window.append(x)
            score = self.anomaly_score(np.array([x]))[0][0]
            pred = 1 if score >= self.threshold else 0
            scores.append(score)
            preds.append(pred)
            cur_anamoly_count += pred
            if i >= self.window_size:
                cur_anamoly_count -= preds[i - self.window_size]
            
            # If concept drift happens, update the forest
            if cur_anamoly_count / self.window_size > self.contamination:
                preds_cur, scores_cur = self.update_forest(np.array(window), improved)
                scores = scores[:i - self.window_size] + scores_cur
                preds = preds[:i - self.window_size] + preds_cur
                
        return preds, scores