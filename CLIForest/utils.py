import numpy as np
import random
from sklearn.metrics import confusion_matrix


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    threshold = 1

    while threshold > 0:
        y_pred = [1 if p[0] >= threshold else 0 for p in scores]
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        TPR = tp / (tp + fn)
        FPR = fp / (fp + tn)
        if TPR >= desired_TPR:
            return threshold, FPR

        threshold = threshold - 0.001

    return threshold, FPR



def c(n):
    """
    Average path length of an unsuccessful search in a binary search tree given n nodes.
    (From the Isolation Forest paper. 0.57.. is Eulers constant.)
    """
    if n > 2:
        return 2 * (np.log(n-1) + 0.5772156649) - (2 * (n-1) / n)
    elif n == 2:
        return 1
    if n == 1:
        return 0