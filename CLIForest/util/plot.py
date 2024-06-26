import numpy as np
import pandas as pd
import time
from sklearn.metrics import confusion_matrix, f1_score, average_precision_score
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from models.iforest import IsolationForest
from util.utils import find_TPR_threshold


def plot_anomalies(X, y, sample_size=256, n_trees = 100, desired_TPR=None, percentile = None, normal_ymax=None, bins=20):
    N = len(X)

    it = IsolationForest(sample_size=sample_size, n_trees=n_trees)

    fit_start = time.time()
    it.fit(X)
    fit_stop = time.time()
    fit_time = fit_stop - fit_start
    print(f"fit time {fit_time:3.2f}s")

    score_start = time.time()
    scores = it.anomaly_score(X)
    score_stop = time.time()
    score_time = score_stop - score_start
    print(f"score time {score_time:3.2f}s")

    if desired_TPR is not None:
        threshold, FPR = find_TPR_threshold(y, scores, desired_TPR)
        print(f"Computed {desired_TPR:.4f} TPR threshold {threshold:.4f} with FPR {FPR:.4f}")
    else:
        threshold = np.percentile(scores, percentile)
    y_pred = it.predict_from_anomaly_scores(scores, threshold=threshold)
    confusion = confusion_matrix(y, y_pred)
    print(confusion)

    TN, FP, FN, TP = confusion.flat
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    normal = scores[y==0]
    anomalies = scores[y==1]
    F1 = f1_score(y, y_pred)
    PR = average_precision_score(y, scores)
    print(f"Proportion anomalies/normal = {len(anomalies)}/{len(normal)} = {(len(anomalies)/len(normal))*100:.1f}%")
    print(f"F1 score {F1:.4f}, avg PR {PR:.4f}")

    fig, axes = plt.subplots(2, 1, sharex=True)
    counts0, binlocs0, _ = axes[0].hist(normal, color='#c7e9b4', bins=bins)
    counts1, binlocs1, _ = axes[1].hist(anomalies, color='#fee090', bins=bins)
    axes[1].set_xlabel("Anomaly score")
    axes[0].set_ylabel("Normal sample count")
    axes[1].set_ylabel("Anomalous sample count")
    axes[0].plot([threshold,threshold],[0,max(counts0)], '--', color='grey')
    axes[1].plot([threshold,threshold],[0,max(counts1)], '--', color='grey')
    text_xr = 0.97 * axes[0].get_xlim()[1]
    axes[0].text(text_xr, .85 * max(counts0), f"N {N}, {n_trees} trees", horizontalalignment='right')
    axes[0].text(text_xr, .75 * max(counts0), f"F1 score {F1:.4f}, avg PR {PR:.4f}", horizontalalignment='right')
    axes[0].text(text_xr, .65 * max(counts0), f"TPR {TPR:.4f}, FPR {FPR:.4f}", horizontalalignment='right')
    axes[0].text(threshold+.005, .20 * max(counts0), f"score threshold {threshold:.3f}")
    axes[0].text(threshold+.005, .10 * max(counts0), f"True anomaly rate {len(anomalies) / len(normal):.4f}")
    if normal_ymax is not None:
        axes[0].set_ylim(0, normal_ymax)
    plt.tight_layout()
    plt.savefig(f"{datafile.split('.')[0]}-{n_trees}-{int(desired_TPR*100)}.svg",
                bbox_inches='tight',
                pad_inches=0)
    plt.show()
