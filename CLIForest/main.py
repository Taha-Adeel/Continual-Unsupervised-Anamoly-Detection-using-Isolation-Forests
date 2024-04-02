from models.iforest import IsolationForest
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from util.utils import find_TPR_threshold

def main():
    # Load data
    data = pd.read_csv('../datasets/HTTP.csv')
    X = data.drop('3', axis=1)
    y = data['3']

    # Train Isolation Forest
    clf = IsolationForest(sample_size=256, n_trees=100)
    clf.fit(X)

    # Predict
    scores = clf.anomaly_score(X)
    threshold, FPR = find_TPR_threshold(y, scores, 0.8)
    y_pred = clf.predict_from_anomaly_scores(scores, threshold)
    
    # Evaluate
    print(f'Confusion matrix: {confusion_matrix(y, y_pred)}')
    print(f'Classification report: {classification_report(y, y_pred)}')
    print(f'ROC AUC score: {roc_auc_score(y, y_pred)}')
    print(f'Threshold: {threshold}')
    print(f'FPR: {FPR}')
    
    
if __name__ == '__main__':
    main()