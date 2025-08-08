# src/ml/anomaly.py
from sklearn.ensemble import IsolationForest
import numpy as np

def fit_isolation_forest(features_df, contamination=0.01):
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    model.fit(features_df)
    return model

def score_and_flag(model, features_df, threshold=None):
    scores = -model.decision_function(features_df)  # higher = more anomalous
    if threshold is None:
        threshold = np.quantile(scores, 1 - model.contamination)
    flags = scores >= threshold
    return scores, flags
