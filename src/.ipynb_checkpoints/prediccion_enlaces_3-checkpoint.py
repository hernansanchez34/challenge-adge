from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

# Cargar
data = np.load("data/link_prediction_features.npz")
X, y = data["X"], data["y"]

# Entrenar y evaluar
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)
y_prob = clf.predict_proba(X)[:, 1]
auc = roc_auc_score(y, y_prob)

print(f"AUC: {auc:.4f}")
