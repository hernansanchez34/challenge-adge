import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

# 1. Cargar embeddings desde .emb
def load_embeddings(path):
    with open(path, 'r') as f:
        lines = f.readlines()[1:]  # saltar encabezado
    emb = {}
    for line in lines:
        parts = line.strip().split()
        node = int(parts[0])
        vector = np.array(list(map(float, parts[1:])))
        emb[node] = vector
    return emb

embeddings = load_embeddings("emb/blogcatalog_prueba.emb")

# 2. Cargar etiquetas desde groups.csv
df = pd.read_csv('data/blog/data/group-edges.csv', header=None, names=['node', 'group'])
grouped = df.groupby('node')['group'].apply(list)

# 3. Filtrar nodos que tengan embedding
nodes_with_embedding = [n for n in grouped.index if n in embeddings]
X = np.array([embeddings[n] for n in nodes_with_embedding])
Y = grouped.loc[nodes_with_embedding]

# 4. Binarizar etiquetas
mlb = MultiLabelBinarizer()
Y_bin = mlb.fit_transform(Y)

# 5. Repetir 10 splits aleatorios 50/50 y promediar métricas
rs = ShuffleSplit(n_splits=10, test_size=0.5, random_state=42)
macro_scores = []
micro_scores = []

for train_idx, test_idx in rs.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y_bin[train_idx], Y_bin[test_idx]

    clf = OneVsRestClassifier(LogisticRegression(penalty="l2",max_iter=1000))
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    macro = f1_score(Y_test, Y_pred, average='macro', zero_division=0)
    micro = f1_score(Y_test, Y_pred, average='micro', zero_division=0)

    macro_scores.append(macro)
    micro_scores.append(micro)

# 6. Mostrar resultados promedio
print("Macro-F1 promedio:", np.mean(macro_scores))
print("Micro-F1 promedio:", np.mean(micro_scores))
