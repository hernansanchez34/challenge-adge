import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

# ====== PARÁMETRO AJUSTABLE ======
min_samples_per_class = 0  # puedes subir a 20 si quieres más limpieza
# =================================

# 1. Cargar embeddings
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

embeddings = load_embeddings('emb/blogcatalog_lib.emb')

# 2. Cargar y filtrar etiquetas poco frecuentes
df = pd.read_csv('data/blog/data/group-edges.csv', header=None, names=['node', 'group'])
group_counts = df['group'].value_counts()
grupos_frecuentes = group_counts[group_counts >= min_samples_per_class].index
df_filtrado = df[df['group'].isin(grupos_frecuentes)]

# 3. Agrupar etiquetas por nodo
grouped = df_filtrado.groupby('node')['group'].apply(list)

# 4. Filtrar nodos que tengan embedding
nodes_with_embedding = [n for n in grouped.index if n in embeddings]

if len(nodes_with_embedding) == 0:
    raise ValueError(
        f"No quedan nodos válidos con embeddings y etiquetas después del filtrado (mínimo {min_samples_per_class} por clase). "
        "Prueba con un umbral más bajo."
    )

X = np.array([embeddings[n] for n in nodes_with_embedding])
Y = grouped.loc[nodes_with_embedding]

# 5. Binarizar etiquetas
mlb = MultiLabelBinarizer()
Y_bin = mlb.fit_transform(Y)

# 6. Repetir múltiples splits y evaluar
rs = ShuffleSplit(n_splits=10, test_size=0.5, random_state=42)
macro_scores = []
micro_scores = []

for train_idx, test_idx in rs.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y_bin[train_idx], Y_bin[test_idx]

    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    macro = f1_score(Y_test, Y_pred, average='macro', zero_division=0)
    micro = f1_score(Y_test, Y_pred, average='micro', zero_division=0)

    macro_scores.append(macro)
    micro_scores.append(micro)

# 7. Mostrar resultados
print("Etiquetas retenidas:", len(mlb.classes_))
print("Nodos evaluados:", len(X))
print("Macro-F1 promedio:", np.mean(macro_scores))
print("Micro-F1 promedio:", np.mean(micro_scores))
