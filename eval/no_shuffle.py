import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

# ====== PARÁMETRO AJUSTABLE ======
min_samples_per_class = 0
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

embeddings = load_embeddings("emb/blogcatalog_prueba.emb")

# 2. Cargar etiquetas reales desde group-edges.csv
df = pd.read_csv('data/blog/data/group-edges.csv', header=None, names=['node', 'group'])

# 3. Filtrar clases con pocas apariciones
group_counts = df['group'].value_counts()
grupos_frecuentes = group_counts[group_counts >= min_samples_per_class].index
df_filtrado = df[df['group'].isin(grupos_frecuentes)]

# 4. Agrupar por nodo
grouped = df_filtrado.groupby('node')['group'].apply(list)
nodes_with_embedding = [n for n in grouped.index if n in embeddings]

if len(nodes_with_embedding) == 0:
    raise ValueError("No quedan nodos válidos con embeddings y etiquetas después del filtrado.")

X = np.array([embeddings[n] for n in nodes_with_embedding])
Y = grouped.loc[nodes_with_embedding]

# 5. Binarizar etiquetas
mlb = MultiLabelBinarizer()
Y_bin = mlb.fit_transform(Y)

# 6. División 50/50 sin shuffle
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_bin, test_size=0.5, shuffle=False
)

# 7. Entrenamiento y predicción
clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

# 8. Métricas
macro = f1_score(Y_test, Y_pred, average='macro', zero_division=0)
micro = f1_score(Y_test, Y_pred, average='micro'), zero_division=0)

# 9. Mostrar resultados
print("Etiquetas retenidas:", len(mlb.classes_))
print("Nodos evaluados:", len(X))
print("Macro-F1:", macro)
print("Micro-F1:", micro)
