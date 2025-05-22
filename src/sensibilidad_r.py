import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

# Crear carpetas si no existen
os.makedirs("emb", exist_ok=True)
os.makedirs("graficas", exist_ok=True)
os.makedirs("archivos_csv", exist_ok=True)

# Cargar grafo
G = nx.read_edgelist("graph/blogcatalog.edgelist", nodetype=int, create_using=nx.DiGraph())
for u, v in G.edges():
    G[u][v]['weight'] = 1.0

# Cargar etiquetas
df = pd.read_csv("data/blog/data/group-edges.csv", header=None, names=["node", "group"])
grouped = df.groupby("node")["group"].apply(list)
mlb = MultiLabelBinarizer()
Y_bin = mlb.fit_transform(grouped)
nodes = grouped.index.tolist()
labels_df = pd.DataFrame(Y_bin, columns=[f"label_{i}" for i in range(Y_bin.shape[1])])
labels_df.insert(0, "node", nodes)

# Parámetros fijos
dimensions = 128
walk_length = 80
window = 10
p = 1
q = 1
epochs = 1
alpha = 0.025
workers = 4

# Valores de r a probar
r_values = [6, 8, 10, 12, 14, 16, 18, 20]
f1_scores = []

for r in r_values:
    print(f"\n=== Entrenando con r={r} caminatas por nodo ===")
    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=r,
        p=p,
        q=q,
        workers=workers,
        seed=42
    )
    model = node2vec.fit(
        window=window,
        min_count=0,
        sg=1,
        epochs=epochs,
        alpha=alpha,
        min_alpha=alpha
    )

    # Guardar embeddings
    model.wv.save_word2vec_format(f"emb/blogcatalog_r{r}.emb")

    # Evaluar Macro-F1
    X = np.array([model.wv[str(n)] for n in labels_df["node"]])
    Y = labels_df.drop(columns="node").values
    X = normalize(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.5, random_state=42)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    f1 = f1_score(Y_test, Y_pred, average="macro", zero_division=0)
    f1_scores.append(f1)
    print(f"Macro-F1: {f1:.4f}")

# Guardar CSV
df_r = pd.DataFrame({
    "r": r_values,
    "Macro-F1": f1_scores
})
df_r.to_csv("archivos_csv/f1_vs_r.csv", index=False)

# Graficar
plt.figure(figsize=(8, 5))
plt.plot(r_values, f1_scores, marker='o', linestyle='-', color='teal')
plt.title("Macro-F1 vs número de caminatas (r) en BlogCatalog (p=1, q=1)")
plt.xlabel("Número de caminatas por nodo (r)")
plt.ylabel("Macro-F1")
plt.grid(True)
plt.tight_layout()
plt.savefig("graficas/f1_vs_r.png", dpi=300)
print("Gráfico guardado en graficas/f1_vs_r.png")
