import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import random

# Carpetas
os.makedirs("archivos_csv", exist_ok=True)
os.makedirs("graficas", exist_ok=True)

# Cargar grafo original
G_original = nx.read_edgelist("graph/blogcatalog.edgelist", nodetype=int, create_using=nx.DiGraph())
for u, v in G_original.edges():
    G_original[u][v]['weight'] = 1.0
nodes_list = list(G_original.nodes())

# Cargar etiquetas
df = pd.read_csv("data/blog/data/group-edges.csv", header=None, names=["node", "group"])
grouped = df.groupby("node")["group"].apply(list)
mlb = MultiLabelBinarizer()
Y_bin = mlb.fit_transform(grouped)
nodes = grouped.index.tolist()
labels_df = pd.DataFrame(Y_bin, columns=[f"label_{i}" for i in range(Y_bin.shape[1])])
labels_df.insert(0, "node", nodes)

# Parámetros node2vec
dimensions = 128
walk_length = 80
num_walks = 10
p = 1
q = 1
window = 10
epochs = 1
alpha = 0.025
workers = 4

# Fracciones de aristas a agregar
fractions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
f1_scores = []

for frac in fractions:
    print(f"\n=== Agregando {int(frac*100)}% de aristas aleatorias ===")
    G = G_original.copy()
    num_edges = G.number_of_edges()
    num_add = int(frac * num_edges)

    added = 0
    attempts = 0
    random.seed(42)
    while added < num_add and attempts < num_add * 10:
        u, v = random.sample(nodes_list, 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=1.0)
            added += 1
        attempts += 1

    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
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
df_noise = pd.DataFrame({
    "fraction_added": fractions,
    "Macro-F1": f1_scores
})
df_noise.to_csv("archivos_csv/f1_vs_added_edges.csv", index=False)

# Graficar
plt.figure(figsize=(8, 5))
plt.plot(fractions, f1_scores, marker='o', linestyle='-', color='navy')
plt.title("Macro-F1 vs fracción de aristas agregadas (ruido) en BlogCatalog")
plt.xlabel("Fracción de aristas aleatorias agregadas")
plt.ylabel("Macro-F1")
plt.grid(True)
plt.tight_layout()
plt.savefig("graficas/f1_vs_added_edges.png", dpi=300)
print("Gráfico guardado en graficas/f1_vs_added_edges.png")
