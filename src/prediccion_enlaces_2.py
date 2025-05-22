# Reejecutar tras reinicio de estado
from node2vec import Node2Vec
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import normalize

# Cargar el grafo degradado
G = nx.read_edgelist("data/facebook/facebook_combined.txt", nodetype=int)
dataset = pd.read_csv("data/link_prediction_dataset.csv")

# Eliminar aristas positivas
positive_edges = dataset[dataset["label"] == 1][["node1", "node2"]].values
G_removed = G.copy()
G_removed.remove_edges_from(positive_edges)

# Entrenar Node2Vec sobre el grafo incompleto
node2vec = Node2Vec(
    G_removed,
    dimensions=128,
    walk_length=80,
    num_walks=10,
    p=4,
    q=4,
    workers=4,
    seed=42
)
model = node2vec.fit(window=10, min_count=0, sg=1, epochs=1, alpha=0.025, min_alpha=0.025)

# Extraer embeddings
embeddings = {str(node): model.wv[str(node)] for node in G_removed.nodes() if str(node) in model.wv}

# Operador Hadamard
def hadamard(u, v):
    return u * v

# Generar vectores de características
X = []
y = []

for _, row in dataset.iterrows():
    n1, n2, label = str(row["node1"]), str(row["node2"]), row["label"]
    if n1 in embeddings and n2 in embeddings:
        vec = hadamard(embeddings[n1], embeddings[n2])
        X.append(vec)
        y.append(label)

X = normalize(np.array(X))
y = np.array(y)

# Guardar para evaluación
np.savez("data/link_prediction_features.npz", X=X, y=y)
