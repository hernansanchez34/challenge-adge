import networkx as nx
import random
import numpy as np
import pandas as pd
from itertools import combinations

# 1. Cargar el grafo original
G_full = nx.read_edgelist("data/facebook/facebook_combined.txt", nodetype=int)
original_edges = list(G_full.edges())
num_edges_to_remove = len(original_edges) // 2

# 2. Eliminar 50% de las aristas aleatoriamente pero mantener conectividad
random.seed(42)
removed_edges = []
G = G_full.copy()

attempts = 0
while len(removed_edges) < num_edges_to_remove and attempts < 10 * num_edges_to_remove:
    edge = random.choice(original_edges)
    if G.has_edge(*edge):
        G.remove_edge(*edge)
        if nx.is_connected(G):
            removed_edges.append(edge)
        else:
            G.add_edge(*edge)
    attempts += 1

# 3. Crear ejemplos positivos (edges removidos)
positive_pairs = removed_edges

# 4. Crear ejemplos negativos (pares no conectados)
non_edges = list(nx.non_edges(G))
random.shuffle(non_edges)
negative_pairs = non_edges[:len(positive_pairs)]

# 5. Guardar dataset
df_pos = pd.DataFrame(positive_pairs, columns=["node1", "node2"])
df_pos["label"] = 1
df_neg = pd.DataFrame(negative_pairs, columns=["node1", "node2"])
df_neg["label"] = 0
df_all = pd.concat([df_pos, df_neg], ignore_index=True)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

df_all.to_csv("data/link_prediction_dataset.csv", index=False)
df_summary = {
    "Nodos": [G.number_of_nodes()],
    "Aristas después de remoción": [G.number_of_edges()],
    "Ejemplos positivos": [len(positive_pairs)],
    "Ejemplos negativos": [len(negative_pairs)],
    "Conectado": [nx.is_connected(G)]
}

