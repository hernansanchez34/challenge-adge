import os
import numpy as np
import networkx as nx
import pandas as pd
from node2vec import Node2Vec
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

# Parámetros fijos
dimensions = 128
walk_length = 80
num_walks = 10
window_size = 10
alpha = 0.025
epochs = 1
workers = 4
min_count = 0

# Grid de p y q
p_values = [0.25, 0.5, 1, 2, 4]
q_values = [0.25, 0.5, 1, 2, 4]

# Cargar grafo
G = nx.read_edgelist('graph/blogcatalog.edgelist', nodetype=int, create_using=nx.DiGraph())
for edge in G.edges():
    G[edge[0]][edge[1]]['weight'] = 1.0

# Cargar etiquetas
df = pd.read_csv('data/blog/data/group-edges.csv', header=None, names=['node', 'group'])
grouped = df.groupby('node')['group'].apply(list)

# Crear carpeta de salida
os.makedirs("emb", exist_ok=True)

# Evaluar cada combinación
results = []
for p in p_values:
    for q in q_values:
        print(f"\n=== p={p}, q={q} ===")
        # Generar caminatas y entrenar
        node2vec = Node2Vec(
            G, dimensions=dimensions, walk_length=walk_length,
            num_walks=num_walks, p=p, q=q, workers=workers, seed=42
        )
        model = node2vec.fit(
            window=window_size,
            min_count=min_count,
            sg=1,
            epochs=epochs,
            alpha=alpha,
            min_alpha=alpha
        )

        emb_path = f"emb/blogcatalog_p{p}_q{q}.emb"
        model.wv.save_word2vec_format(emb_path)

        # === Evaluar F1 ===
        # Cargar embeddings
        with open(emb_path, 'r') as f:
            lines = f.readlines()[1:]
        embeddings = {
            int(line.split()[0]): np.array(list(map(float, line.split()[1:])))
            for line in lines
        }

        nodes = [n for n in grouped.index if n in embeddings]
        if not nodes:
            print("No hay nodos con embeddings. Saltando.")
            continue

        X = np.array([embeddings[n] for n in nodes])
        Y = grouped.loc[nodes]
        Y_bin = MultiLabelBinarizer().fit_transform(Y)

        # Normalizar embeddings
        X = normalize(X)

        rs = ShuffleSplit(n_splits=10, test_size=0.5, random_state=42)
        macro_scores, micro_scores = [], []
        for train_idx, test_idx in rs.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y_bin[train_idx], Y_bin[test_idx]
            clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
            clf.fit(X_train, Y_train)
            Y_pred = clf.predict(X_test)
            macro_scores.append(f1_score(Y_test, Y_pred, average='macro', zero_division=0))
            micro_scores.append(f1_score(Y_test, Y_pred, average='micro', zero_division=0))

        macro_f1 = np.mean(macro_scores)
        micro_f1 = np.mean(micro_scores)
        print(f"Macro-F1: {macro_f1:.4f} | Micro-F1: {micro_f1:.4f}")
        results.append((p, q, macro_f1, micro_f1))

# Guardar resultados
df_out = pd.DataFrame(results, columns=["p", "q", "Macro-F1", "Micro-F1"])
df_out.to_csv("archivos_csv/f1_gridsearch_results.csv", index=False)
print("\n=== Grid Search completado. Resultados guardados en archivos_csv/f1_gridsearch_results.csv ===")
