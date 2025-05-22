import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

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

# Cargar etiquetas
df = pd.read_csv("data/blog/data/group-edges.csv", header=None, names=['node', 'group'])
grouped = df.groupby('node')['group'].apply(list)
mlb = MultiLabelBinarizer()
train_sizes = np.arange(0.1, 1.0, 0.1)

# Ruta de los embeddings
paths = {
    "DeepWalk (p=1, q=1)": "emb/blogcatalog_p1_q1.emb",
    "node2vec (p=4, q=4)": "emb/blogcatalog_p4_q4.emb"
}

# Diccionarios para resultados
results = {
    "DeepWalk (p=1, q=1)": {"Micro-F1": [], "Macro-F1": []},
    "node2vec (p=4, q=4)": {"Micro-F1": [], "Macro-F1": []}
}

# Procesar cada embedding
for label, path in paths.items():
    embeddings = load_embeddings(path)
    nodes_with_embedding = [n for n in grouped.index if n in embeddings]
    X = np.array([embeddings[n] for n in nodes_with_embedding])
    Y = grouped.loc[nodes_with_embedding]
    Y_bin = mlb.fit_transform(Y)
    X = normalize(X)

    for train_size in train_sizes:
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y_bin, train_size=train_size, random_state=42, shuffle=True
        )
        clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        macro = f1_score(Y_test, Y_pred, average='macro', zero_division=0)
        micro = f1_score(Y_test, Y_pred, average='micro', zero_division=0)
        results[label]["Macro-F1"].append(macro)
        results[label]["Micro-F1"].append(micro)

# Crear carpetas de salida
os.makedirs("graficas", exist_ok=True)
os.makedirs("archivos_csv", exist_ok=True)

# Crear DataFrame de resultados
train_pct = (train_sizes * 100).astype(int)
df_micro = pd.DataFrame({
    "Train %": train_pct,
    "DeepWalk (Micro-F1)": results["DeepWalk (p=1, q=1)"]["Micro-F1"],
    "node2vec (Micro-F1)": results["node2vec (p=4, q=4)"]["Micro-F1"]
})
df_macro = pd.DataFrame({
    "Train %": train_pct,
    "DeepWalk (Macro-F1)": results["DeepWalk (p=1, q=1)"]["Macro-F1"],
    "node2vec (Macro-F1)": results["node2vec (p=4, q=4)"]["Macro-F1"]
})

# Guardar CSV
df_combined = pd.concat([df_micro, df_macro.drop(columns="Train %")], axis=1)
df_combined.to_csv("archivos_csv/comparacion_deepwalk_node2vec_f1.csv", index=False)
print("Resultados guardados en archivos_csv/comparacion_deepwalk_node2vec_f1.csv")

# Graficar Micro-F1
plt.figure(figsize=(8, 5))
plt.plot(train_pct, results["DeepWalk (p=1, q=1)"]["Micro-F1"], marker='s', linestyle='--', label='DeepWalk (p=1, q=1)', color='orange')
plt.plot(train_pct, results["node2vec (p=4, q=4)"]["Micro-F1"], marker='o', label='node2vec (p=4, q=4)', color='blue')
plt.title("Comparación Micro-F1: DeepWalk vs node2vec")
plt.xlabel("Porcentaje de entrenamiento (%)")
plt.ylabel("Micro-F1")
plt.xticks(train_pct)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("graficas/comparacion_micro_f1.png", dpi=300)

# Graficar Macro-F1
plt.figure(figsize=(8, 5))
plt.plot(train_pct, results["DeepWalk (p=1, q=1)"]["Macro-F1"], marker='s', linestyle='--', label='DeepWalk (p=1, q=1)', color='orange')
plt.plot(train_pct, results["node2vec (p=4, q=4)"]["Macro-F1"], marker='o', label='node2vec (p=4, q=4)', color='blue')
plt.title("Comparación Macro-F1: DeepWalk vs node2vec")
plt.xlabel("Porcentaje de entrenamiento (%)")
plt.ylabel("Macro-F1")
plt.xticks(train_pct)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("graficas/comparacion_macro_f1.png", dpi=300)

print("Gráficos guardados en graficas/")
