import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
#source node2vec-310/bin/activate
# === Configura tus rutas ===
embeddings_path = "emb/ppi.emb"  # Cambia por tu archivo real
class_map_path = "data/ppi-class_map.json"

# === Función para cargar embeddings desde archivo .emb ===
def load_embeddings(emb_path):
    with open(emb_path, 'r') as f:
        lines = f.readlines()[1:]
        embeddings = {}
        for line in lines:
            parts = line.strip().split()
            node_id = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            embeddings[node_id] = vector
    return embeddings

# === Cargar etiquetas multietiqueta ===
with open(class_map_path, "r") as f:
    class_map = json.load(f)

# === Cargar embeddings ===
embeddings = load_embeddings(embeddings_path)

# === Cruzar nodos con embeddings y etiquetas ===
X = []
Y = []
for node_id in class_map:
    if node_id in embeddings:
        X.append(embeddings[node_id])
        Y.append(class_map[node_id])

X = np.array(X)
Y = np.array(Y)

# === División entrenamiento/prueba 50% como en el paper ===
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

# === Entrenar clasificador multietiqueta ===
clf = OneVsRestClassifier(LogisticRegression(max_iter=500))
clf.fit(X_train, Y_train)

# === Predecir y evaluar ===
Y_pred = clf.predict(X_test)
micro = f1_score(Y_test, Y_pred, average='micro')
macro = f1_score(Y_test, Y_pred, average='macro')

# === Mostrar resultados ===
print(f"Micro-F1: {micro:.4f}")
print(f"Macro-F1: {macro:.4f}")