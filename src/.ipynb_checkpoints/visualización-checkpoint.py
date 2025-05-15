import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE  # Alternativa para visualización más avanzada

# Cargar los embeddings desde archivo .emb
def load_embeddings(path):
    with open(path, 'r') as f:
        lines = f.readlines()[1:]  # Saltar la primera línea
        labels = []
        vectors = []
        for line in lines:
            parts = line.strip().split()
            labels.append(parts[0])
            vectors.append([float(x) for x in parts[1:]])
    return labels, np.array(vectors)

# Ruta a tu archivo .emb
ruta = "emb/karate_nuevo.emb"  # cámbiala si es necesario
labels, vectors = load_embeddings(ruta)

# Reducción de dimensionalidad con PCA (puedes cambiar a t-SNE si quieres)
pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

# Graficar
plt.figure(figsize=(8, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c='blue', edgecolor='k')

# Etiquetas de los nodos
for i, label in enumerate(labels):
    plt.annotate(label, (reduced[i, 0], reduced[i, 1]), fontsize=8, alpha=0.7)

plt.title("Visualización de embeddings con PCA")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig("embeddings_visualizacion.png", dpi=300)