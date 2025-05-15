import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
import community.community_louvain as community

# Cargar los embeddings desde archivo .emb
def load_embeddings(path):
    with open(path, 'r') as f:
        lines = f.readlines()[1:]  # saltar cabecera
        labels = []
        vectors = []
        for line in lines:
            parts = line.strip().split()
            labels.append(int(parts[0]))
            vectors.append([float(x) for x in parts[1:]])
    return labels, np.array(vectors)

# === Configura tus rutas ===
embedding_path = "emb/karate_nuevo.emb"
graph_path = "graph/karate.edgelist"

# Cargar grafo y embeddings
G = nx.read_edgelist(graph_path, nodetype=int)
labels, vectors = load_embeddings(embedding_path)

# Reducir dimensión a 2D (PCA)
pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

# Posición de cada nodo en 2D
pos = {node: reduced[i] for i, node in enumerate(labels)}

# Detección de comunidades con Louvain
partition = community.best_partition(G)
communities = [partition[node] for node in labels]

# Mapa de colores para comunidades
unique_comms = sorted(set(communities))
color_map = {comm: plt.cm.tab10(i % 10) for i, comm in enumerate(unique_comms)}
node_colors = [color_map[partition[node]] for node in labels]

# Visualizar
plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, edgecolors='k')
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_labels(G, pos, font_size=10)

plt.title("Embeddings node2vec + comunidades Louvain (PCA)")
plt.axis('off')
plt.tight_layout()
#plt.show()
plt.savefig("embeddings_visualizacion2.png", dpi=300)