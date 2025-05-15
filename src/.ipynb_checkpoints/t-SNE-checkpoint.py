import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import community.community_louvain as community

# === Configura tus rutas ===
embedding_path = "emb/karate_nuevo.emb"
graph_path = "graph/karate.edgelist"

# Función para cargar embeddings desde archivo .emb
def load_embeddings(path):
    with open(path, 'r') as f:
        lines = f.readlines()[1:]
        labels = []
        vectors = []
        for line in lines:
            parts = line.strip().split()
            labels.append(int(parts[0]))
            vectors.append([float(x) for x in parts[1:]])
    return labels, np.array(vectors)

# Cargar grafo y embeddings
G = nx.read_edgelist(graph_path, nodetype=int)
labels, vectors = load_embeddings(embedding_path)

# Detección de comunidades con Louvain
partition = community.best_partition(G)
communities = [partition[node] for node in labels]
unique_comms = sorted(set(communities))
color_map = {comm: plt.cm.tab10(i % 10) for i, comm in enumerate(unique_comms)}
node_colors = [color_map[partition[node]] for node in labels]

# Aplicar t-SNE a los vectores
tsne = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=1000)
reduced_tsne = tsne.fit_transform(vectors)

# Posición de cada nodo en 2D (t-SNE)
pos_tsne = {node: reduced_tsne[i] for i, node in enumerate(labels)}

# Visualización
plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(G, pos_tsne, node_size=300, node_color=node_colors, edgecolors='k')
nx.draw_networkx_edges(G, pos_tsne, alpha=0.3)
nx.draw_networkx_labels(G, pos_tsne, font_size=10)

plt.title("Embeddings node2vec + comunidades Louvain (t-SNE)")
plt.axis('off')
plt.tight_layout()
#plt.show()
plt.savefig("embeddings_t-SNE.png", dpi=300)