import json
import networkx as nx

# Ruta al JSON original
json_path = "data/ppi-G.json"
# Ruta destino del edgelist
output_path = "data/ppi.edgelist"

# Cargar grafo desde archivo JSON (formato node-link)
with open(json_path, 'r') as f:
    data = json.load(f)

G = nx.readwrite.node_link_graph(data)

# Guardar en formato .edgelist sin pesos
nx.write_edgelist(G, output_path, data=False)

print(f"Grafo exportado exitosamente a: {output_path}")
print(f"Nodos: {G.number_of_nodes()}, Aristas: {G.number_of_edges()}")
