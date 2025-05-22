import os
import time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec

# Crear carpetas
os.makedirs("archivos_csv", exist_ok=True)
os.makedirs("graficas", exist_ok=True)

# Tamaños de grafos a probar
sizes = [100, 300, 1000, 3000, 10000, 100000]
sampling_times = []
total_times = []

# Parámetros fijos
avg_degree = 10
dimensions = 128
walk_length = 80
num_walks = 10
p = 1
q = 1
window = 10
epochs = 1
workers = 2

for n in sizes:
    print(f"\n== Grafo con {n} nodos ==")
    prob = avg_degree / (n - 1)
    G = nx.erdos_renyi_graph(n, prob, seed=42)
    
    # Sampling time
    start_sample = time.time()
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
    sampling_time = time.time() - start_sample

    # Optimization time
    start_total = time.time()
    model = node2vec.fit(
        window=window,
        min_count=0,
        sg=1,
        epochs=epochs,
        alpha=0.025,
        min_alpha=0.025
    )
    total_time = time.time() - start_sample

    sampling_times.append(sampling_time)
    total_times.append(total_time)
    print(f"Sampling: {sampling_time:.2f}s | Total: {total_time:.2f}s")

# Guardar CSV
df = pd.DataFrame({
    "nodes": sizes,
    "sampling_time": sampling_times,
    "total_time": total_times
})
df.to_csv("archivos_csv/scalability_times.csv", index=False)

# Graficar (log-log)
plt.figure(figsize=(8, 5))
plt.plot(np.log10(sizes), np.log10(total_times), 'o-', label='sampling + optimization time', color='red')
plt.plot(np.log10(sizes), np.log10(sampling_times), '^-', label='sampling time', color='blue')
plt.xlabel("log₁₀(nodes)")
plt.ylabel("log₁₀(time in seconds)")
plt.title("Scalability of node2vec (average degree 10)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("graficas/scalability_node2vec.png", dpi=300)
print("Gráfico guardado en graficas/scalability_node2vec.png")
