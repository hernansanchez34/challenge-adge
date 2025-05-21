import pandas as pd

# Leer el archivo CSV
edges = pd.read_csv("data/blog/data/edges.csv")

# Crear archivo edgelist sin cabecera, con espacio como separador
edges.to_csv("graph/blogcatalog.edgelist", sep=" ", index=False, header=False)
