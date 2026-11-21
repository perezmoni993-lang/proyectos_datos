"""
Actividad 3.2: Técnicas No Lineales - t-SNE (t-Distributed Stochastic Neighbor Embedding)
Asignatura: Extracción de Conocimiento en Bases de Datos - Unidad IV
"""
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE

# Cargar dataset multivariado Iris (4 características)
iris = load_iris()
X_iris = iris.data

# Reducción no lineal con t-SNE a 2 dimensiones
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X_iris)

print("=== 3.2 REDUCCIÓN NO LINEAL CON t-SNE ===")
print(f"Dimensión Original de Entrada: {X_iris.shape}")
print(f"Dimensión Reducida con t-SNE: {X_tsne.shape}")
print(f"Divergencia KL Final (Error): {tsne.kl_divergence_:.4f}")