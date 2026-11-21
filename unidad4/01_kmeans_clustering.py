"""
Actividad 2.1: Agrupación K-Means (K-Means Clustering)
Asignatura: Extracción de Conocimiento en Bases de Datos - Unidad IV
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Generación de datos sintéticos: [Monto total gastado, Frecuencia de compra anual]
np.random.seed(42)
grupo1 = np.random.normal(loc=[200, 5], scale=[30, 2], size=(50, 2))
grupo2 = np.random.normal(loc=[1500, 25], scale=[150, 4], size=(50, 2))
grupo3 = np.random.normal(loc=[800, 12], scale=[80, 3], size=(50, 2))
X_clientes = np.vstack([grupo1, grupo2, grupo3])

# 1. Configurar y entrenar K-Means con K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_clientes)

# 2. Cálculo de métricas e inspección de centroides
inercia = kmeans.inertia_
silueta = silhouette_score(X_clientes, labels_kmeans)
centroides = kmeans.cluster_centers_

print("=== 2.1 CLUSTERIZACIÓN K-MEANS ===")
print(f"Inercia Intracluster (WCSS): {inercia:.2f}")
print(f"Coeficiente de Silueta: {silueta:.4f}")
print("\nCentroides de los Grupos (Monto Promedio, Frecuencia Promedio):")
for i, c in enumerate(centroides):
    print(f"  Cluster {i}: Gastos de ${c[0]:.2f} USD | Frecuencia: {c[1]:.1f} compras/año")