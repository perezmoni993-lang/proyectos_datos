"""
Actividad 2.2: Agrupación Jerárquica (Agglomerative Hierarchical Clustering)
Asignatura: Extracción de Conocimiento en Bases de Datos - Unidad IV
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Generación del mismo dataset sintético de clientes: [Monto total gastado, Frecuencia anual]
np.random.seed(42)
grupo1 = np.random.normal(loc=[200, 5], scale=[30, 2], size=(50, 2))
grupo2 = np.random.normal(loc=[1500, 25], scale=[150, 4], size=(50, 2))
grupo3 = np.random.normal(loc=[800, 12], scale=[80, 3], size=(50, 2))
X_clientes = np.vstack([grupo1, grupo2, grupo3])

# Aplicación de Clustering Jerárquico Aglomerativo con enlace Ward
hierarchical = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
labels_hierarchical = hierarchical.fit_predict(X_clientes)

silueta_h = silhouette_score(X_clientes, labels_hierarchical)

print("=== 2.2 CLUSTERIZACIÓN JERÁRQUICA AGLOMERATIVA ===")
print(f"Número de Clusters asignados: {hierarchical.n_clusters_}")
print("Métrica de Distancia / Enlace: Euclidiana / Ward")
print(f"Coeficiente de Silueta: {silueta_h:.4f}")
print("Distribución de elementos por cluster:", np.bincount(labels_hierarchical))