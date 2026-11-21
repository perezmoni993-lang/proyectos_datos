"""
Actividad 2.3: Agrupación Basada en Densidad (DBSCAN)
Asignatura: Extracción de Conocimiento en Bases de Datos - Unidad IV
"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Generación del dataset sintético de clientes: [Monto total gastado, Frecuencia anual]
np.random.seed(42)
grupo1 = np.random.normal(loc=[200, 5], scale=[30, 2], size=(50, 2))
grupo2 = np.random.normal(loc=[1500, 25], scale=[150, 4], size=(50, 2))
grupo3 = np.random.normal(loc=[800, 12], scale=[80, 3], size=(50, 2))
X_clientes = np.vstack([grupo1, grupo2, grupo3])

# Escalado estandarizado necesario para métricas de distancia en DBSCAN
scaler_db = StandardScaler()
X_scaled = scaler_db.fit_transform(X_clientes)

# Configuración de DBSCAN: eps (radio de vecindad), min_samples (mínimo de puntos)
dbscan = DBSCAN(eps=0.4, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)

n_clusters_db = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise = list(labels_dbscan).count(-1)

print("=== 2.3 DBSCAN (DENSITY-BASED CLUSTERING) ===")
print(f"Número de Clusters Detectados automáticamente: {n_clusters_db}")
print(f"Número de Puntos Ruidosos (Outliers / -1): {n_noise}")
print("Conteo por etiqueta de grupo:", dict(zip(*np.unique(labels_dbscan, return_counts=True))))