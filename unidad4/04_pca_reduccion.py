"""
Actividad 3.1: Análisis de Componentes Principales (PCA)
Asignatura: Extracción de Conocimiento en Bases de Datos - Unidad IV
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Cargar dataset continuo multivariado (4 características)
iris = load_iris()
X_iris = iris.data

# Reducir de 4 dimensiones a 2 Componentes Principales
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_iris)

varianza_explicada = pca.explained_variance_ratio_
varianza_acumulada = np.sum(varianza_explicada)

print("=== 3.1 ANÁLISIS DE COMPONENTES PRINCIPALES (PCA) ===")
print(f"Dimensión Original: {X_iris.shape} -> Dimensión Reducida: {X_pca.shape}")
print(f"Varianza explicada por Componente 1: {varianza_explicada[0]*100:.2f}%")
print(f"Varianza explicada por Componente 2: {varianza_explicada[1]*100:.2f}%")
print(f"Varianza Acumulada Total (2D): {varianza_acumulada*100:.2f}%")
print("Matriz de Componentes (Loadings/Vectores Propios):\n", np.round(pca.components_, 4))