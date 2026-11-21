"""
Actividad 3.3: Máquinas de Vectores de Soporte (SVM) y Naive Bayes
Asignatura: Extracción de Conocimiento en Bases de Datos - Unidad III
"""
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Generar dataset sintético para clasificación binaria
X_data, y_data = make_classification(
    n_samples=300, n_features=5, n_classes=2, random_state=42
)

# 1. Máquina de Vectores de Soporte con Kernel RBF
svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
svm.fit(X_data, y_data)

# 2. Clasificador Naive Bayes Gaussiano
nb = GaussianNB()
nb.fit(X_data, y_data)

print("=== 3.3 SVM Y NAIVE BAYES ===")
print(f"Precisión SVM (Kernel RBF): {accuracy_score(y_data, svm.predict(X_data))*100:.2f}%")
print(f"Precisión Naive Bayes: {accuracy_score(y_data, nb.predict(X_data))*100:.2f}%")