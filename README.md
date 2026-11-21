# Extracción de Conocimiento en Bases de Datos

Repositorio de prácticas de Ciencia de Datos y Machine Learning desarrolladas e implementadas en una instancia **AWS EC2 (Ubuntu 22.04 LTS Server)** mediante **VS Code (Remote - SSH)**.

---

## Estructura del Proyecto

### Unidad III: Análisis Supervisado (`unidad3/`)
1. `01_regresion_lineal.py` — Regresión Lineal Simple (Tema 2.1).
2. `02_regularizacion_ridge_lasso.py` — Regresión con Regularización L1 y L2: Ridge y Lasso (Tema 2.2).
3. `03_random_forest_regressor.py` — Bosques Aleatorios para Regresión (Tema 2.3).
4. `04_regresion_logistica.py` — Regresión Logística Binaria (Tema 3.1).
5. `05_arboles_random_forest.py` — Árboles de Decisión vs. Random Forest Classifier (Tema 3.2).
6. `06_svm_naive_bayes.py` — Máquinas de Vectores de Soporte (Kernel RBF) y Naive Bayes Gaussiano (Tema 3.3).
7. `07_practica_integradora_fraudes.py` — Pipeline completo de detección de fraudes con escalado `StandardScaler`, validación cruzada (5-Fold), optimización con `GridSearchCV` y serialización con `joblib` (`modelo_fraude_opt.pkl` y `scaler_fraude.pkl`) (Tema 4.2).

### Unidad IV: Análisis No Supervisado (`unidad4/`)
1. `01_kmeans_clustering.py` — Agrupación Particional K-Means e inspección de centroides y silueta (Tema 2.1).
2. `02_agrupacion_jerarquica.py` — Agrupación Jerárquica Aglomerativa con enlace Ward (Tema 2.2).
3. `03_dbscan_clustering.py` — Agrupación Basada en Densidad (DBSCAN) y detección de ruido/outliers (Tema 2.3).
4. `04_pca_reduccion.py` — Análisis de Componentes Principales (PCA) y varianza explicada (Tema 3.1).
5. `05_tsne_reduccion.py` — Reducción Dimensional No Lineal mediante t-SNE (Tema 3.2).
6. `06_practica_integradora_segmentacion.py` — Pipeline completo de segmentación multivariada de clientes con selección óptima de *K* (Silueta, Davies-Bouldin, Calinski-Harabasz), compresión 2D con PCA y serialización (`modelo_kmeans_clientes.pkl`, `scaler_clientes.pkl` y `pca_clientes.pkl`) (Tema 4.2).

---

## Despliegue y Ejecución en Ubuntu EC2

* **Activar entorno virtual:** `source unidad3/venv/bin/activate`
* **Instalar dependencias:** `pip install -r requirements.txt`
* **Ejecutar prácticas Integradoras:**
  * Unidad III: `python3 unidad3/07_practica_integradora_fraudes.py`
  * Unidad IV: `python3 unidad4/06_practica_integradora_segmentacion.py`

---

## Referencias Bibliográficas (APA 7ª Edición)
* Hapke, H., & Nelson, C. (2020). *Building Machine Learning Pipelines: Automating Model Life Cycles with TensorFlow and Scikit-Learn*. O'Reilly Media.
* Joyanes Aguilar, L. (2016). *BIG DATA: Análisis de grandes volúmenes de datos en organizaciones*. Alfaomega.
* Pérez Marqués, M. (2015). *Minería de datos a través de ejemplos*. Alfaomega / RC Libros.
* Raschka, S., & Mirjalili, V. (2019). *Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow 2* (3rd ed.). Marcombo / Packt Publishing.
* Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
* Witten, I. H., Frank, E., Hall, M. A., & Pal, C. J. (2016). *Data Mining: Practical Machine Learning Tools and Techniques* (4th ed.). Morgan Kaufmann.
* Zaki, M. J., & Meira, W., Jr. (2019). *Data Mining and Machine Learning: Fundamental Concepts and Algorithms*. Cambridge University Press / Bravex Publications.
