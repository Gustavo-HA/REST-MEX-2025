# 📂 src/ - Código fuente del proyecto Rest-Mex

Esta carpeta contiene los scripts de inferencia utilizados en el proyecto **Rest-Mex 2025**.  
Aquí se incluyen herramientas para predecir automáticamente la polaridad de reseñas turísticas, basadas en el modelo SVM entrenado.

---

## 📜 Scripts incluidos

### `predict_svm.py`

- **Descripción**: 
  Carga el modelo de clasificación `LinearSVM` entrenado y predice la polaridad de **una o varias reseñas manualmente ingresadas**.
- **Uso**:
  - Ejecutar el script directamente.
  - Modificar las reseñas de prueba en la sección `new_reviews`.
  - Se imprimen las predicciones en consola.

- **Ejemplo rápido**:

```bash
python predict_svm.py


---

## 📜 Descripción de `predict_batch.py`

```markdown
### `predict_batch.py`

- **Descripción**:  
  Procesa un archivo `.csv` completo de reseñas, predice la polaridad de cada reseña utilizando el modelo SVM entrenado, y guarda las predicciones en un nuevo archivo `.csv`.

- **Uso**:
  - Definir la ruta del CSV de entrada (`input_csv`) y la ruta del CSV de salida (`output_csv`) dentro del script.
  - Asegurarse de que el CSV de entrada contenga una columna llamada `"Review"`.
  - El script limpia, vectoriza y predice automáticamente para todas las reseñas.

- **Ejemplo rápido de ejecución**:

```bash
python src/predict_batch.py
