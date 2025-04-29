import joblib
import os
import pandas as pd
import unicodedata
import re
import nltk
from nltk.tokenize import word_tokenize

# Descargar recursos de nltk si no están
nltk.download('punkt')

# --- Función de limpieza (reutilizamos) ---
def tokenize_clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha() or token in ['!', '?']]
    text = ' '.join(tokens)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Cargar modelo y vectorizador ---
model_path = os.path.join('..', 'model', 'svm_polarity_model.pkl')
vectorizer_path = os.path.join('..', 'model', 'tfidf_vectorizer.pkl')

svm_model = joblib.load(model_path)
tfidf_vectorizer = joblib.load(vectorizer_path)

print("Modelo y vectorizador cargados exitosamente.")

# --- Función para hacer predicción por lote ---
def predict_batch(input_csv_path, output_csv_path, review_column='Review'):
    # Leer CSV
    df = pd.read_csv(input_csv_path)
    
    if review_column not in df.columns:
        raise ValueError(f"La columna '{review_column}' no existe en el archivo CSV.")
    
    # Limpieza de reseñas
    cleaned_reviews = df[review_column].apply(tokenize_clean_text)
    
    # Vectorización
    X_reviews = tfidf_vectorizer.transform(cleaned_reviews)
    
    # Predicción
    predictions = svm_model.predict(X_reviews)
    
    # Agregar predicciones al dataframe
    df['Predicted_Polarity'] = predictions
    
    # Guardar nuevo CSV
    df.to_csv(output_csv_path, index=False)
    
    print(f"Predicciones guardadas exitosamente en {output_csv_path}")

# --- Ejemplo de uso ---
if __name__ == "__main__":
    input_csv = os.path.join('..', 'data', 'test', 'test_reviews.csv')  # <-- Modifica esto si tu test está en otra carpeta
    output_csv = os.path.join('..', 'data', 'test', 'test_predictions.csv')
    
    predict_batch(input_csv, output_csv)
