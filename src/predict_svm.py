import joblib
import os
import unicodedata
import re
import nltk
from nltk.tokenize import word_tokenize

# Descargar recursos si no se han descargado
nltk.download('punkt')

# --- Función de limpieza (reutilizamos lo que hicimos) ---
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

# --- Función para predecir polaridad de nuevas reseñas ---
def predict_polarity(reviews):
    # Limpieza de texto
    cleaned_reviews = [tokenize_clean_text(review) for review in reviews]
    
    # Vectorización
    X_reviews = tfidf_vectorizer.transform(cleaned_reviews)
    
    # Predicción
    preds = svm_model.predict(X_reviews)
    
    return preds

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Reseñas de prueba
    new_reviews = [
        "El servicio fue excelente, la comida deliciosa y la atención inmejorable!",
        "El hotel estaba sucio y el personal fue muy grosero.",
        "Nada especial, una experiencia normalita.",
    ]
    
    predictions = predict_polarity(new_reviews)
    
    for review, polarity in zip(new_reviews, predictions):
        print(f"\nReseña: {review}\n→ Predicción de polaridad: {int(polarity)}")
