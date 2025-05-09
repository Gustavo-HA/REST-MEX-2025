import pandas as pd
import numpy as np
try:
    from nltk.corpus import stopwords
except:
    import nltk
    nltk.download("stopwords")
    from nltk.corpus import stopwords
import re
from functools import lru_cache
import config
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import SnowballStemmer
from sklearn.pipeline import Pipeline

class ArreglaMojibake(BaseEstimator, TransformerMixin):
    """
    Transformer de scikit-learn para arreglar mojibake en textos.
    """
    def __init__(self, variables):
        self.variables = variables
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Aplica la transformación para arreglar mojibake en una lista o serie de textos.
        """
        return X[self.variables].apply(self._arregla_mojibake)

    @staticmethod
    def _arregla_mojibake(texto: str) -> str:
        """
        Convierte un texto con mojibake (texto mal codificado) a texto legible.
        Si la conversión no es posible, se devuelve el texto original.
        """
        try:
            return texto.encode('latin1').decode('utf-8')
        except:
            return texto



@lru_cache(maxsize=None)
def get_stopwords(lang='spanish'):
    return set(stopwords.words(lang))

class QuitaStopwords(BaseEstimator, TransformerMixin):
    """
    Transformer de scikit-learn para quitar stopwords en textos.
    """
    def __init__(self, variables, lang='spanish'):
        self.lang = lang
        self.stopwords = get_stopwords(lang)
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X : pd.DataFrame):
        """
        Aplica la transformación para quitar stopwords en una lista o serie de textos.
        """
        X = X.copy()
        for variable in self.variables:
            X[variable] = X[variable].apply(self._quita_stopwords)
        return X

    def _quita_stopwords(self, texto: str) -> str:
        """
        Elimina las stopwords de un texto.
        """
        if not isinstance(texto, str):
            return " "
        
        texto = texto.lower()
        texto = re.sub(r'[^a-záéíóúñü\s]', '', texto)
        palabras = texto.split()
        palabras_filtradas = [palabra for palabra in palabras if palabra not in self.stopwords]
        return ' '.join(palabras_filtradas)
    
    
class SpanishStemmer(BaseEstimator, TransformerMixin):
    """
    Transformer de scikit-learn para aplicar stemming en textos en español.
    """
    def __init__(self, variables, lang='spanish'):
        self.variables = variables
        self.stemmer = SnowballStemmer(lang)

    def fit(self, X, y=None):
        return self

    def transform(self, X : pd.DataFrame):
        """
        Aplica la transformación para aplicar stemming en una lista o serie de textos.
        """
        X = X.copy()
        for variable in self.variables:
            X[variable] = X[variable].apply(self._stem)
        return X

    def _stem(self, texto: str) -> str:
        """
        Aplica el stemming a un texto.
        """
        if not isinstance(texto, str):
            return " "
        
        palabras = texto.split()
        palabras_stemmed = [self.stemmer.stem(palabra) for palabra in palabras]
        return ' '.join(palabras_stemmed)
    
    
class DropFeatures(BaseEstimator, TransformerMixin):
    """Dropping Features Which Are Less Significant"""

    def __init__(self, variables_to_drop=None):
        self.variables_to_drop = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(self.variables_to_drop, axis=1)
        return X
    
    
class JuntarFeatures(BaseEstimator, TransformerMixin):
    """Juntar Features de texto significativas en una sola columna"""
    def __init__(self, variables_to_join=None, new_column=None):
        self.variables_to_join = variables_to_join
        self.new_column = new_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X : pd.DataFrame):
        X = X.copy()
        X[self.new_column] = X[self.variables_to_join].agg(" ".join, axis = 1)
        X[self.new_column] = X[self.new_column].str.strip()
        return X
    

# Pipeline para el dataset de RESTMEX.
pp_pipeline = Pipeline(
    [
        ('Arreglar mojibakes', ArreglaMojibake(config.TEXT_COLUMNS)),
        ("Minúsculas y quitar stopwords", QuitaStopwords(config.TEXT_COLUMNS)),
        ("Stemming", SpanishStemmer(config.TEXT_COLUMNS)),
        ("Guardar en una columna", JuntarFeatures(config.TEXT_COLUMNS, config.NEW_COLUMN)),
        ("Quitar features no deseadas", DropFeatures(config.TEXT_COLUMNS))
    ]
)


    

if __name__ == "__main__":
    import io
    
    csv_data = """Title,Review,Polarity,Town,Region,Type
Mi Lugar Favorito!!!!,"Excelente lugar para comer y pasar una buena noche!!! El servicio es de primera y la comida exquisita!!!",5.0,Sayulita,Nayarit,Restaurant
lugares interesantes para visitar,"andar mucho, así que un poco difícil para personas con niños pequeños, pero con mucha historia en la zona, y la diversión de aprender un poco de todo, y explorar las ruinas. La playa también era bastante agradable!",4.0,Tulum,QuintanaRoo,Attractive"""
    df = pd.read_csv(io.StringIO((csv_data)))
    
    # Test the transformers
    
    df = pp_pipeline.fit_transform(df)
    print(df.head())