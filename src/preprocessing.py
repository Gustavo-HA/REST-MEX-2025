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