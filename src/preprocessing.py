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


def arregla_mojibake(texto: str) -> str:
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


def quita_stopwords(texto: str) -> str:
    """
    Elimina las stopwords de un texto.
    """
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñü\s]', '', texto)
    palabras = texto.split()
    stopwords_spanish = get_stopwords()
    palabras_filtradas = [palabra
                          for palabra in palabras
                          if palabra not in stopwords_spanish]
    return ' '.join(palabras_filtradas)
