import re
import contractions
import nltk
from nltk.corpus import stopwords
import re
import string
import spacy
import contractions
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Descargar stopwords de nltk
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")

# Preprocesado utilizado para nuestro texto
def preprocess_text(text):
    text = text.lower() # Convertir a minúsculas para uniformidad
    text = re.sub(r"\[.*?\]", "", text) # Quitar todo contenido entre corchetes
    text = re.sub(f"[{string.punctuation}]", "", text) # Quitar cualquier carácter de puntuación
    text = re.sub(r"[‘’“”\"\']", "", text) # Eliminar comillas raras
    text = re.sub(r"\w*\d\w*", "", text) # Quitar palabras con números
    text = re.sub(r'https://\S+|www\.\S+', '', text) # Quitar URLs
    text = contractions.fix(text) # Expandir contracciones
    text = re.sub(r"\s.\s", " ", text) # Eliminar caracteres que miden solo 1 (espacios a ambos lados)
    text = re.sub(r"\s+", " ", text).strip() # Quitar espacios en blanco adicionales
    text = " ".join([word for word in text.split() if word not in stop_words]) # Quitar stopwords
    text = " ".join([token.lemma_ for token in nlp(text)]) # Lematización
    text = text.str.replace(r"\b(reuter|reuters)\b", "", regex=True) # Eliminamos la palabra reuters si aparece
    return text