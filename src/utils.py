import re
import unidecode
from nltk.corpus import stopwords


def preprocess_text(texto):
    texto = unidecode.unidecode(texto.lower())
    texto = re.sub(r"[^a-zA-Z\s]", "", texto)
    palavras = texto.split()
    stop_words = set(stopwords.words("portuguese"))
    palavras = [p for p in palavras if p not in stop_words]
    return " ".join(palavras)


def extrair_palavras_relevantes(texto, importancias, top_n=10):
    palavras = texto.split()
    palavras_relevantes = [(p, importancias[p]) for p in palavras if p in importancias]
    palavras_relevantes.sort(key=lambda x: x[1], reverse=True)
    return palavras_relevantes[:top_n]
