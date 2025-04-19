import pytest
from src.utils import preprocess_text, extrair_palavras_relevantes

@pytest.mark.parametrize("texto,esperado", [
    ("Python e Machine Learning!!!", "python machine learning"),
    ("123456, Teste. Dados@", "teste dados"),
    ("", "")
])
def test_preprocess_text(texto, esperado):
    resultado = preprocess_text(texto)
    for palavra in esperado.split():
        assert palavra in resultado

@pytest.mark.parametrize("texto,importancias,esperado", [
    ("python sql java", {"python": 0.8, "sql": 0.3, "java": 0.2}, ["python", "sql"]),
    ("", {"python": 0.8}, []),
    ("csharp dotnet", {"python": 0.8}, [])
])
def test_extrair_palavras_relevantes(texto, importancias, esperado):
    resultado = extrair_palavras_relevantes(texto, importancias)
    detectadas = [palavra for palavra, _ in resultado]
    for palavra in esperado:
        assert palavra in detectadas
