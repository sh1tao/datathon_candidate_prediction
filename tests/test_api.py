from fastapi.testclient import TestClient
from src.predict_api import app

client = TestClient(app)

def test_predict_valido():
    payload = {
        "cv": "Desenvolvedor com Python, REST e aprendizado de máquina",
        "titulo": "Dev",
        "area": "TI",
        "nivel_academico": "Superior",
        "ingles": "Avançado",
        "remuneracao": "R$ 12.000,00"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "contratado" in json_data
    assert "probabilidade" in json_data
    assert isinstance(json_data["probabilidade"], float)
