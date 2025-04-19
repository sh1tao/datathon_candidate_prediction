from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from fastapi.responses import HTMLResponse

import numpy as np

# Carrega artefatos
model = joblib.load("../model/model.pkl")
tfidf = joblib.load("../model/tfidf.pkl")
encoder = joblib.load("../model/encoder.pkl")
scaler = joblib.load("../model/scaler.pkl")

app = FastAPI(title="API de Predição de Contratação")


class Candidate(BaseModel):
    cv: str
    titulo: str
    area: str
    nivel_academico: str
    ingles: str
    remuneracao: str  # Ex: "R$ 4.500,00"


@app.post("/predict")
def predict(candidate: Candidate):
    # Prepara entrada
    data = pd.DataFrame([candidate.dict()])
    data['remuneracao'] = data['remuneracao'].str.replace('R$', '', regex=False).str.replace('.', '',
                                                                                             regex=False).str.replace(
        ',', '.', regex=False)
    data['remuneracao'] = pd.to_numeric(data['remuneracao'], errors='coerce').fillna(0)

    X_text = tfidf.transform(data['cv'])
    X_cat = encoder.transform(data[['titulo', 'area', 'nivel_academico', 'ingles']].fillna("Desconhecido"))
    X_num = scaler.transform(data[['remuneracao']])
    X_all_sparse = hstack([X_text, X_cat, csr_matrix(X_num)])
    X_all = X_all_sparse.toarray()

    # Palavras detectadas no CV com importância
    tokens = candidate.cv.lower().split()
    feature_names = tfidf.get_feature_names_out()
    importances = model.feature_importances_[:len(feature_names)]
    palavra_importancia = {
        palavra: importancias for palavra, importancias in zip(feature_names, importances) if palavra in tokens
    }
    palavras_ranqueadas = sorted(palavra_importancia.items(), key=lambda x: x[1], reverse=True)[:10]  # top 10

    # Predição
    pred = model.predict(X_all)[0]
    prob = model.predict_proba(X_all)[0][1]

    return {
        "contratado": bool(pred),
        "probabilidade": round(float(prob), 4),
        "palavras_detectadas_ranqueadas": palavras_ranqueadas
    }


@app.get("/", response_class=HTMLResponse)
def serve_form():
    with open("../templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("predict_api:app", host="0.0.0.0", port=8000, reload=True)
