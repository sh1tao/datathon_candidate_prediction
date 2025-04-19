import os
import joblib
import pandas as pd
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from scipy.sparse import hstack, csr_matrix
import logging
from prometheus_client import Summary, Counter, REGISTRY

# Remove métricas duplicadas caso existam
for metric_name in ["probabilidade_prevista", "candidatos_contratados_total"]:
    collector = REGISTRY._names_to_collectors.get(metric_name)
    if collector:
        REGISTRY.unregister(collector)

# Cria a instância da aplicação
app = FastAPI(title="API de Predição de Contratação")

# Expondo métricas Prometheus
Instrumentator().instrument(app).expose(app)

# Métricas personalizadas
media_probabilidade = Summary('probabilidade_prevista', 'Probabilidade prevista pelo modelo')
contador_contratados = Counter('candidatos_contratados_total', 'Número de candidatos classificados como contratados')

# Garante que a pasta de logs existe
log_path = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_path, exist_ok=True)

log_file = os.path.join(log_path, "api.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Carrega artefatos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "../model/model.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "../model/tfidf.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "../model/encoder.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "../model/scaler.pkl"))

class Candidate(BaseModel):
    cv: str
    titulo: str
    area: str
    nivel_academico: str
    ingles: str
    remuneracao: str

@app.post("/predict")
def predict(candidate: Candidate):
    data = pd.DataFrame([candidate.dict()])
    data['remuneracao'] = (
        data['remuneracao']
        .str.replace('R$', '', regex=False)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
    )
    data['remuneracao'] = pd.to_numeric(data['remuneracao'], errors='coerce').fillna(0)

    X_text = tfidf.transform(data['cv'])
    X_cat = encoder.transform(data[['titulo', 'area', 'nivel_academico', 'ingles']].fillna("Desconhecido"))
    X_num = scaler.transform(data[['remuneracao']])
    X_all = hstack([X_text, X_cat, csr_matrix(X_num)]).toarray()

    tokens = candidate.cv.lower().split()
    feature_names = tfidf.get_feature_names_out()
    importances = model.feature_importances_[:len(feature_names)]
    palavras = {palavra: imp for palavra, imp in zip(feature_names, importances) if palavra in tokens}
    palavras_ranqueadas = sorted(palavras.items(), key=lambda x: x[1], reverse=True)[:10]

    pred = model.predict(X_all)[0]
    prob = model.predict_proba(X_all)[0][1]

    # Logging estruturado
    logger.info({
        "titulo": candidate.titulo,
        "area": candidate.area,
        "nivel_academico": candidate.nivel_academico,
        "ingles": candidate.ingles,
        "remuneracao": candidate.remuneracao,
        "probabilidade": round(float(prob), 4),
        "contratado": bool(pred)
    })

    # Métricas personalizadas
    media_probabilidade.observe(prob)
    if pred == 1:
        contador_contratados.inc()

    return {
        "contratado": bool(pred),
        "probabilidade": round(float(prob), 4),
        "palavras_detectadas_ranqueadas": palavras_ranqueadas
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("predict_api:app", host="0.0.0.0", port=8000)
