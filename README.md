﻿# 📊 Projeto de Predição de Contratação com Monitoramento

Este projeto utiliza aprendizado de máquina para prever se um(a) candidato(a) será contratado, com API de inferência em FastAPI, monitoramento com Prometheus e Grafana, e deploy via Docker.

---

## 🧠 Funcionalidades

- Treinamento de modelo com dados históricos (`RandomForestClassifier`)
- Extração de features textuais via `TfidfVectorizer`
- API REST para predição usando FastAPI
- Monitoramento com Prometheus + Grafana
- Exportação de métricas personalizadas
- Deploy com Docker e Docker Compose

---
## Produçao em Nuvem

### [https://datathon-candidate-prediction.onrender.com/docs](https://datathon-candidate-prediction.onrender.com/docs)

## Como rodar localmente

### 1. Clone o projeto
```bash
git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo
```

### 2. Crie um ambiente virtual
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Rode a API localmente
```bash
uvicorn src.predict_api:app --reload
```
Acesse em [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Como rodar com Docker Compose
```bash
docker compose up --build
```

### Serviços disponíveis:
- API: [http://localhost:8000/docs](http://localhost:8000/docs)
- Prometheus: [http://localhost:9090](http://localhost:9090)
- Grafana: [http://localhost:3000](http://localhost:3000)

**Login Grafana padrão:**
- user: `admin`
- senha: `admin`

---

## 📈 Diagrama

![image](https://github.com/user-attachments/assets/37d8838d-295a-4999-b0f0-143921130aa1)



## 📈 Métricas customizadas expostas no `/metrics`

- `probabilidade_prevista` – Summary com a probabilidade prevista nas predições
- `candidatos_contratados_total` – Contador de candidatos classificados como contratados

Estas métricas podem ser visualizadas e graficadas no painel Grafana incluso (`grafana_dashboard_updated.json`).

---

## 🧪 Testes

```bash
pytest tests/
```
Inclui testes unitários para pré-processamento e API.

---

## ☁️ Deploy na nuvem (Render)

- Serviço da API é publicado com Dockerfile
- Observação: o Render não suporta múltiplos containers, então Prometheus e Grafana devem ser rodados localmente ou via VPS.

---

## 📁 Estrutura de Diretórios
```
📁 datathon_candidate_predict/
├── 📁 model/                            # Artefatos do modelo treinado (.pkl)
│   ├── model.pkl
│   ├── tfidf.pkl
│   ├── encoder.pkl
│   └── scaler.pkl
│
├── 📁 logs/                             # Arquivos de log
│   └── api.log
│
├── 📁 src/                              # Código principal
│   ├── predict_api.py
│   ├── preprocess.py
│   ├── train_model.py
│   └── utils.py
│
├── 📁 monitor/                       # Configurações Prometheus
│   ├── prometheus.yml
│   └── monitor_drift.py
│
├── 📁 tests/                            # Testes unitários
│   ├── test_api.py
│   └── test_utils.py
│
├── 📁 data/                             # Dados simulados de produção/treinamento
│   ├── dados_treinamento.csv
│   └── amostra_dados_recentes.csv
│
├── grafana_dashboard_updated.json      # Painel de monitoramento Grafana
├── docker-compose.yml                  # Orquestrador Docker
├── Dockerfile                          # Imagem da API
├── README.md                           # Documentação do projeto
└── requirements.txt                    # Dependências do projeto
```

---

## ✨ Futuras melhorias

- Autenticação JWT para proteger a API
- Upload em lote (CSV)
- Re-treinamento contínuo com agendamento
- Deploy completo com CI/CD + HTTPS

---

Feito com 💻 por Washington Santos
