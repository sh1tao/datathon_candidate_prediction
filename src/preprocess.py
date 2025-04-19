import json
import pandas as pd

def load_applicants(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for codigo, dados in data.items():
        row = {
            "codigo": codigo,
            "nome": dados.get("informacoes_pessoais", {}).get("nome"),
            "email": dados.get("informacoes_pessoais", {}).get("email"),
            "cv": dados.get("cv_pt", ""),
            "titulo": dados.get("informacoes_profissionais", {}).get("titulo_profissional", ""),
            "area": dados.get("informacoes_profissionais", {}).get("area_atuacao", ""),
            "nivel_academico": dados.get("formacao_e_idiomas", {}).get("nivel_academico", ""),
            "ingles": dados.get("formacao_e_idiomas", {}).get("nivel_ingles", ""),
            "remuneracao": dados.get("informacoes_profissionais", {}).get("remuneracao", "")
        }
        rows.append(row)
    return pd.DataFrame(rows)

def load_prospects(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for vaga_id, vaga in data.items():
        for prospect in vaga.get("prospects", []):
            rows.append({
                "codigo": prospect.get("codigo"),
                "nome": prospect.get("nome"),
                "vaga_id": vaga_id,
                "situacao": prospect.get("situacao_candidado")
            })
    return pd.DataFrame(rows)

def classificar_alvo(status):
    if isinstance(status, str):
        status_lower = status.lower().strip()
        if "não aprovado" in status_lower:
            return 0
        if "contratado" in status_lower or status_lower == "aprovado":
            return 1
    return 0

def preprocess_merge(applicants_path, prospects_path):
    df_app = load_applicants(applicants_path)
    df_pros = load_prospects(prospects_path)
    df = pd.merge(df_app, df_pros, on="codigo", how="inner")

    df["target"] = df["situacao"].apply(classificar_alvo)
    return df

def save_preprocessed_csv(output_path="../data/preprocessed_applicants.csv"):
    df = preprocess_merge('../data/applicants.json', '../data/prospects.json')
    df.to_csv(output_path, index=False)
    print(f"📁 CSV salvo em: {output_path}")

    # Estatísticas de distribuição
    contagem = df["target"].value_counts().to_dict()
    print("\n📊 Distribuição dos targets:")
    print(f"  ➤ Não contratados (0): {contagem.get(0, 0)}")
    print(f"  ➤ Contratados ou Aprovados (1): {contagem.get(1, 0)}")

# Execução direta
if __name__ == "__main__":
    save_preprocessed_csv()
