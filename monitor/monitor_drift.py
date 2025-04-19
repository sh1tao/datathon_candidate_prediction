import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

# Caminhos dos arquivos
ref_path = "../data/dados_treinamento.csv"
curr_path = "../data/amostra_dados_recentes.csv"

ref = pd.read_csv(ref_path)
curr = pd.read_csv(curr_path)

relatorio = ["# Relatório de Drift Manual\n"]

# Variáveis numéricas
variaveis_numericas = ["remuneracao"]
for col in variaveis_numericas:
    stat, p = ks_2samp(ref[col], curr[col])
    relatorio.append(f"## {col}")
    relatorio.append(f"- KS-Statistic: {stat:.4f}")
    relatorio.append(f"- p-value: {p:.4f}")
    relatorio.append(f"- Drift detectado: {'Sim' if p < 0.05 else 'Não'}\n")

# Variáveis categóricas
variaveis_categoricas = ["titulo", "area", "nivel_academico", "ingles"]
for col in variaveis_categoricas:
    tabela = pd.crosstab(ref[col], ["ref"])
    tabela_curr = pd.crosstab(curr[col], ["curr"])
    tabela_combinada = tabela.join(tabela_curr, how="outer").fillna(0)
    stat, p, _, _ = chi2_contingency(tabela_combinada)
    relatorio.append(f"## {col}")
    relatorio.append(f"- Chi-squared stat: {stat:.4f}")
    relatorio.append(f"- p-value: {p:.4f}")
    relatorio.append(f"- Drift detectado: {'Sim' if p < 0.05 else 'Não'}\n")

# Target
if "contratado" in ref.columns and "contratado" in curr.columns:
    ref_mean = ref["contratado"].mean()
    curr_mean = curr["contratado"].mean()
    relatorio.append("## contratado")
    relatorio.append(f"- Média treinamento: {ref_mean:.2f}")
    relatorio.append(f"- Média atual: {curr_mean:.2f}")
    relatorio.append(f"- Mudança significativa: {'Sim' if abs(ref_mean - curr_mean) > 0.1 else 'Não'}\n")

# Salva como txt
with open("drift_manual_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(relatorio))

print("Relatório salvo em drift_manual_report.txt")
