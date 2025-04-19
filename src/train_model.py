import pandas as pd
import joblib
import nltk
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack, csr_matrix

from preprocess import preprocess_merge

# Baixa stopwords se ainda não tiver
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words_pt = stopwords.words('portuguese')

tqdm.pandas()


def train_pipeline():
    print("Iniciando pipeline de treinamento...")

    # 1. Carregamento e pré-processamento
    df = preprocess_merge('../data/applicants.json', '../data/prospects.json')
    print(f"Total de candidatos processados: {len(df)}")
    df.dropna(subset=["cv"], inplace=True)
    print(f"Após remoção de CVs vazios: {len(df)} candidatos")

    X_text = df["cv"]
    y = df["target"]

    # 2. Preparar features adicionais
    print("Preparando features adicionais...")
    categorical_cols = ['titulo', 'area', 'nivel_academico', 'ingles']
    numerical_cols = ['remuneracao']

    df[categorical_cols] = df[categorical_cols].fillna("Desconhecido")
    df[numerical_cols] = df[numerical_cols].fillna("0")

    # Converte 'remuneracao' em número
    df['remuneracao'] = df['remuneracao'].str.replace('R$', '', regex=False).str.replace('.', '',
                                                                                         regex=False).str.replace(',',
                                                                                                                  '.',
                                                                                                                  regex=False)
    df['remuneracao'] = pd.to_numeric(df['remuneracao'], errors='coerce').fillna(0)

    encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    scaler = StandardScaler()

    X_cat = encoder.fit_transform(df[categorical_cols])
    X_num = scaler.fit_transform(df[numerical_cols])

    # 3. Vetorização TF-IDF
    print("Aplicando TF-IDF...")
    tfidf = TfidfVectorizer(max_features=1000, stop_words=stop_words_pt)
    X_tfidf = tfidf.fit_transform(X_text)

    # 4. Junta tudo
    print("Concatenando todas as features...")
    X_all = hstack([X_tfidf, X_cat, csr_matrix(X_num)])

    # 5. SMOTE - balanceamento
    print("Aplicando SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_all, y)
    print(f"Após SMOTE: Classe 0 = {sum(y_resampled == 0)} | Classe 1 = {sum(y_resampled == 1)}")

    # 6. Split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled,
                                                        random_state=42)

    # 7. Treinamento
    print("Treinando modelo...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    with tqdm(total=1, desc="Treinamento do Modelo") as pbar:
        clf.fit(X_train, y_train)
        pbar.update(1)

    # 8. Avaliação
    print("\nAvaliação do modelo:\n")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 9. Importância das palavras (apenas do TF-IDF)
    print("Calculando importância das palavras...")
    feature_names = tfidf.get_feature_names_out()
    importances = clf.feature_importances_[:len(feature_names)]

    feature_df = pd.DataFrame({
        'palavra': feature_names,
        'importancia': importances
    }).sort_values(by='importancia', ascending=False)

    feature_df.to_csv('../data/feature_importance.csv', index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_df['palavra'].head(20), feature_df['importancia'].head(20))
    plt.xlabel("Importância")
    plt.ylabel("Palavra")
    plt.title("Top 20 Palavras mais Relevantes")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('../data/feature_importance_plot.png')
    plt.close()

    # 10. Salvamento dos artefatos
    print("Salvando artefatos...")
    joblib.dump(clf, '../model/model.pkl')
    joblib.dump(tfidf, '../model/tfidf.pkl')
    joblib.dump(encoder, '../model/encoder.pkl')
    joblib.dump(scaler, '../model/scaler.pkl')
    print("Pipeline finalizada com sucesso!")


if __name__ == "__main__":
    train_pipeline()
