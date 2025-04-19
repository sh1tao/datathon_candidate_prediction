# Usa imagem leve com Python
FROM python:3.10-slim

# Cria diretório de trabalho dentro do container
WORKDIR /app

# Copia as dependências
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da API da pasta src
COPY src/ ./src

# Copia o diretório do modelo (separado)
COPY model/ ./model

# Define o diretório de execução
WORKDIR /app/src

# Expõe a porta padrão da API
EXPOSE 8000

# Comando para iniciar a API
CMD ["python", "predict_api.py"]
