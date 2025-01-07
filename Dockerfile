# Usa a imagem oficial Python como base
FROM python:3.9-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Definir o diretório de trabalho
WORKDIR /app

# Copiar arquivos de dependências
COPY requirements.txt .

# Instalar dependências principais
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código do projeto
COPY . .

# Configurar o PYTHONPATH para permitir imports absolutos
ENV PYTHONPATH=/app

# Expor a porta necessária
EXPOSE 8501

# Definir comando padrão (ajuste para o arquivo correto)
CMD ["streamlit", "run", "src/app/chatbot.py", "--server.port=8501", "--server.address=0.0.0.0"]