# Usa uma imagem base do Python
FROM python:3.8

# Define o diretório de trabalho
WORKDIR /app

# Copia o arquivo de requisitos e instala as dependências
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copia o script Python para o diretório de trabalho
COPY . /app/

# Define o comando padrão para rodar o script
CMD ["python", "app/teste.py"]
