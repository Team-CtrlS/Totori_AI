FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    default-jdk-headless \
    ffmpeg \
    build-essential \
    curl \
    git \
    mecab \
    libmecab-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh | bash

WORKDIR /
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV MECAB_DIC_PATH=/usr/local/lib/mecab/dic/mecab-ko-dic
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
