FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN apt-get update && apt-get install -y \
    default-jdk-headless \
    ffmpeg \
    mecab \
    libmecab-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh | bash
ENV MECABRC=/etc/mecabrc
ENV MECAB_DIC_PATH=/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ko-dic

WORKDIR /
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
