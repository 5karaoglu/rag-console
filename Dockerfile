FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Locale ayarlarını düzelt
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# HTTP istekleri için kullanıcı ajanını ayarla
ENV TRANSFORMERS_USER_AGENT=Mozilla/5.0
ENV HF_USER_AGENT=Mozilla/5.0

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizinini oluştur
WORKDIR /app

# Cache dizinlerini oluştur
RUN mkdir -p /app/cache/huggingface
RUN mkdir -p /app/cache/torch
RUN mkdir -p /app/chroma_db

# Cache için ortam değişkenlerini ayarla
ENV TRANSFORMERS_CACHE=/app/cache/huggingface
ENV TORCH_HOME=/app/cache/torch
ENV HF_HOME=/app/cache/huggingface

# Python bağımlılıklarını kopyala ve yükle
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Uygulama kodlarını kopyala
COPY rag_app.py .

# JSON dosyasını kopyala (varsayılan dosya)
COPY Book1.json .

# Çalıştırma komutu
ENTRYPOINT ["python3", "rag_app.py"] 