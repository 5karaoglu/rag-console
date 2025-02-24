FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
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

# Qwen2 modelini klonla
RUN git clone https://huggingface.co/Qwen/Qwen2 /app/Qwen2

# Uygulama kodlarını kopyala
COPY rag_app.py .

# Excel dosyasını kopyala (varsayılan dosya)
COPY Book1.xlsx .

# Çalıştırma komutu
ENTRYPOINT ["python3", "rag_app.py"] 