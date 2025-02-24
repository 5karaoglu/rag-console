# Excel RAG Sistemi

Bu proje, Excel dosyalarındaki verileri kullanarak soruları yanıtlayan bir RAG (Retrieval Augmented Generation) sistemidir. DeepSeek-R1-Distill-Qwen-32B modelini kullanarak doğal dil yanıtları üretir.

## Özellikler

- Excel dosyasından veri okuma
- ChromaDB ile vektör veritabanı desteği
- DeepSeek-R1-Distill-Qwen-32B LLM modeli
- GPU desteği ile hızlı yanıt üretimi
- Docker container desteği
- Kalıcı cache sistemi

## Gereksinimler

- NVIDIA GPU (en az 24GB RAM)
- Docker ve Docker Compose
- NVIDIA Container Toolkit

## Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/kullanici/rag-console.git
cd rag-console
```

2. Cache dizinlerini oluşturun:
```bash
chmod +x setup.sh
./setup.sh
```

3. Container'ı başlatın:
```bash
docker-compose up --build
```

## Excel Dosyası Kullanımı

Varsayılan olarak `Book1.xlsx` dosyası kullanılır. Farklı bir Excel dosyası kullanmak için:

1. `.env` dosyasını düzenleyin:
```bash
EXCEL_PATH=/tam/yol/baska_dosya.xlsx
```

2. Veya komut satırından belirtin:
```bash
EXCEL_PATH=/tam/yol/baska_dosya.xlsx docker-compose up
```

## Cache Sistemi

Sistem aşağıdaki verileri cache'ler:
- Hugging Face modelleri (`./data/cache/huggingface`)
- PyTorch modelleri (`./data/cache/torch`)
- ChromaDB veritabanı (`./data/chroma_db`)

Cache'i temizlemek için:
```bash
rm -rf data/cache/*
```

## Sunucuda Çalıştırma

Detaylı sunucu kurulum adımları için [INSTALL.md](INSTALL.md) dosyasına bakın.

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın. 