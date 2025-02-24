#!/bin/bash

# Gerekli dizinleri oluştur
mkdir -p data/chroma_db
mkdir -p data/cache/huggingface
mkdir -p data/cache/torch

# İzinleri ayarla
chmod -R 777 data/

echo "✅ Cache dizinleri oluşturuldu!" 