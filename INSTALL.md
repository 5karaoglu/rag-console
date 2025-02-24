# Sunucu Kurulum Talimatları

Bu dokümanda, RAG sisteminin sunucuda nasıl kurulacağı ve çalıştırılacağı anlatılmaktadır.

## 1. Sistem Gereksinimleri

- Ubuntu 22.04 LTS (önerilen)
- En az 24GB RAM'li NVIDIA GPU
- En az 100GB disk alanı
- Sudo yetkisi

## 2. Kurulum Adımları

### 2.1. Temel Sistem Güncellemesi

```bash
# Sistem güncellemesi
sudo apt-get update
sudo apt-get upgrade -y
```

### 2.2. Docker Kurulumu

```bash
# Docker kurulumu
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Docker Compose kurulumu
sudo apt-get update
sudo apt-get install -y docker-compose-plugin

# Kullanıcıyı docker grubuna ekleyin
sudo usermod -aG docker $USER

# Değişikliklerin etkili olması için oturumu yeniden başlatın
exit
```

### 2.3. NVIDIA Driver ve Container Toolkit Kurulumu

```bash
# NVIDIA Driver kurulumu
sudo apt-get update
sudo apt-get install -y nvidia-driver-535  # veya sunucunuz için uygun sürüm

# NVIDIA Container Toolkit kurulumu
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# NVIDIA Container Runtime'ı yapılandırın
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 3. Uygulama Kurulumu

### 3.1. Projeyi Klonlama

```bash
# Projeyi klonlayın
git clone https://github.com/kullanici/rag-console.git
cd rag-console

# Cache dizinlerini oluşturun
chmod +x setup.sh
./setup.sh
```

### 3.2. Uygulamayı Başlatma

```bash
# Container'ı oluşturun ve başlatın
docker-compose up --build -d

# Logları kontrol edin
docker-compose logs -f
```

## 4. Sistem Kontrolü

### 4.1. Durum Kontrolü

```bash
# GPU durumunu kontrol edin
nvidia-smi

# Container durumunu kontrol edin
docker ps
docker-compose ps
```

### 4.2. Log Kontrolü

```bash
# Container loglarını görüntüleyin
docker-compose logs -f rag-app
```

## 5. Otomatik Başlatma (İsteğe Bağlı)

Sistemin sunucu yeniden başlatıldığında otomatik başlaması için:

```bash
# Sistem servisi oluşturun
sudo nano /etc/systemd/system/rag-app.service
```

Service dosyası içeriği:
```ini
[Unit]
Description=RAG Application
After=docker.service
Requires=docker.service

[Service]
WorkingDirectory=/home/kullanici/rag-console
ExecStart=/usr/bin/docker-compose up
ExecStop=/usr/bin/docker-compose down
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Servisi etkinleştirin
sudo systemctl enable rag-app
sudo systemctl start rag-app
```

## 6. Sorun Giderme

### 6.1. GPU Sorunları
- `nvidia-smi` komutunun çalıştığından emin olun
- NVIDIA driver'ın doğru yüklendiğini kontrol edin
- Container'ın GPU'ya erişebildiğini kontrol edin

### 6.2. Docker Sorunları
- Docker servisinin çalıştığını kontrol edin: `sudo systemctl status docker`
- Docker log'larını kontrol edin: `docker-compose logs`
- Container'ı yeniden başlatın: `docker-compose restart`

### 6.3. Disk Alanı Kontrolü
```bash
# Disk kullanımını kontrol edin
df -h

# Cache boyutunu kontrol edin
du -sh data/
``` 