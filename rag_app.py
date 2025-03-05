import pandas as pd
import json
import chromadb
from chromadb.utils import embedding_functions
import typer
from rich.console import Console
from rich.prompt import Prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import warnings
import traceback
import os
import glob
import sys
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import gc

# CUDA bellek ayarları
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

console = Console()
app = typer.Typer()

class RAGSystem:
    def __init__(self, json_path: str):
        try:
            self.json_path = json_path
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            console.print(f"\nCihaz: {self.device}", style="yellow")
            
            # Hugging Face token kontrolü
            self.hf_token = os.getenv('HUGGING_FACE_TOKEN')
            if not self.hf_token:
                raise ValueError("HUGGING_FACE_TOKEN bulunamadı! Lütfen .env dosyasına token'ınızı ekleyin.")
            
            self.setup_model()
            self.setup_database()
        except Exception as e:
            console.print("\n❌ Başlatma Hatası:", style="bold red")
            console.print(f"Hata Mesajı: {str(e)}", style="red")
            console.print("\nHata Detayı:", style="bold red")
            console.print(traceback.format_exc(), style="red")
            raise e
        
    def setup_model(self):
        try:
            console.print("\nModel yükleniyor...", style="yellow")
            console.print(f"Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", style="yellow")
            
            self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
            
            # Cache dizinleri için bilgi
            cache_dir = os.getenv('TRANSFORMERS_CACHE', './cache/huggingface')
            console.print(f"Cache Konumu: {cache_dir}", style="yellow")
            
            # Tokenizer yapılandırması
            console.print("\nTokenizer yükleniyor...", style="yellow")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True,
                padding_side="left",
                use_fast=False,
                cache_dir=cache_dir,
                local_files_only=False
            )
            console.print("Tokenizer yüklendi!", style="green")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Model yapılandırması
            console.print("\nModel dosyaları indiriliyor veya cache'den yükleniyor...", style="yellow")
            
            # Cache durumunu kontrol et
            model_cache_path = os.path.join(cache_dir, "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B")
            
            # Çevre değişkeni kontrolü
            use_local_first = os.getenv('USE_LOCAL_MODEL_FIRST', 'false').lower() == 'true'
            
            # Daha detaylı cache kontrolü
            model_files_complete = False
            if os.path.exists(model_cache_path):
                # Shard dosyalarını kontrol et
                shard_pattern = os.path.join(model_cache_path, "snapshots", "*", "model-*.safetensors")
                shard_files = glob.glob(shard_pattern)
                if len(shard_files) >= 8:  # DeepSeek modeli 8 shard içeriyor
                    model_files_complete = True
            
            if os.path.exists(model_cache_path) and use_local_first and model_files_complete:
                console.print(f"Model cache bulundu: {model_cache_path}", style="green")
                console.print("USE_LOCAL_MODEL_FIRST=true ayarlandı ve tüm model dosyaları mevcut, önce yerel model deneniyor", style="green")
                try_local_first = True
            else:
                if not os.path.exists(model_cache_path):
                    console.print("Model cache bulunamadı, indiriliyor...", style="yellow")
                elif not model_files_complete:
                    console.print("Model cache bulundu fakat eksik dosyalar var, indiriliyor...", style="yellow")
                else:
                    console.print("Model cache bulundu fakat USE_LOCAL_MODEL_FIRST=false, indiriliyor...", style="yellow")
                try_local_first = False
            
            try:
                # Önce yerel dosyalardan yüklemeyi dene
                if try_local_first:
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            token=self.hf_token,
                            torch_dtype=torch.bfloat16,
                            device_map="auto",  # Otomatik cihaz haritalaması
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            use_cache=True,
                            cache_dir=cache_dir,
                            local_files_only=True,
                            max_memory={0: "18GiB", 1: "18GiB"},  # Her iki GPU için bellek sınırlaması
                            offload_folder="offload",  # Gerekirse CPU'ya offload et
                        )
                        console.print("Model yerel cache'den yüklendi!", style="green")
                    except Exception as local_error:
                        console.print(f"Yerel cache'den yükleme başarısız: {str(local_error)}", style="yellow")
                        raise local_error
                else:
                    raise FileNotFoundError("Yerel cache atlanıyor")
                    
            except Exception as e:
                console.print("Online'dan model indiriliyor...", style="yellow")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    token=self.hf_token,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",  # Otomatik cihaz haritalaması
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    use_cache=True,
                    cache_dir=cache_dir,
                    local_files_only=False,
                    resume_download=True,
                    max_memory={0: "18GiB", 1: "18GiB"},  # Her iki GPU için bellek sınırlaması
                    offload_folder="offload",  # Gerekirse CPU'ya offload et
                )
                console.print("Model indirildi ve cache'e kaydedildi!", style="green")
            
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            console.print("\nModel yapılandırması tamamlandı!", style="green")
            
        except Exception as e:
            console.print("\n❌ Model Yükleme Hatası:", style="bold red")
            console.print(f"Hata Mesajı: {str(e)}", style="red")
            console.print("\nHata Detayı:", style="bold red")
            console.print(traceback.format_exc(), style="red")
            raise e
        
    def setup_database(self):
        console.print("Veritabanı hazırlanıyor...", style="yellow")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Sentence transformer embedding fonksiyonunu kullan
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2"
        )
        
        # Koleksiyon oluştur veya var olanı al
        self.collection = self.chroma_client.get_or_create_collection(
            name="json_data",
            embedding_function=self.embedding_function
        )
        
        # JSON verilerini yükle ve veritabanına ekle
        if self.collection.count() == 0:
            self.load_json_data()
            
        console.print("Veritabanı hazır!", style="green")
        
    def load_json_data(self):
        console.print("JSON verisi yükleniyor...", style="yellow")
        try:
            with open(self.json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # JSON yapısını kontrol et
            if "Sheet1" in data:
                # Veriyi işle
                documents = []
                ids = []
                
                for idx, item in enumerate(data["Sheet1"]):
                    # Satırdaki tüm değerleri stringe çevir ve birleştir
                    content = " ".join([f"{key}: {str(value)}" for key, value in item.items()])
                    documents.append(content)
                    ids.append(f"doc_{idx}")
                
                # Belgeleri batch halinde ekle (bellek kullanımını optimize etmek için)
                batch_size = 100
                for i in range(0, len(documents), batch_size):
                    end_idx = min(i + batch_size, len(documents))
                    self.collection.add(
                        documents=documents[i:end_idx],
                        ids=ids[i:end_idx]
                    )
                    
                console.print(f"Toplam {len(documents)} belge eklendi.", style="green")
            else:
                console.print("JSON dosyası beklenen formatta değil.", style="red")
                raise ValueError("JSON dosyası 'Sheet1' anahtarını içermiyor.")
                
        except Exception as e:
            console.print(f"JSON veri yükleme hatası: {str(e)}", style="red")
            raise e
        console.print("JSON verisi yüklendi!", style="green")
        
    def query(self, question: str) -> str:
        # Bellek temizliği
        gc.collect()
        torch.cuda.empty_cache()
        
        # Benzer dökümanları bul
        results = self.collection.query(
            query_texts=[question],
            n_results=3
        )
        
        # Prompt oluştur
        context = "\n".join(results["documents"][0])
        prompt = f"""### GÖREV:
Aşağıda verilen bağlam bilgilerini kullanarak kullanıcının sorusuna kapsamlı ve doğru bir yanıt oluştur.

### BAĞLAM:
{context}

### SORU:
{question}

### TALİMATLAR:
1. Önce bağlamı dikkatlice analiz et ve soruyla ilgili önemli bilgileri belirle.
2. Adım adım düşünerek yanıtı oluştur.
3. Yanıtın tamamen bağlam içindeki bilgilere dayalı olmalıdır.
4. Eğer bağlamda soruyu yanıtlamak için yeterli bilgi yoksa, "Üzgünüm, bu soruyu yanıtlamak için yeterli bilgim yok" şeklinde belirt.
5. Yanıtında bağlamda olmayan bilgileri uydurma.
6. Yanıtını açık, anlaşılır ve öz bir şekilde sunmaya çalış.

### YANITIM:
"""
        
        # Model ile yanıt oluştur
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Bellek optimizasyonu için batch size'ı küçült
        with torch.cuda.amp.autocast():  # Otomatik karışık hassasiyet kullan
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,  # Yanıt için maksimum yeni token sayısı
                num_return_sequences=1,
                temperature=0.3,  # Daha tutarlı yanıtlar için düşük sıcaklık
                top_p=0.85,  # Nucleus sampling için
                do_sample=True,  # Çeşitlilik için örnekleme yap
                no_repeat_ngram_size=3,  # Tekrarları önle
                repetition_penalty=1.2,  # Tekrarları cezalandır
                pad_token_id=self.tokenizer.pad_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response.split("Yanıt:")[-1].strip()

@app.command()
def main(json_path: str = "Book1.json"):
    """
    RAG sistemini başlat ve kullanıcı sorularını yanıtla.
    """
    try:
        rag = RAGSystem(json_path)
        console.print("\n🤖 RAG Sistemi hazır! Çıkmak için 'exit' yazın.\n", style="bold green")
        
        while True:
            try:
                question = Prompt.ask("\nSorunuzu yazın")
                
                if question.lower() == "exit":
                    break
                    
                with console.status("Yanıt oluşturuluyor..."):
                    response = rag.query(question)
                    
                console.print("\nYanıt:", style="bold blue")
                console.print(response, style="green")
            except EOFError:
                console.print("\n\n👋 Program sonlandırılıyor (EOF alındı)...", style="bold yellow")
                break
            except KeyboardInterrupt:
                console.print("\n\n👋 Program sonlandırılıyor (Ctrl+C)...", style="bold yellow")
                break
            
    except Exception as e:
        console.print(f"\n❌ Hata: {str(e)}", style="bold red")

if __name__ == "__main__":
    app() 