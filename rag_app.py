import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import typer
from rich.console import Console
from rich.prompt import Prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

console = Console()
app = typer.Typer()

class RAGSystem:
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_model()
        self.setup_database()
        
    def setup_model(self):
        console.print("Model yükleniyor...", style="yellow")
        self.model_name = "deepseek-ai/deepseek-r1-distill-qwen-32b"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        console.print("Model yüklendi!", style="green")
        
    def setup_database(self):
        console.print("Veritabanı hazırlanıyor...", style="yellow")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Sentence transformer embedding fonksiyonunu kullan
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Koleksiyon oluştur veya var olanı al
        self.collection = self.chroma_client.get_or_create_collection(
            name="excel_data",
            embedding_function=self.embedding_function
        )
        
        # Excel verilerini yükle ve veritabanına ekle
        if self.collection.count() == 0:
            self.load_excel_data()
            
        console.print("Veritabanı hazır!", style="green")
        
    def load_excel_data(self):
        console.print("Excel verisi yükleniyor...", style="yellow")
        df = pd.read_excel(self.excel_path)
        
        # Her satırı bir döküman olarak ekle
        for idx, row in df.iterrows():
            # Satırdaki tüm değerleri stringe çevir ve birleştir
            content = " ".join([f"{col}: {str(val)}" for col, val in row.items()])
            self.collection.add(
                documents=[content],
                ids=[f"row_{idx}"]
            )
        console.print("Excel verisi yüklendi!", style="green")
        
    def query(self, question: str) -> str:
        # Benzer dökümanları bul
        results = self.collection.query(
            query_texts=[question],
            n_results=3
        )
        
        # Prompt oluştur
        context = "\n".join(results["documents"][0])
        prompt = f"""Aşağıdaki bağlam verilmiştir. Bu bağlamı kullanarak soruyu yanıtla.

Bağlam:
{context}

Soru: {question}

Yanıt:"""
        
        # Model ile yanıt oluştur
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response.split("Yanıt:")[-1].strip()

@app.command()
def main(excel_path: str = "Book1.xlsx"):
    """
    RAG sistemini başlat ve kullanıcı sorularını yanıtla.
    """
    try:
        rag = RAGSystem(excel_path)
        console.print("\n🤖 RAG Sistemi hazır! Çıkmak için 'exit' yazın.\n", style="bold green")
        
        while True:
            question = Prompt.ask("\nSorunuzu yazın")
            
            if question.lower() == "exit":
                break
                
            with console.status("Yanıt oluşturuluyor..."):
                response = rag.query(question)
                
            console.print("\nYanıt:", style="bold blue")
            console.print(response, style="green")
            
    except Exception as e:
        console.print(f"\n❌ Hata: {str(e)}", style="bold red")

if __name__ == "__main__":
    app() 