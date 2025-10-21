"""
📚 KitapYurdu Yorum Asistanı Chatbot (Gemini 2.0 Flash)
--------------------------------------------------------
Bu proje, Hugging Face üzerindeki 'alibayram/kitapyurdu_yorumlar' veri setini kullanarak
Gemini 2.0 Flash modeliyle RAG (Retrieval Augmented Generation) mimarisine dayalı
bir kitap asistanı chatbotu oluşturur.
"""

import os
from dotenv import load_dotenv
import streamlit as st
from datasets import load_dataset
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- 1. Ortam Değişkenlerini Yükle
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- 2. Streamlit Başlığı
st.set_page_config(page_title="📖 KitapYurdu Yorum Asistanı (Gemini)")
st.title("📖 KitapYurdu Yorum Asistanı (Gemini 2.0 Flash)")

# --- 3. Hugging Face'ten Veri Çek
@st.cache_data
def load_kitapyurdu_dataset():
    dataset = load_dataset("alibayram/kitapyurdu_yorumlar", split="train", token=HF_TOKEN)
    return dataset

st.write("📡 Veri seti yükleniyor...")
dataset = load_kitapyurdu_dataset()
st.success("✅ Veri seti başarıyla yüklendi!")

# --- 4. Metinleri Böl
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_text(" ".join(dataset["yorum"][:500]))  # İlk 500 yorum örnek olarak alınır

# --- 5. ChromaDB (vektör veritabanı)
PERSIST_DIR = "chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)

client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIR))

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)

vectorstore = Chroma.from_texts(
    texts,
    embeddings,
    persist_directory=PERSIST_DIR
)

# --- 6. Retriever + LLM (Gemini 2.0 Flash)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.2
)

from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
)

# --- 7. Kullanıcı Girdisi
st.markdown("### 💬 Kitaplar hakkında bir soru sor:")
user_query = st.text_input("Örnek: 'En çok beğenilen kitap hangisi?'", "")

if user_query:
    with st.spinner("Yanıt oluşturuluyor..."):
        response = qa_chain.run(user_query)
        st.markdown("### 🧠 Yanıt:")
        st.write(response)
