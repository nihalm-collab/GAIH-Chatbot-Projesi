"""
📚 Kitap Asistanı Chatbot (Akbank GenAI Bootcamp)
--------------------------------------------------
Bu proje, Kitapyurdu yorum verisetine dayalı olarak
RAG (Retrieval Augmented Generation) mimarisiyle
çalışan bir kitap asistanı chatbotudur.
"""

# ===============================
# 1️⃣ Gerekli kütüphaneler
# ===============================
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from datasets import load_dataset
import google.generativeai as genai
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ===============================
# 2️⃣ Ortam değişkenleri
# ===============================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

# ===============================
# 3️⃣ Veri seti yükleme
# ===============================
DATA_PATH = "data/kitapyurdu_sample.csv"
os.makedirs("data", exist_ok=True)

if not os.path.exists(DATA_PATH):
    dataset = load_dataset("alibayram/kitapyurdu_yorumlar", split="train", use_auth_token=HF_TOKEN)
    df = pd.DataFrame(dataset)
    df = df[["kitap_adi", "yorum", "puan"]].dropna(subset=["yorum"])
    df = df.sample(n=2000, random_state=42)
    df.to_csv(DATA_PATH, index=False)
else:
    df = pd.read_csv(DATA_PATH)

# ===============================
# 4️⃣ Embedding ve ChromaDB
# ===============================
persist_dir = "chroma_db"
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if os.path.exists(persist_dir):
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
else:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_text("\n".join(df["yorum"].astype(str).tolist()))
    vector_store = Chroma.from_texts(texts=texts, embedding=embedding_model, persist_directory=persist_dir)
    vector_store.persist()

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
model = genai.GenerativeModel("gemini-1.5-flash")

# ===============================
# 5️⃣ Streamlit Arayüzü
# ===============================
st.set_page_config(page_title="📚 Kitap Asistanı Chatbot", layout="wide")

st.title("📚 Kitap Asistanı Chatbot")
st.markdown("### 💬 Kitapyurdu yorumlarına dayalı akıllı kitap asistanı")
st.markdown("Soru sor: Örneğin, *'Zülfü Livaneli kitapları hakkında genel izlenimler nasıl?'*")

# Sohbet geçmişi
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Sorunu yaz:", key="user_input")

def search_kitap(query):
    results = retriever.get_relevant_documents(query)
    context = "\n".join([r.page_content for r in results])

    prompt = f"""
    Aşağıdaki kullanıcı yorumlarına dayanarak kitaplarla ilgili soruya yanıt ver:
    ---
    {context}
    ---
    Soru: {query}
    Yanıtın Türkçe, doğal ve özet olsun.
    """

    response = model.generate_content(prompt)
    return response.text

# Kullanıcı sorgusu işlendiğinde
if st.button("Gönder") and user_input:
    answer = search_kitap(user_input)
    st.session_state.chat_history.append((user_input, answer))
    st.text_area("Chatbot Yanıtı", answer, height=200)

# Sohbet geçmişini göster
if st.session_state.chat_history:
    st.markdown("### 🕒 Sohbet Geçmişi")
    for q, a in st.session_state.chat_history[-5:]:
        st.markdown(f"**Sen:** {q}")
        st.markdown(f"**Asistan:** {a}")
        st.markdown("---")
