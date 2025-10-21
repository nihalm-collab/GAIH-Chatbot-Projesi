# Kitapyurdu Yorum Chatbotu

Bu proje, Kitapyurdu yorumları üzerine geliştirilmiş bir **RAG tabanlı chatbot** içerir. 
Kullanıcılar kitaplarla ilgili sorular sorabilir ve chatbot, kullanıcı yorumlarını baz alarak yanıt üretir.

---

## Projenin Amacı
- Kitap yorumlarını kullanarak kullanıcıların sorularını yanıtlamak
- Kitapların olumlu/olumsuz yönlerini özetlemek
- RAG mimarisi ile doğruluğu artırmak

---

## Veri Seti
- **Ad:** Kitapyurdu Yorumları
- **Kaynak:** [HuggingFace](https://huggingface.co/datasets/alibayram/kitapyurdu_yorumlar) (token gerektirir)
- **Sütunlar:**
  - yorum: Kullanıcı yorumu
  - kitap_adi: Kitap adı
  - puan: 1-5 arası kullanıcı puanı
  - tarih: Yorum tarihi
- **İçerik:** Türkçe kitap yorumları, olumlu ve olumsuz görüşler içerir.
- **Hazırlık:** Boş veya kısa yorumlar çıkarıldı, Türkçe karakterler normalize edildi. Metadata olarak kitap adı ve puan saklandı.

---

## Kullanılan Yöntemler
1. **RAG Pipeline**
   - Metinler embedding modeline (OpenAI/Gemini) gönderildi
   - Chroma vektör veritabanında saklandı
   - Soru geldiğinde retriever benzer yorumları buluyor ve LLM yanıt üretiyor
2. **Embedding Modeli**
   - OpenAI `text-embedding-3-small` modeli
3. **Vektör Veritabanı**
   - Chroma kullanıldı
4. **Web Arayüzü**
   - Streamlit ile kullanıcı etkileşimi sağlandı

---

## Elde Edilen Sonuçlar
- Kullanıcı sorularına hızlı ve bağlamsal cevaplar üretildi
- Örnek sorular:
  - "Bu kitabın konusu nedir?"
  - "Kullanıcılar bu kitabı beğenmiş mi?"
- Arayüz minimal ama işlevsel, soru-yanıt geçmişi tutuluyor

---

## Kurulum ve Çalıştırma
1. Reponuzu klonlayın:
```bash
git clone <REPO_LINK>
cd <REPO_NAME>
```
2. Virtual environment oluşturun ve aktif edin:
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```
3. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```
4. `.env` dosyasını oluşturun ve API anahtarlarınızı ekleyin:
```bash
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
```
5. Streamlit uygulamasını çalıştırın:
```bash
streamlit run app.py
``` 
