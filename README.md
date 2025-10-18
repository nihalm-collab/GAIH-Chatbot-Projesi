# Deprem Bilgilendirme ve Risk Analizi Chatbotu

Generative AI 101 Bootcamp için hazırlanmış Türkçe RAG (Retrieval-Augmented Generation) tabanlı chatbot projesi.

## Proje Hakkında

Bu proje, Kaggle'daki Türkiye deprem verilerini (1914-2023) çekerek kullanıcıların deprem tarihi, yeri, büyüklüğü gibi sorularına yanıt verir.

## Çözüm Mimarisi

Proje, **





## Veri Seti Hakkında Bilgi

1914-2023 yılları arasındaki Türkiye'deki depremlere ait tarih, saat, enlem, boylam, büyüklük gibi kayıtları içeren Kaggle veri seti kullanılmıştır.

**Ana Veri Seti:** Turkey Earthquake Data (1914 - 2023)
**Kaynak:** Kaggle (Özge Çinko)
**İçerik** 1914-2023 yılları arasındaki depremlerin; Enlem, Boylam, Büyüklük, Derinlik ve Oluş Tarihi bilgileri.
**Lisans:** **CC BY-SA 4.0 Uluslararası** 
**Hazırlanış Metodolojisi:** Veri seti hazır olarak kullanılmıştır. Veri seti içerisindeki `Enlem`, `Boylam` ve `Büyüklük` sütunları, bölgesel risk skorlaması için işlenmiştir ve RAG sürecinde kullanılmak üzere metin parçalarına (`chunks`) dönüştürülmüştür.
**Atıf Yükümlülüğü:** Lisans (CC BY-SA 4.0) gereği, veri setinin sahibine ve lisansa atıf yapılmıştır.

## Kullanılan Teknolojiler



## Kurulum

Python 3.9+ kurulu olmalıdır.
**1.
**2.Sanal bir ortam kurun ve etkinleştirin:** 
    '''python -m venv venv
    '''source venv/bin/activate # Linux/macOS
    '''.\venv\Scripts\activate   # Windows
**3.Gerekli paketleri kurun:**
    '''pip install -r requirements.txt
**4.Bir .env dosyası oluşturun ve Gemini API ile Kaggle API anahtarlarını ekleyin:** 
    '''export GEMINI_API_KEY="SİZİN_GEMINI_ANAHTARINIZ"
    '''export KAGGLE_USERNAME="SİZİN_KULLANICI_ADINIZ"
    '''export KAGGLE_KEY="SİZİN_ANAHTARINIZ"
**5.Uygulamayı başlatın:**
    '''streamlit run project.py


## [cite_start]🔗 Canlı Web Arayüzü Linki [cite: 13, 25]

> [cite_start]**[BURAYA PROJE DEPLOY EDİLDİĞİNDE GÜNCEL WEB LİNKİ MUTLAKA EKLENMELİDİR]** [cite: 13, 25]
