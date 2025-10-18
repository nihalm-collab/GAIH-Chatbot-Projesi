# Deprem Bilgilendirme ve Risk Analizi Chatbotu

> [cite_start]Bu proje, **Akbank GenAI Bootcamp** kapsamında, RAG (Retrieval Augmented Generation) temelli bir chatbot geliştirme yönergesine uygun olarak hazırlanmıştır[cite: 2]. Chatbot, Türkiye'deki tarihi sismik verilere dayalı risk analizi ve acil durum bilgilendirmesi sağlamayı hedefler.

## [cite_start] Projenin Amacı [cite: 9]

[cite_start]Bu projenin temel amacı[cite: 9], kullanıcıların deprem öncesi hazırlık, deprem anı ve sonrası süreçler için **güvenilir ve hızlı** bilgiye erişimini sağlamaktır.

1.  **Veriye Dayalı Risk Analizi:** Kullanıcının konumuna (enlem/boylam) ait geçmişteki büyük deprem verilerini işleyerek bölgesel sismik risk skorunu ve geçmiş olayları sunmak.
2.  **RAG ile Güvenilir Bilgi:** Deprem öncesi, anı ve sonrası gibi kritik konularda, **Gemini API** ve RAG mimarisi ile doğru ve entegre edilmiş kılavuzlardan bilgi sağlamak.
3.  [cite_start]**Hızlı Erişim:** Basit bir web arayüzü üzerinden proje kabiliyetlerini sergilemek[cite: 25].

## [cite_start] Veri Seti Hakkında Bilgi [cite: 10]

Bu projenin temel sismik veri kaynağı, Kaggle'dan elde edilen tarihi deprem verileridir.

| Başlık | Detay |
| :--- | :--- |
| **Ana Veri Seti** | Turkey Earthquake Data (1914 - 2023) |
| **Kaynak** | Kaggle (Ozge Cinko) |
| **İçerik** | 1914-2023 yılları arasındaki depremlerin; Enlem, Boylam, Büyüklük, Derinlik ve Oluş Tarihi bilgileri. |
| **Lisans** | **CC BY-SA 4.0 Uluslararası** |
| **Hazırlanış Metodolojisi:** | [cite_start]Veri seti hazır olarak kullanılmıştır[cite: 17]. Veri seti içerisindeki `Enlem`, `Boylam` ve `Büyüklük` sütunları, bölgesel risk skorlaması için işlenmiştir ve RAG sürecinde kullanılmak üzere metin parçalarına (`chunks`) dönüştürülmüştür. |
| **Atıf Yükümlülüğü:** | Lisans (CC BY-SA 4.0) gereği, veri setinin sahibine ve lisansa atıf yapılmıştır. |

***

## [cite_start]⚙️ Çözüm Mimarisi ve Kullanılan Yöntemler [cite: 11, 23, 44]

[cite_start]Projenin temel mimarisi RAG (Retrieval Augmented Generation) üzerine kurulmuştur[cite: 2, 23].

| Bileşen | Örnek Teknoloji | Görev ve Amacı |
| :--- | :--- | :--- |
| **LLM (Generation Model)** | [cite_start]Gemini API [cite: 33, 42] | [cite_start]Vektör DB'den gelen bağlamı kullanarak nihai, akıcı ve bilgilendirici cevabı üretmek[cite: 42]. |
| **RAG Çerçevesi** | [cite_start]LangChain veya Haystack [cite: 35, 44] | [cite_start]Veri alımı, bağlam oluşturma ve LLM'e gönderme sürecini yönetmek[cite: 44]. |
| **Vektör Veritabanı** | [cite_start]Chroma / FAISS / Pinecone [cite: 43] | Sismik verilerden ve ek acil durum kılavuzlarından türetilen metin parçalarını vektör olarak saklamak. |
| **Embedding Modeli** | [cite_start]Google Embedding Modeli [cite: 43] | Metin verilerini, vektör veritabanında arama yapılabilir hâle getirmek. |
| **Web Arayüzü** | Streamlit / Gradio | [cite_start]Chatbot'un test edileceği kullanıcı arayüzünü oluşturmak[cite: 24]. |

***

## [cite_start]💡 Elde Edilen Sonuçlar (Özet) [cite: 12]

[cite_start]Projenin sonunda "Güvenli Adım" Chatbot'u, aşağıdaki kabiliyetleri özet şekilde [cite: 12] sunacaktır:

* **Tarihsel Risk Analizi:** Kullanıcının sorguladığı bölgeye ait sismik verileri anlık olarak analiz edebilme.
* **Bağlamsal Bilgilendirme:** RAG sayesinde, deprem anına ve sonrasına dair (ilk yardım, güvenlik önlemleri) sorulara güvenilir kılavuzlardan yanıt üretebilme.
* [cite_start]**Kolay Kullanım:** Web arayüzü üzerinden projeyi sergileme[cite: 25].

***

## [cite_start]🚀 Projeyi Yerelde Çalıştırma Kılavuzu [cite: 19, 20]

[cite_start]Kodun çalıştırılabilmesi için gerekenler bu kılavuzda yer alacaktır[cite: 20].

1.  **Depoyu Klonlayın:**
    ```bash
    git clone [REPO_LİNKİNİZ]
    cd [REPO_ADI]
    ```

2.  **Sanal Ortam Kurulumu:**
    ```bash
    # Sanal ortam oluşturma
    python -m venv venv
    
    # Sanal ortamı aktive etme
    source venv/bin/activate  # Linux/macOS
    # veya
    .\venv\Scripts\activate    # Windows
    ```

3.  **Bağımlılıkları Yükleme:**
    * [cite_start]Gerekli tüm kütüphaneler (`requirements.txt` dosyanızda [cite: 21] yer alacaktır).
    ```bash
    pip install -r requirements.txt 
    ```

4.  **API Anahtarını Ayarlama:**
    * **Gemini API Key** edinin ve ortam değişkeni olarak ayarlayın:
    ```bash
    # Linux/macOS
    export GEMINI_API_KEY="YOUR_API_KEY" 
    # Windows (Command Prompt)
    set GEMINI_API_KEY="YOUR_API_KEY"
    ```

5.  **Chatbot'u Başlatma:**
    ```bash
    [cite_start]streamlit run app.py # veya python your_main_file.py [cite: 21]
    ```

***

## [cite_start]🔗 Canlı Web Arayüzü Linki [cite: 13, 25]

> [cite_start]**[BURAYA PROJE DEPLOY EDİLDİĞİNDE GÜNCEL WEB LİNKİ MUTLAKA EKLENMELİDİR]** [cite: 13, 25]
