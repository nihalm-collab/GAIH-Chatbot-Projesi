## İlaç Bilgilendirme Chatbotu

## Proje Hakkında
Bu proje,kullanıcıların yaygın ilaçların etken maddeleri, temel kullanım amaçları ve potansiyel yaygın yan etkileri hakkında güvenilir ve kaynak bazlı bilgi almasını sağlayan bir AI asistanı oluşturur. AI asistan, kesinlikle tıbbi tavsiye vermeyecektir.

## Kullanılan Teknolojiler
Haystack: RAG pipeline framework
**Kaynak:** Kaggle - Ahmed Mohamed Zaki
**İçerik Özeti:**
Bu veri seti, 2001-2022 yılları arasında kaydedilmiş **782 önemli deprem kaydını** içermektedir. Her kayıt, depremin büyüklüğü (magnitude), derinliği (depth), coğrafi koordinatları (latitude/longitude), uyarı seviyesi (alert: green, yellow, orange, red) ve **tsunami potansiyeli (tsunami: 1/0)** gibi temel jeolojik ve afet risk özelliklerini barındırır.

**Veri Hazırlama Metodolojisi:**
[cite_start]Hazır CSV veri seti kullanılmıştır[cite: 16]. Veri, Pandas kütüphanesi ile yüklenmiş ve her bir deprem kaydı için ilgili sayısal ve kategorik sütunlar, RAG sistemi tarafından anlamlandırılabilmesi için tek bir açıklayıcı metin bloğu (`chunk`) halinde birleştirilmiştir.

## [cite_start]3. Kullanılan Yöntemler ve Çözüm Mimarisi [cite: 11]

Proje, **Retrieval-Augmented Generation (RAG)** temelinde inşa edilmiştir.

[cite_start]**Teknolojik Bileşenler:** [cite: 42, 43, 44]
* **RAG Framework:** <LangChain / Haystack / Özel Python Scripti>
* **Generation Model (LLM):** <Gemini API / OpenAI API / Diğer>
* **Embedding Model:** <Google-embed-001 / Cohere / Diğer>
* **Vektör Database:** <ChromaDB / FAISS / Pinecone / Diğer>
* **Web Arayüzü:** <Streamlit / Flask / Diğer>

**Çözüm Mimarisi Akışı (Özet):**
1.  Kullanıcı bir sorgu gönderir (Örn: "En büyük tsunami riski taşıyan 7.0 büyüklüğündeki deprem hangisiydi?").
2.  Sorgu, **Embedding Model** ile vektöre dönüştürülür.
3.  Vektör, **Vektör Database** içinde aranarak, sorguyla en alakalı (en benzer vektörlere sahip) **deprem kayıtları** (metin parçaları) alınır (Retrieval).
4.  Alınan bu kayıtlar (kanıt metinleri), orijinal sorguyla birlikte **Generation Model**'e gönderilir.
5.  **Generation Model (LLM)**, sağlanan bilgilere dayanarak tutarlı ve bilgilendirici bir yanıt üretir ve kullanıcıya sunar.

## [cite_start]4. Elde Edilen Sonuçlar Özeti [cite: 12]

<Buraya, chatbot'unuzun başarısını gösteren kısa bir özet yazın. Örneğin:>
* Chatbot, <Hassasiyet Oranı>% doğrulukla, veri setindeki deprem bilgilerini sorgulara yanıtlayabilmiştir.
* En başarılı yanıtlar, büyüklük veya yıl bazlı sorgulamalarda elde edilmiştir.
* Proje, RAG'ın yapılandırılmış veriyi (CSV) doğal dil yanıtlarına dönüştürme yeteneğini başarılı bir şekilde kanıtlamıştır.

## 5. Proje Çalışma Kılavuzu ve Web Arayüzü

Projenin kurulumu ve çalıştırılmasına dair detaylı adımlar, <Ayrı Bir Dosya (Örn: `SETUP.md`) veya Bu README'nin İlgili Bölümü> altında yer almaktadır.

Projenin web arayüzü ile ilgili detaylı kullanım kılavuzu ve ekran görüntüleri/videosu <Ayrı Bir Dosya (Örn: `PRODUCT.md`) veya Bu README'nin İlgili Bölümü> altında bulunmaktadır.

## [cite_start]🌐 Web Arayüzü Linki (Mutlaka Paylaşılmalıdır) [cite: 13]

Projenin deploy edildiği çalışan link:

**<BURAYA ÇALIŞAN CHATBOT WEB LİNKİNİZİ EKLEYİN>**
