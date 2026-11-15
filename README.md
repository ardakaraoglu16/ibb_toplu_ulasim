
# İstanbul Toplu Taşıma Yolcu Talebi Tahmini ve Karar Destek Sistemi

Python, XGBoost ve Streamlit kullanılarak geliştirilen bu proje, İstanbul toplu taşıma verilerini analiz ederek saatlik yolcu talebini tahmin etmeyi ve veri odaklı karar destek mekanizmaları sunmayı amaçlamaktadır.

Proje kapsamında Mart 2021 dönemine ait 9 milyondan fazla toplu taşıma kaydı analiz edilmiş, makine öğrenmesi modeli geliştirilmiş ve sonuçlar etkileşimli bir Streamlit uygulaması üzerinden sunulmuştur.

## Proje Hakkında

Proje üç temel modülden oluşmaktadır:

### 1. Veri Analizi Dashboard

Toplu taşıma verilerinin genel yapısını ve yolcu yoğunluğu eğilimlerini incelemek amacıyla geliştirilmiştir.

- Toplam yolcu sayısı ve veri setine ait temel istatistikler
- Saatlik ortalama yolcu yoğunluğu analizi
- Hafta içi ve hafta sonu yolcu dağılımı
- Yoğun saatlerin belirlenmesi
- Toplu taşıma talebinin zamana göre incelenmesi

### 2. Makine Öğrenmesi ile Yolcu Talebi Tahmini

Saatlik yolcu talebini tahmin etmek amacıyla XGBoost regresyon modeli geliştirilmiştir.

Modelde kullanılan temel özellikler:

- Saat
- Haftanın günü
- Hafta sonu bilgisi
- Önceki saat yolcu sayısı (Lag 1)
- Önceki gün aynı saat yolcu sayısı (Lag 24)
- Hareketli ortalama (Rolling Mean)
- Yoğun saat bilgisi

**Model Performansı:**

| Metrik | Değer |
|---|---:|
| MAE | 5.476 |
| RMSE | 8.987 |
| R² | 0,996 |

Model performansı test verisi üzerinde değerlendirilmiştir.

### 3. Karar Destek Sistemi

Model tahminlerinden elde edilen sonuçları daha anlaşılır hâle getirmek ve toplu taşıma planlamasına katkı sağlamak amacıyla geliştirilmiştir.

- Tahmini yolcu yoğunluğunun sınıflandırılması
- Tahmini etkileyen özelliklerin incelenmesi
- Gerçek ve tahmin edilen yolcu sayılarının karşılaştırılması
- Tahmin sonuçlarına göre toplu taşıma planlamasına yönelik öneriler

## Uygulama Ekran Görüntüleri

### Veri Analizi Dashboard

Veri setinin genel istatistikleri, saatlik yolcu yoğunluğu ve hafta içi / hafta sonu dağılımı.

<img width="1896" height="815" alt="Ekran görüntüsü 2026-11-15 170213" src="https://github.com/user-attachments/assets/303bab06-b759-4a9d-9752-39b8c3611050" />


### Yolcu Talebi Tahmini

Seçilen tarih ve saate göre model tahmini, gerçek yolcu sayısı ve tahmin hatası.

<img width="1885" height="573" alt="Ekran görüntüsü 2026-11-15 170248" src="https://github.com/user-attachments/assets/4feec713-57be-4e3d-aee1-b619bad322b5" />


### Karar Destek Sistemi

Tahmini etkileyen faktörler, yolcu yoğunluğu seviyesi ve belediyeye yönelik öneriler.

<img width="1880" height="807" alt="Ekran görüntüsü 2026-11-15 170307" src="https://github.com/user-attachments/assets/b76c9836-d2b0-458d-a076-316bd3a5f925" />


## Kullanılan Teknolojiler

| Teknoloji | Kullanım Alanı |
|---|---|
| Python | Veri işleme ve model geliştirme |
| Pandas | Veri analizi ve manipülasyonu |
| NumPy | Sayısal işlemler |
| Scikit-learn | Model değerlendirme |
| XGBoost | Yolcu talebi tahmini |
| Streamlit | İnteraktif web uygulaması |
| Plotly | Veri görselleştirme |
| Joblib | Model ve özelliklerin kaydedilmesi |

## Veri Seti

Projede Mart 2021 dönemine ait İstanbul toplu taşıma verileri kullanılmıştır.

Veri setinde bulunan temel değişkenler:

- İşlem tarihi ve saati
- Ulaşım türü
- Hat bilgisi
- Yolcu sayısı
- Aktarma bilgisi
- İlçe ve istasyon bilgileri

Veriler analiz edilerek makine öğrenmesi modelinde kullanılabilecek zamansal özellikler oluşturulmuştur.

## Kurulum

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

**1. Depoyu klonlayın**

```bash
git clone https://github.com/ardakaraoglu16/ibb_toplu_ulasim.git
```

**2. Proje klasörüne girin**

```bash
cd ibb_toplu_ulasim
```

**3. Gerekli kütüphaneleri yükleyin**

```bash
pip install -r requirements.txt
```

**4. Streamlit uygulamasını başlatın**

```bash
streamlit run app.py
```

Not: Uygulamanın çalışması için gerekli veri ve model dosyalarının proje dizininde bulunması gerekir.

## Projenin Amacı

Bu çalışma ile makine öğrenmesi tekniklerinin toplu taşıma sistemlerinde yolcu talebinin tahmin edilmesi, yoğunluk eğilimlerinin analiz edilmesi ve veri odaklı karar alma süreçlerinin desteklenmesindeki potansiyelinin gösterilmesi hedeflenmiştir.

## Geliştirici

**Arda Karaoğlu**

Yönetim Bilişim Sistemleri Öğrencisi

- GitHub: https://github.com/ardakaraoglu16
