# renAI: Yapay Zeka Güçlü Belge Yeniden Adlandırıcı ve Düzenleyici

`renAI`, dijital belgelerinizi Büyük Dil Modelleri (LLM) ve Optik Karakter Tanıma (OCR) kullanarak otomatik olarak yeniden adlandıran ve düzenleyen güçlü bir Python aracıdır. Belge içeriğinden doğrudan meta verileri çıkararak tutarlı ve aranabilir dosya adları oluşturur.

**Çıktı Formatı:** `Başlık -- Alt Başlık -- 'Baskı' -- Yazar -- (Yıl).uzantı`

---

## 📖 Giriş

Büyük dijital belge koleksiyonlarını yönetmek zorlu olabilir. Dosyalar genellikle tutarsız veya anlamsız isimlerle gelir, bu da onları bulmayı ve düzenlemeyi zorlaştırır. renAI bunu şu şekilleri çözerek:

- **Belge içeriğini okuyarak** başlıkları, yazarları ve yayın yıllarını belirler
- **Belgeleri sınıflandırarak** özelleştirilebilir kategorilere ayırır
- **Tutarlı adlandırma kuralları uygulayarak** otomatik olarak yeniden adlandırır
- **İstenirse dosyaları** kategori tabanlı klasörlere düzenler

Araç birçok belge formatını (PDF, EPUB, MOBI, DOCX, PPTX, XLSX, TXT, MD) ve çeşitli LLM sağlayıcılarını (DeepInfra, OpenRouter, OpenAI) destekler.

### Adlandırma Kuralı

Dosyalar şu format kullanılarak yeniden adlandırılır:
```
Başlık -- Alt Başlık -- 'Baskı' -- Yazar -- (Yıl).uzantı
```

Örnek: `Makine Öğrenimine Giriş -- Pratik Bir Yaklaşım -- '1. Baskı' -- Andrew Ng -- (2018).pdf`

---

## ✨ Özellikler

| Özellik | Açıklama |
|---------|----------|
| **Çoklu Format Desteği** | PDF, EPUB, MOBI, DOCX, PPTX, XLSX, TXT ve MD dosyalarını işler |
| **Çoklu PDF Çıkarıcıları** | PyMuPDF, pdfplumber, pypdf veya pypdfium2 arasından seçim yapın |
| **Yapay Zeka Destekli Meta Veri** | Başlıkları, yazarları, yılları ve kategorileri çıkarmak için LLM'leri kullanır |
| **Gelişmiş OCR** | Düşük kaliteli taramalar için Enhanced Mode ile Tesseract entegrasyonu |
| **mutool Desteği** | Sabit düzenli PDF'lerden yüksek kaliteli görüntüler çıkarır |
| **Akıllı Önbellekleme** | API maliyetlerini ve işleme süresini azaltan MD5 tabanlı önbellekleme |
| **Özel Kategoriler** | `categories.txt` dosyasında dil başına kendi kategorilerinizi tanımlayın |

### Çalışma Modları

- **`rename`**: Çıkarılan meta verilere göre dosyaları yeniden adlandırır
- **`organize`**: Yeniden adlandırılmış dosyaları kategori klasörlerine taşır
- **`benchmark`**: PDF çıkarma kütüphanesi performansını karşılaştırır
- **`eval-model`**: Farklı LLM modellerini doğruluk açısından test eder

---

## 🛠️ Kurulum

### Ön Gereksinimler

1. **Python 3.10+**
2. **Tesseract OCR** - [Windows için indirin](https://github.com/UB-Mannheim/tesseract/wiki/Tesseract-at-UB-Mannheim) veya paket yöneticisi ile yükleyin:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # macOS
   brew install tesseract
   ```
3. **mutool (İsteğe bağlı)** - [MuPDF](https://mupdf.com/releases/index.html) sürümlerinin bir parçasıdır

### Kurulum

```bash
# Depoyu klonla
git clone https://github.com/ozgurulukir/renAI.git
cd renAI

# Sanal ortam oluştur (önerilen)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# veya: .venv\Scripts\activate  # Windows

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### API Anahtarları

API anahtarlarınızı ortam değişkenleri olarak ayarlayın:

```bash
# Windows (PowerShell)
$env:DEEPINFRA_API_KEY = "your_key_here"
$env:OPENROUTER_API_KEY = "your_key_here"
$env:OPENAI_API_KEY = "your_key_here"

# Linux/Mac
export DEEPINFRA_API_KEY="your_key_here"
export OPENROUTER_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"
```

Anahtarları şuradan alın: [DeepInfra](https://deepinfra.com) | [OpenRouter](https://openrouter.ai) | [OpenAI](https://platform.openai.com/api-keys)

---

## 📖 Kullanım Örnekleri

### Temel Yeniden Adlandırma

```bash
# Dizindeki tüm dosyaları yeniden adlandır
python main.py "C:/Kitaplar/Yolu" --mode rename
```

### Klasörlere Düzenleme

```bash
# Yeniden adlandır ve kategori klasörlerine düzenle
python main.py "C:/Kitaplar/Yolu" --mode organize
```

### Taranmış Belgeler için Yüksek Kaliteli OCR

```bash
# Görüntü ön işleme ile geliştirilmiş OCR
python main.py "C:/Kitaplar/Yolu" --ocr-enhanced --use-mutool
```

### Farklı LLM Sağlayıcıları Kullanma

```bash
# DeepInfra (varsayılan)
python main.py "C:/Kitaplar/Yolu" --mode rename --provider deepinfra

# OpenRouter
python main.py "C:/Kitaplar/Yolu" --mode rename --provider openrouter

# OpenAI
python main.py "C:/Kitaplar/Yolu" --mode rename --provider openai
```

### Yerel Çıkarım (Ollama / LM Studio)

`custom` sağlayıcısını kullanarak [Ollama](https://ollama.com/) veya [LM Studio](https://lmstudio.ai/) gibi yerel çıkarım sunucularına bağlanabilirsiniz. Bu araçlar OpenAI uyumlu API sağlar.

> **⚠️ Önemli: Tüm modeller yapılandırılmış çıktı üretemez, özellikle 7B parametrenin altındaki LLM'ler. Modelin yapılandırılmış çıktıyı destekleyip desteklemediğinden emin değilseniz model kartı README'sini kontrol edin.**
> 
> renAI güvenilir meta veri çıkarma için LLM'den JSON çıktısı gerektirir. 7B'den daha az parametreye sahip modeller yapılandırılmış çıktı formatlamada zorlanabilir.

```bash
# Ortam değişkenlerini ayarla (Windows PowerShell)
$env:CUSTOM_API_BASE_URL = "http://localhost:11434/v1"  # Ollama varsayılanı
$env:CUSTOM_MODEL = "llama3"  # Model adınız

# Özel sağlayıcı ile çalıştır
python main.py "C:/Kitaplar/Yolu" --mode rename --provider custom

# Veya modeli doğrudan geçersiz kıl
python main.py "C:/Kitaplar/Yolu" --mode rename --provider custom --model mistral
```

#### Yerel Modeller için Metin Uzunluğunu Kontrol Etme

Tüketici donanımlarında yerel modeller çalıştırırken, bellek kullanımını azaltmak ve performansı artırmak için LLM'e gönderilen metin uzunluğunu azaltabilirsiniz:

```bash
# Metin uzunluğunu 5000 karaktere azalt (varsayılan: 10000)
python main.py "C:/Kitaplar/Yolu" --mode rename --provider custom --text-length 5000

# Modeliniz kaldırabiliyorsa daha iyi bağlam için artırın
python main.py "C:/Kitaplar/Yolu" --mode rename --provider custom --text-length 20000
```

| Seçenek | Açıklama | Varsayılan |
|--------|-------------|-----------|
| `--text-length` | LLM'e gönderilen hedef metin uzunluğu | 10000 |

#### Özel Sağlayıcı için Ortam Değişkenleri

| Değişken | Açıklama | Varsayılan |
|----------|-------------|-----------|
| `CUSTOM_API_BASE_URL` | Yerel API sunucunuzun temel URL'si | `http://localhost:8000/v1` |
| `CUSTOM_MODEL` | Kullanılacak model adı | `llama3` |
| `CUSTOM_API_KEY` | API anahtarı (yerel sunucular için genellikle gerekmez) | `not-required` |

#### Yaygın Yerel Sunucu URL'leri

| Araç | URL |
|------|-----|
| Ollama | `http://localhost:11434/v1` |
| LM Studio | `http://localhost:1234/v1` |
| LocalAI | `http://localhost:8080/v1` |
| Text Generation WebUI | `http://localhost:5001/v1` |

### Özel Modeller

```bash
# DeepInfra ile özel model
python main.py "C:/Kitaplar/Yolu" --mode rename --model Qwen/Qwen2.5-72B-Instruct

# OpenRouter ile Gemini
python main.py "C:/Kitaplar/Yolu" --mode rename --provider openrouter --model google/gemini-2.0-flash-exp:free
```

### Daha Hızlı İşleme

```bash
# Eşzamanlı çalışanları artır
python main.py "C:/Kitaplar/Yolu" --mode rename --workers 8
```

### Meta Verileri Zorla Yenileme

```bash
# Önbelleği atla ve yeniden çıkar
python main.py "C:/Kitaplar/Yolu" --mode rename --update-metadata
```

### Kıyaslama ve Model Değerlendirme

#### PDF Çıkarıcı Kıyaslaması
Farklı PDF çıkarma kütüphanelerinin belgeleriniz üzerindeki performansını ve doğruluğunu karşılaştırın.
```bash
python main.py "C:/Kitaplar/Yolu" --benchmark
```

#### LLM Model Değerlendirmesi
Özel belge koleksiyonunuz için maliyet ve doğruluk arasındaki en iyi dengeyi bulmak amacıyla bir sağlayıcıdan birden fazla modeli test edin.
```bash
python main.py "C:/Kitaplar/Yolu" --eval-model --provider deepinfra
```

**Nasıl çalışır:**
- Modeller sağlayıcıya özel yapılandırma dosyalarından yüklenir: `eval_models_deepinfra.txt`, `eval_models_openai.txt` veya `eval_models_openrouter.txt`.
- Her dosya, her satırda bir model adı olacak şekilde bir model listesi içermelidir.
- Araç, dizininizdeki ilk 2 dosyayı listedeki **tüm** modelleri kullanarak işler.
- Sonuçlar ve güven puanları, ayrıntılı karşılaştırma için `evaluation.log` dosyasına kaydedilir.

### Önbellek Yönetimi

```bash
# Tüm önbelleği temizle
python main.py --clear-cache

# Belirli dizin için önbelleği temizle
python main.py "C:/Kitaplar/Yolu" --clear-cache
```

---

## ⚠️ Uyarılar ve Sınırlamalar

### Veri Kaybını Önleme

> **⚠️ `rename` veya `organize` modunda renAI çalıştırmadan önce her zaman dosyalarınızın yedeğini alın.**

- Araç dosyaları yeniden adlandırır ve taşır
- Güvenlik kontrolleri dahildir, ancak beklenmeyen sorunlar oluşabilir
- Koleksiyonunuza uygulamadan önce adlandırma kuralını inceleyin
- Önce küçük bir dizinle test edin

### Dizin Yapısı Uyarısı (Organize Modu)

`--mode organize` kullanırken:
- Alt dizinlerdeki tüm dosyalar kök seviyedeki kategori klasörlerine taşınır
- Dosyalar taşındıktan sonra boş alt dizinler kalabilir
- Kategori klasörleri kök seviyede oluşturulur, alt dizinlerin içinde değil
- Birden fazla kez çalıştırmak dosyaları yeniden işler

### Teknik Sınırlamalar

| Sınırlama | Açıklama |
|------------|-------------|
| **Metin Uzunluğu** | Dosya başına LLM'e en fazla 8000 karakter gönderilir |
| **API Maliyetleri** | İşlenen her dosya metni bir LLM API'ye gönderir |
| **Oran Sınırları** | Sağlayıcılar oran sınırları uygulayabilir |
| **OCR Kalitesi** | Taranmış belgeler kaliteli taramalar gerektirir |
| **Dil Desteği** | Türkçe ve İngilizce için optimize edilmiştir |

### OCR İşleme Seçenekleri

- **`--skip-needs-ocr`**: Yetersiz metinli dosyaları `Needs_OCR` klasörüne taşır (OCR'yi atlar)
- **`--move-ocr-failures`**: Tamamen işlenemeyen dosyaları `OCR_Failures` klasörüne taşır

---

## ✅ En İyi Uygulamalar

1. **Düzenli Yedekler**: Orijinal dosyalarınızın her zaman bir yedeğini tutun
2. **Küçük Başlayın**: Kaliteyi doğrulamak için önce 5-10 dosya ile test edin
3. **Maliyetleri İzleyin**: LLM sağlayıcınızla kullanım uyarıları ayarlayın
4. **Önbellekleme Kullanın**: Gerekmedikçe `--update-metadata` kullanmaktan kaçının
5. **Akıllı OCR Seçimi**: Taranmış belgeler için yalnızca `--ocr-enhanced --use-mutool` kullanın
6. **Sağlayıcıları Karşılaştırın**: OpenRouter genellikle rekabetçi fiyatlar sunar
7. **Önbelleği Temizleyin**: Modelleri veya ayarları değiştirdiğinde `--clear-cache` kullanın
8. **Gizliliği Koruyun**: Hassas veya özel belgeler için yerel bir LLM (Ollama, LM Studio) kullanarak `custom` sağlayıcısını tercih edin. Bu, belge içeriğinizi bulut hizmetlerine göndermek yerine makinenizde tutar.

### Önbellek Stratejisi

renAI, sağlayıcı-model farkındalığına sahip akıllı bir önbellek sistemi kullanır:

| Önbellek Türü | Anahtar Bileşenleri | Geçersiz Kılma |
|---------------|---------------------|----------------|
| **Metin Önbelleği** | Dosya hash + OCR ayarları | Dosya içeriği veya OCR modu değiştiğinde |
| **Meta Veri Önbelleği** | Dosya hash + sağlayıcı + model | Dosya içeriği, sağlayıcı veya model değiştiğinde |

**Nasıl çalışır:**
- Her dosya MD5 hash'i ile tanımlanır
- Meta veri önbellek anahtarları sağlayıcı-model çiftinin hash'ini içerir
- Sağlayıcı veya model değiştirmek eski önbelleği otomatik olarak geçersiz kılar
- `--update-metadata` gerektiğinde taze çıkarma zorlar

**Örnek:**
```bash
# DeepInfra ile ilk çalıştırma - meta verileri önbelleğe alır
python main.py "C:/Kitaplar" --provider deepinfra

# OpenRouter ile aynı dosyalar - YENİ önbellek girişi (eski korunur)
python main.py "C:/Kitaplar" --provider openrouter

# Aynı sağlayıcı/model - önbelleğe alınmış meta verileri kullanır
python main.py "C:/Kitaplar" --provider deepinfra

# Zorla yenileme - önbelleği atlar
python main.py "C:/Kitaplar" --provider deepinfra --update-metadata
```

**Önbellek Konumu:** `platformdirs` kullanılarak işletim sistemi tarafından otomatik yönetilir (ör. Linux'ta `~/.cache/renAI`, macOS'te `~/Library/Caches/renAI`, Windows'ta `%LOCALAPPDATA%\renAI\Cache`).

**Önbelleği Temizle:**
```bash
# Tüm önbelleği temizle
python main.py --clear-cache

# Belirli dizin için önbelleği temizle
python main.py "C:/Kitaplar" --clear-cache
```

### Gizlilik Değerlendirmeleri

| Senaryo | Önerilen Sağlayıcı |
|---------|-------------------|
| Genel belgeler | Herhangi bir bulut sağlayıcısı (DeepInfra, OpenRouter, OpenAI) |
| Özel/kişisel dosyalar | Yerel LLM ile `custom` sağlayıcısı |
| Hassas iş belgeleri | Yerel LLM ile `custom` sağlayıcısı |
| Test/deneme | Kolaylık için bulut sağlayıcıları |

Detaylı gizlilik rehberi, veri saklama politikaları ve uyumluluk konuları için bkz. [PRIVACY.md](PRIVACY.md).

### Maliyet Tahmini

| Dosya | Tahmini Maliyet (~$0.001/dosya) |
|-------|--------------------------|
| 100 | 0,10$ |
| 1.000 | 1,00$ |
| 10.000 | 10,00$ |

Gerçek maliyetler sağlayıcı ve modele göre değişir.

---

## 🔧 Yapılandırma

### İşletim Sisteminden Bağımsız (OS-Agnostic) Dosya Yönetimi

renAI artık `platformdirs` kütüphanesi aracılığıyla modern ve işletim sisteminden bağımsız (OS-agnostic) bir yol yönetimi kullanmaktadır.

- **Önbellek Verileri:** İşletim sisteminin önbellek dizininde yönetilir.
- **Log ve Hata Ayıklama (Debug) Verileri:** İşletim sisteminin kullanıcı veri dizininde yönetilir.
- **Yapılandırma (`categories.txt` gibi):** İşletim sisteminin yapılandırma dizininde yönetilir.

### Özel Kategoriler

Kendi işletim sisteminizin `renAI` için ayrılmış yapılandırma dizininde (örneğin Linux'ta `~/.config/renAI/categories.txt`, Windows'ta `%APPDATA%\renAI\categories.txt`) bir `categories.txt` dosyası oluşturun:

```txt
# İngilizce Kategoriler
LANG: en
Article
Fiction
Non-Fiction
Textbook
Report
Gov
Presentation
Manual
Thesis
Research
Other
Dictionary

# Türkçe Kategoriler
LANG: tr
Makale
Kurgu
Kurgu Dışı
Ders Kitabı
Rapor
Devlet
Sunum
Kılavuz
Tez
Araştırma
Diğer
Sözlük
```

### Komut Seçenekleri

| Seçenek | Açıklama |
|--------|-------------|
| `--provider` | `deepinfra`, `openrouter`, `openai` veya `custom` |
| `--extractor` | `pymupdf`, `pypdf`, `pdfplumber`, `pypdfium2` |
| `--workers` | Eşzamanlı çalışanlar (varsayılan: 4) |
| `--ocr-enhanced` | Geliştirilmiş OCR'yi etkinleştir |
| `--use-mutool` | Çıkarma için mutool kullan |
| `--debug-ocr` | Hata ayıklama görüntülerini kaydet |
| `--recursive` | Alt dizinleri işle (varsayılan: True) |
| `--no-recursive` | Yalnızca kök dizini işle |

---

## 🏗️ Mimari

```
renAI/
├── main.py              # Giriş noktası ve CLI
├── config.py            # Yapılandırma ve API ayarları
├── utils.py             # Yardımcı fonksiyonlar
├── llm_engine.py        # LLM sağlayıcı entegrasyonu
├── ocr_engine.py        # OCR işleme
├── file_processor.py    # Çekirdek işleme mantığı
├── extractors/          # Belge ayrıştırıcıları
│   ├── base.py          # Soyut temel sınıf
│   └── pdf_extractors.py # PDF çıkarma uygulamaları
├── test_file_processor.py # Birim testleri
└── requirements.txt     # Bağımlılıklar
```

---

## 🤝 Katkıda Bulunma

Katkılar memnuniyetle karşılanır!

1. Depoyu fork edin
2. Bir özellik dalı oluşturun: `git checkout -b feature/YeniOzellik`
3. Değişiklikleri commit edin: `git commit -m 'YeniOzellik Ekle'`
4. Dalı push edin: `git push origin feature/YeniOzellik`
5. Bir Pull Request açın

---

## 📝 Lisans

MIT Lisansı - Ayrıntılar için [LICENSE](LICENSE) dosyasına bakın.

---

## 🤖 Görüntü Modelleri

Görüntü modu, renAI'nin metin çıkarmadan multimodal yapay zeka modelleri kullanarak belge içeriğini "görmesini" sağlar.

### Resim Yeniden Adlandırma

Yapay zeka kullanarak resim dosyalarını (fotoğraflar, ekran görüntüleri) yeniden adlandırın:

```bash
python main.py "C:/Fotograflar" --mode rename --multimodal-provider deepinfra --multimodal-model meta-llama/Llama-3.2-90B-Vision-Instruct --rename-images
```

Bu komut her resmi analiz edecek ve görsel içeriğine göre yeniden adlandıracaktır.

### Karışık Klasörler (PDF + Resim)

Hem PDF hem de resim içeren klasörleri işleyin:

```bash
python main.py "C:/Belgeler" --mode rename --multimodal-provider deepinfra --multimodal-model meta-llama/Llama-3.2-90B-Vision-Instruct --rename-images
```

Bu komut şunları işler:
- PDF'leri görüntü modeli ile (OCR gerekmez)
- Resimleri görüntü modeli ile

### EXIF Tarih Desteği

Resimleri işlerken, dosya adlarına EXIF tarih bilgisi ekleyebilirsiniz:

```bash
python main.py "C:/Fotograflar" --mode rename --multimodal-provider deepinfra --multimodal-model meta-llama/Llama-3.2-90B-Vision-Instruct --rename-images --exif
```

Bu komut, mevcutsa fotoğrafın EXIF meta verilerinden tarihi dosya adına ekler.

### Önerilen Görüntü Modelleri

| Model | Sağlayıcı | Notlar |
|-------|----------|--------|
| `meta-llama/Llama-3.2-90B-Vision-Instruct` | DeepInfra | Kalite ve hız arasında en iyi denge |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | DeepInfra | Daha hızlı, büyük partiler için iyi |
| `openai/gpt-4o` | OpenAI | En yüksek kalite, daha pahalı |
| `google/gemini-1.5-pro` | OpenRouter | İyi alternatif |

### Görüntü Modu vs OCR

| Özellik | Görüntü Modu | OCR Modu |
|---------|-------------|----------|
| Hız | Daha hızlı (metin çıkarma yok) | Daha yavaş |
| Doğruluk | Modele bağlı | Temiz metin için iyi |
| Maliyet | Dosya başına API çağrısı | Dosya başına API çağrısı |
| En iyi için | Taranmış PDF'ler, resimler | Metin tabanlı PDF'ler |

---

## 📧 Destek

- **Sorunlar**: GitHub'da sorun açın
- **Tartışmalar**: GitHub Discussions kullanın
- **E-posta**: Bakımcılarla iletişime geçin

---

## 🧪 Test Etme

Test paketini çalıştırın:

```bash
# Tüm testleri çalıştır
pytest test_file_processor.py -v

# Kapsam ile çalıştır
pytest test_file_processor.py --cov=file_processor
```

---

*renAI - Dijital kütüphanenizi yapay zeka ile düzenleyin*
