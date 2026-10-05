# renAI: Yapay Zeka Destekli Belge ve Görsel Düzenleyici

`renAI`, dijital belge ve görsellerinizi Büyük Dil Modelleri (LLM) ve Optik Karakter Tanıma (OCR) kullanarak otomatik olarak yeniden adlandıran ve düzenleyen bir Python aracıdır. Dosyalarınızdan metin çıkarma veya multimodal yapay zeka görüşü kullanarak anlamlı meta veriler (metadata) çıkarır, tutarlı ve aranabilir dosya isimleri oluşturur.

Son güncelleme: 2026-10-05

---

## 📖 Giriş

Dijital dosya koleksiyonlarınız anlamsız veya tutarsız isimlerle geldiğinde yönetimi zorlaşabilir. renAI, dosyalarınızı akıllıca analiz ederek açıklayıcı ve düzenli dosya isimleri oluşturur.

**İki Güçlü Yaklaşım:**

- **Metin Çıkarma Modu**: Geleneksel metin tabanlı LLM'ler kullanarak belge içeriğinden (PDF, EPUB, DOCX, vb.) başlık, yazar ve yayın yılı gibi bilgileri çıkarır.
- **Multimodal Görüş Modu**: Görüş yeteneğine sahip LLM'ler kullanarak belgeleri ve görselleri "görür". Metin çıkarma başarısız olursa, taranmış PDF'leri OCR yerine vision model ile işler.

İster taranmış PDF'lerle, ister dijital belgelerle veya fotoğraflarla uğraşıyor olun, renAI size en iyi sonuçları verecek şekilde uyum sağlar.

Araç hem bulut tabanlı LLM'leri (DeepInfra, OpenRouter, OpenAI) hem de yerel çalışan modelleri (Ollama, LM Studio) destekler, böylece kullanım kolaylığı ve gizlilik arasında seçim yapabilirsiniz.

### İsimlendirme Standardı

**Belgeler (Metin Çıkarma Modu):**

```text
Başlık -- Alt Başlık -- Baskı -- Yazar -- (Yıl).uzantı
```

_Örnek: `Introduction to Machine Learning -- 3rd Edition -- Andrew Ng -- (2018).pdf`_

**Belgeler (Multimodal Görüş Modu):**

```text
Başlık -- Yazar -- (Yıl) -- (Dil).uzantı
```

_Örnek: `Deep Learning Research -- Ian Goodfellow -- (2016) -- (en).pdf`_

**Görseller (Multimodal Görüş Modu):**

```text
Açıklama -- [Tarih] -- (Dil).uzantı
```

_Örnek: `Sunset -- [2024-07-15] -- (en).jpg`_

---

## ✨ Özellikler

| Özellik                           | Açıklama                                                                                            |
| :-------------------------------- | :-------------------------------------------------------------------------------------------------- |
| **Çoklu Format Desteği**          | PDF, EPUB, MOBI, DOCX, PPTX, XLSX, TXT, MD ve görüntü dosyalarını (JPG, PNG, HEIC vb.) işler        |
| **Çoklu PDF Çıkarıcıları**        | PyMuPDF, pdfplumber, pypdf veya pypdfium2 arasından seçim yapın                                     |
| **Yapay Zeka Destekli Meta Veri** | Başlıkları, yazarları, yılları ve kategorileri çıkarmak için LLM'leri kullanır                      |
| **Multimodal Modu**               | Önce metin çıkarmayı dener; başarısız olursa PDF'leri OCR yerine vision model ile işle (daha doğru) |
| **Görüntü Yeniden Adlandırma**    | Yapay zeka görüntü analizi ile görüntü dosyalarını yeniden adlandır                                 |
| **Gelişmiş OCR**                  | Düşük kaliteli taramalar için Enhanced Mode ile Tesseract entegrasyonu                              |
| **mutool Desteği**                | Sabit düzenli PDF'lerden yüksek kaliteli görüntüler çıkarır                                         |
| **Akıllı Önbellekleme**           | SHA256 tabanlı önbellekleme sistemi (Hem metin hem metadata için ayrı katmanlar).                   |
| **AsyncIO-First**                 | Bloklamayan I/O ve yüksek eşzamanlılık için `asyncio` kullanan tamamen asenkron mimari              |
| **Async RPM Sınırı**              | `AsyncRateLimiter` ile eşzamanlı görevlerde sağlayıcı limitlerine tam uyum                          |
| **Ruff Desteği**                  | Kod stili tutarlılığı için ultra hızlı linting ve formatlama                                        |
| **SSoT Mimarisi**                 | Sıkı tipli Pydantic modelleri kullanarak meta veriler için Tek Gerçek Kaynak (SSoT)                 |
| **Modüler Promptlar**             | Dinamik JSON şema enjeksiyonuna sahip merkezi ve yeniden kullanılabilir prompt bileşenleri          |
| **Bütünleşik İyileştirme**        | Hem metin hem de görsel modları için standartlaştırılmış ikinci geçiş 'Kıdemli Editör' mantığı      |
| **Token Optimizasyonu**           | Bağlamdan bağımsız iyileştirme geçişi ile ikincil API maliyetlerinde ~%90 tasarruf                  |
| **Özel Kategoriler**              | `categories.toml` dosyasında dil başına kendi kategorilerinizi tanımlayın                           |

### Çalışma Modları

- **`rename`**: Çıkarılan meta verilere göre dosyaları yeniden adlandırır
- **`organize`**: Yeniden adlandırılmış dosyaları kategori klasörlerine taşır
- **`benchmark`**: PDF çıkarma kütüphanesi performansını karşılaştırır
- **`evaluate`**: Farklı LLM modellerini ve sağlayıcılarını doğruluk açısından test eder
- **`--fallback-mode ocr`**: Önce metin çıkarmayı dener. Başarısız olursa Tesseract OCR kullanarak metin çıkarmaya çalışır (varsayılan davranış).
- **`--fallback-mode vision`**: Görüntü yetenekli LLM ile PDF'leri işler. Önce metin çıkarmayı dener (OCR kullanmaz). Başarısız olursa, vision modele fallback yapar.
- **`--fallback-mode move`**: Metin çıkarma başarısız olan dosyaları `Needs_Scan/` klasörüne taşır.
- **`--rename-images`**: Görüntü dosyalarını yeniden adlandırır. Ayrıca görüntü olmayan dosyaları metin çıkarma + fallback ile işler.
- **`--dry-run`**: Dosyaları yeniden adlandırmadan veya taşımadan işlemi simüle eder.
- **`undo` / `undo-last`**: En son yeniden adlandırma veya düzenleme işlemlerini geri alır.

---

## 🛠️ Kurulum

### Ön Gereksinimler

1. **Python 3.12+**
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

# uv kullanarak yükle
uv sync --all-extras

# renAI'yi Hazırla (İnteraktif Sihirbaz)
# Bu komut sağlayıcı ayarlarını yapar ve gerekli yapılandırma dosyalarını oluşturur
uv run renai init
```

### Kurulum & API Anahtarları

renAI, ilk kurulum için interaktif bir sihirbaz kullanır. Şu komutu çalıştırın:

```bash
uv run renai init
```

Bu komut, işletim sisteminize uygun dizinde yapılandırma dosyalarınızı oluşturacaktır:

- **Windows**: `%LOCALAPPDATA%\renAI`
- **Linux**: `~/.config/renAI`
- **macOS**: `~/Library/Application Support/renAI`

Sihirbaz şu konularda size yardımcı olur:

- **LLM Sağlayıcısı**: DeepInfra (varsayılan), OpenRouter, OpenAI veya Özel
- **API Anahtarları**: `secrets.toml` dosyasında güvenli bir şekilde saklanır (sihirbaz kurulum sırasında anahtarınızı ilgili sağlayıcının API ucu üzerinden otomatik olarak doğrular, yetkilendirme hatası olması durumunda düzeltmeniz için tekrar sorar).
- **Genel Ayarlar**: Eşzamanlı çalışan sayısı, fallback stratejileri vb.

### Yapılandırma Hiyerarşisi

renAI sağlam ve katmanlı bir yapılandırma sistemi kullanır:

1. **Kullanıcı Geçersiz Kılmaları**: Kullanıcı yapılandırma dizininizdeki `settings.toml`.
2. **Fiziksel Varsayılanlar**: `settings.default.toml` (otomatik oluşturulan referans).
3. **Dahili Varsayılanlar**: Sıfır kurulum operasyonu için paket içine gömülü temel ayarlar.

#### Önemli Dosyalar

- `settings.toml`: Uygulama davranışını özelleştirin ve sağlayıcı varsayılanlarını ezin.
- `secrets.toml`: Hassas API anahtarları. **Bu dosyayı asla paylaşmayın!**
- `providers.toml`: Özel LLM sağlayıcılarını kaydedin veya uç noktaları (endpoint) değiştirin.
- `categories.toml`: Klasörleme düzeni için kendi kategorilerinizi tanımlayın (`--mode organize`).
- `eval_models.toml`: `evaluate` komutu ile test edilecek modelleri listeleyin.

Anahtarları şuradan alın: [DeepInfra](https://deepinfra.com) | [OpenRouter](https://openrouter.ai) | [OpenAI](https://platform.openai.com/api-keys)

### Ortam Değişkenleri

renAI'yi ortam değişkenlerini (`RENAI_` ön eki ile) kullanarak da yapılandırabilirsiniz:

- `RENAI_PROVIDER`: örn. `openai`
- `RENAI_MODEL`: örn. `gpt-4o`
- `RENAI_WORKERS`: örn. `8`

### Sıfır Kodlu Varsayılanlar (Zero-Code Defaults)

renAI, "Sıfır Kodlu Varsayılanlar" felsefesini izler. Başlamak için yapılandırma dosyalarını düzenlemenize _gerek yoktur_. Temel fabrika varsayılanları paketle birlikte gelir, böylece basit bir ilklendirmeden sonra "direkt çalışır".

Özelleştirme seçenekleri hiyerarşisi:

1. Ortam değişkenleri (`RENAI_` ön eki ile)
2. Kullanıcı dizininizdeki `settings.toml` dosyası
3. `settings.default.toml` (Global temel ayarlar)
4. CLI parametreleri (en yüksek öncelik)

---

## 📖 Kullanım Örnekleri

### Temel Yeniden Adlandırma

```bash
# Dizindeki tüm dosyaları yeniden adlandır
uv run renai process "C:/Kitaplar/Yolu" --mode rename
```

### Klasörlere Düzenleme

```bash
# Yeniden adlandır ve kategori klasörlerine düzenle
renai process "C:/Kitaplar/Yolu" --mode organize
```

### Model Değerlendirme (Evaluation)

En iyi sonuç veren modeli bulmak için birden fazla LLM sağlayıcısını ve modelini örnek dosyalarınız üzerinde test edin. Sonuçlar uygulama veri dizininizdeki `evaluation.log` dosyasına kaydedilir.

```bash
# eval_models.toml içinde tanımlanan TÜM sağlayıcıları ve modelleri test et
renai evaluate "C:/Kitaplar/Ornekler"

# Sadece belirli bir sağlayıcıyı test et
renai evaluate "C:/Kitaplar/Ornekler" --provider deepinfra

# Özel iş parçacığı (workers) sayısı ile test et
renai evaluate "C:/Kitaplar/Ornekler" --workers 8
```

### Taranmış Belgeler için Yüksek Kaliteli OCR

```bash
# Görüntü ön işleme ile geliştirilmiş OCR
renai process "C:/Kitaplar/Yolu" --ocr-enhanced --use-mutool
```

### Multimodal Modu (OCR'suz PDF)

Vision modu akıllı bir fallback stratejisi uygular: Önce (daha hızlı ve ucuz olan) metin çıkarmayı dener. Başarısız olursa, PDF'i bir görsel olarak işlemek için görüş yeteneğine sahip bir LLM'e başvurur.

```bash
# Vision model ile PDF'leri işle (OCR yok) - varsayılan provider/model kullanır
renai process "C:/Kitaplar/Yolu" --fallback-mode vision

# Belirli vision model kullan
renai process "C:/Kitaplar/Yolu" --fallback-mode vision --provider openrouter --model openai/gpt-4o

# DeepInfra ile Llama Vision
renai process "C:/Kitaplar/Yolu" --fallback-mode vision --provider deepinfra --model meta-llama/Llama-3.2-90B-Vision-Instruct

# OpenAI GPT-4o
renai process "C:/Kitaplar/Yolu" --fallback-mode vision --provider openai --model gpt-4o
```

### Görüntü Dosyalarını Yeniden Adlandırma

Yapay zeka görüntü analizi kullanarak görüntü dosyalarını yeniden adlandırın. Görüntü yetenekli bir model belirtilmesi zorunludur:

```bash
# Görüntü modeli ile görüntüleri yeniden adlandır
renai process "C:/Fotoğraflar" --rename-images --provider openai --model gpt-4o

# EXIF tarih ekle (varsa)
renai process "C:/Fotoğraflar" --fallback-mode vision --rename-images --provider openai --model gpt-4o --exif

# DeepInfra ile
renai process "C:/Görüntüler" --fallback-mode vision --rename-images --provider deepinfra --model Qwen/Qwen2-VL-72B-Instruct
```

### Karışık Klasörler (PDF + Görüntü)

Aynı klasördeki PDF ve görüntü dosyalarını işleyin:

| Parametre                                | PDF/Desteklenen          | Resim                          |
| :--------------------------------------- | :----------------------- | :----------------------------- |
| `--fallback-mode vision`                 | Normal → Vision fallback | ⏭️ Atlanır                     |
| `--fallback-mode vision --rename-images` | Normal → Vision fallback | 🔄 Vision ile yeniden adlandır |
| `--rename-images`                        | Normal → OCR fallback    | 🔄 Vision ile yeniden adlandır |

```bash
# PDF'ler vision ile, görüntüler vision ile yeniden adlandırılır
renai process "C:/Karışık" --fallback-mode vision --rename-images --provider openai --model gpt-4o

# Sadece görüntüleri yeniden adlandır (PDF'ler OCR ile)
renai process "C:/Karışık" --rename-images --provider openai --model gpt-4o

# Sadece PDF'leri vision ile işle (görüntüler atlanır)
renai process "C:/PDFler" --fallback-mode vision --provider openai --model gpt-4o
```

### Farklı LLM Sağlayıcıları Kullanma

```bash
# DeepInfra (varsayılan)
renai process "C:/Kitaplar/Yolu" --mode rename --provider deepinfra

# OpenRouter
renai process "C:/Kitaplar/Yolu" --mode rename --provider openrouter

# OpenAI
renai process "C:/Kitaplar/Yolu" --mode rename --provider openai
```

### Yerel Çıkarım (Ollama / LM Studio)

`custom` sağlayıcısını kullanarak [Ollama](https://ollama.com/) veya [LM Studio](https://lmstudio.ai/) gibi yerel çıkarım sunucularına bağlanabilirsiniz. Bu araçlar OpenAI uyumlu API sağlar.

> **⚠️ Önemli: Tüm modeller yapılandırılmış çıktı üretemez, özellikle 7B parametrenin altındaki LLM'ler. Modelin yapılandırılmış çıktıyı destekleyip desteklemediğinden emin değilseniz model kartı README'sini kontrol edin.**
>
> renAI güvenilir meta veri çıkarma için LLM'den JSON çıktısı gerektirir. 7B'den daha az parametreye sahip modeller yapılandırılmış çıktı formatlamada zorlanabilir.

```bash
# Ortam değişkenlerini ayarla (Windows PowerShell veya .env dosyası)
$env:CUSTOM_API_BASE_URL = "http://localhost:11434/v1"  # Ollama varsayılanı
$env:CUSTOM_MODEL = "llama3"  # Model adınız

# Özel sağlayıcı ile çalıştır
uv run renai process "C:/Kitaplar/Yolu" --mode rename --provider custom

# Veya modeli doğrudan geçersiz kıl
uv run renai process "C:/Kitaplar/Yolu" --mode rename --provider custom --model mistral
```

#### Çıkarım Uzunluğu ve Örnekleme (Sampling) Kontrolü

Tüketici donanımlarında yerel modeller çalıştırırken veya token tasarrufu sağlamak için LLM'e gönderilen metin uzunluğunu ve okuma mantığını ayarlayabilirsiniz:

```bash
# Metin uzunluğunu 5000 karaktere azalt (varsayılan: 10000)
uv run renai process "C:/Kitaplar/Yolu" --mode rename --provider custom --text-length 5000

# Modeliniz kaldırabiliyorsa daha iyi bağlam için artırın
uv run renai process "C:/Kitaplar/Yolu" --mode rename --provider custom --text-length 20000

# Düzenli (regular) mod: Sadece ilk 5 sayfayı sırayla okur, head-tail bölme yapmaz
uv run renai process "C:/Kitaplar/Yolu" --text-extraction regular --extraction-pages 5
```

| Seçenek              | Açıklama                                                    | Varsayılan  |
| :------------------- | :---------------------------------------------------------- | :---------- |
| `--text-length`      | LLM'e gönderilen hedef metin uzunluğu                       | 10000       |
| `--text-extraction`  | Metin çıkarma modu (`head-tail` veya `regular`)             | `head-tail` |
| `--extraction-pages` | Belgenin her bir tarafından işlenecek maksimum sayfa sayısı | 10          |

#### Head-Tail Örnekleme (Sampling)

Varsayılan `head-tail` modu, belgenin başından (head) ve sonundan (tail) prosedürel olarak sayfaları okur. Bu, başlangıçtaki meta veri açısından zengin içeriği (başlık, yazar, telif hakkı) ve sondaki referansları/ISBN'yi yakalarken, meta veri açısından fakir olan orta bölümü atlar.

- **`head_ratio`** (varsayılan: 0.7): Head/tail karakter bölme oranını kontrol eder. 0.7, hedef metnin %70'inin head'den, %30'unun tail'den geldiği anlamına gelir.
- **`"mixed"`** değeri geriye dönük uyumluluk için sessizce `"head-tail"` olarak normalize edilir.

#### Özel Sağlayıcı için Ortam Değişkenleri

| Değişken              | Açıklama                                                | Varsayılan                 |
| :-------------------- | :------------------------------------------------------ | :------------------------- |
| `CUSTOM_API_BASE_URL` | Yerel API sunucunuzun temel URL'si                      | `http://localhost:8000/v1` |
| `CUSTOM_MODEL`        | Kullanılacak model adı                                  | `llama3`                   |
| `CUSTOM_API_KEY`      | API anahtarı (yerel sunucular için genellikle gerekmez) | `not-required`             |

#### Yaygın Yerel Sunucu URL'leri

| Araç                  | URL                         |
| :-------------------- | :-------------------------- |
| Ollama                | `http://localhost:11434/v1` |
| LM Studio             | `http://localhost:1234/v1`  |
| LocalAI               | `http://localhost:8080/v1`  |
| Text Generation WebUI | `http://localhost:5001/v1`  |

### Özel Modeller

```bash
# DeepInfra ile özel model
uv run renai process "C:/Kitaplar/Yolu" --mode rename --model Qwen/Qwen2.5-72B-Instruct

# OpenRouter ile Gemini
uv run renai process "C:/Kitaplar/Yolu" --mode rename --provider openrouter --model google/gemini-2.0-flash-exp:free
```

### Daha Hızlı İşleme

```bash
# Eşzamanlı çalışanları artır
uv run renai process "C:/Kitaplar/Yolu" --mode rename --workers 8
```

### Meta Verileri Zorla Yenileme

```bash
# Önbelleği atla ve yeniden çıkar
uv run renai process "C:/Kitaplar/Yolu" --mode rename --update-metadata
```

### Kıyaslama ve Model Değerlendirme

#### PDF Çıkarıcı Kıyaslaması

Farklı PDF çıkarma kütüphanelerinin belgeleriniz üzerindeki performansını ve doğruluğunu karşılaştırın.

```bash
uv run renai process "C:/Kitaplar/Yolu" --benchmark
```

#### Gelişmiş Model Değerlendirme

`evaluate` komutu rekabetçi bir test gerçekleştirir:

1. Dizininizdeki ilk birkaç dosyayı alır.
2. Bunları `eval_models.toml` dosyanızda listelenen **her** modeli kullanarak işler.
3. Her deneme için çıkarılan meta verileri ve bir güven puanını günlüğe kaydeder.

**Yapılandırma:**
Modeller `eval_models.toml` dosyasından yüklenir (Kullanıcı > Varsayılan > Dahili).

Örnek `eval_models.toml`:

```toml
[deepinfra]
models = ["meta-llama/Llama-3.2-90B-Vision-Instruct", "meta-llama/Llama-3.1-405B-Instruct"]

[openai]
models = ["gpt-4o", "gpt-4o-mini"]
```

### Önbellek Yönetimi

Önbellek dosyanın SHA256 karmasına göre otomatik yönetilir. Meta veri önbellek girdileri sağlayıcı-model çiftlerine göre kapsamlıdır, bu nedenle sağlayıcı veya model değiştirmek otomatik olarak farklı bir önbellek girişi kullanır.

```bash
# Tüm önbelleği manuel olarak temizle (metin, metadata ve ocr_debug)
renai cache --yes
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

| Sınırlama           | Açıklama                                                     |
| :------------------ | :----------------------------------------------------------- |
| **Metin Uzunluğu**  | `--text-length` ile yapılandırılabilir (örn. 10000 karakter) |
| **API Maliyetleri** | İşlenen her dosya metni bir LLM API'ye gönderir              |
| **Oran Sınırları**  | Sağlayıcılar oran sınırları uygulayabilir                    |
| **OCR Kalitesi**    | Taranmış belgeler kaliteli taramalar gerektirir              |
| **Dil Desteği**     | Türkçe ve İngilizce için optimize edilmiştir                 |

### OCR İşleme Seçenekleri

- **`--fallback-mode move`**: Yetersiz metinli dosyaları `Needs_Scan` klasörüne taşır (işlemeyi atlar)
- **`--move-scan-failures`**: Hem standart hem de fallback yöntemleriyle işlenemeyen dosyaları `Scan_Failures` klasörine taşır

---

## 🤖 Görüntü Modelleri

renAI, PDF'leri ve görüntüleri doğrudan işleyebilen görüntü yetenekli LLM'leri destekler. OCR'ye gerek yoktur.

### Önerilen Görüntü Modelleri

| Sağlayıcı      | Model                                      | En İyi İçin                |
| :------------- | :----------------------------------------- | :------------------------- |
| **OpenAI**     | `gpt-4o`                                   | En iyi doğruluk, güvenilir |
| **OpenAI**     | `gpt-4o-mini`                              | Hızlı, uygun maliyetli     |
| **OpenRouter** | `openai/gpt-4o`                            | Esnek fiyatlandırma        |
| **OpenRouter** | `google/gemini-1.5-flash`                  | Ücretsiz katman mevcut     |
| **DeepInfra**  | `meta-llama/Llama-3.2-90B-Vision-Instruct` | Açık kaynak, güçlü         |
| **DeepInfra**  | `Qwen/Qwen2-VL-72B-Instruct`               | Mükemmel değer             |
| **Özel**       | `llava`, `qwen-vl`                         | Yerel çıkarım              |

### Vision Modu vs OCR

| Özellik               | Vision Modu (`--fallback-mode vision`)       | OCR Modu (varsayılan)                     |
| :-------------------- | :------------------------------------------- | :---------------------------------------- |
| **Metin Çıkarma**     | Önce dener, başarısız olursa vision fallback | Önce dener, başarısız olursa OCR fallback |
| **Hız**               | Metin PDF'leri için daha hızlı               | OCR daha yavaş                            |
| **Doğruluk**          | Temiz metin için daha yüksek                 | Tarama kalitesine bağlı                   |
| **Taranmış Belgeler** | Vision ile işler                             | Taramalar için optimize                   |
| **Diller**            | Model ile sınırlı                            | Türkçe+İngilizce optimize                 |
| **API Maliyeti**      | Görüntü tokenleri                            | Metin tokenleri                           |

**Görüntü Modu ne zaman kullanılır:**

- Metin tabanlı PDF'ler (kitaplar, makaleler)
- Tesseract kurulu olmadığında
- OCR çok yavaş olduğunda
- Temiz belgelerde daha iyi doğruluk istendiğinde

**OCR ne zaman kullanılır:**

- Taranmış belgeler
- Düşük kaliteli görüntüler
- Çok dilli belgeler
- Türkçe karakter desteği gerektiğinde

---

## ✅ En İyi Uygulamalar

1. **Düzenli Yedekler**: Orijinal dosyalarınızın her zaman bir yedeğini tutun
2. **Küçük Başlayın**: Kaliteyi doğrulamak için önce 5-10 dosya ile test edin
3. **Maliyetleri İzleyin**: LLM sağlayıcınızla kullanım uyarıları ayarlayın
4. **Önbellekleme Kullanın**: Gerekmedikçe `--update-metadata` kullanmaktan kaçının (önbellek sağlayıcı/model'e göre otomatik kapsamlanır)
5. **Akıllı OCR Seçimi**: Taranmış belgeler için yalnızca `--ocr-enhanced --use-mutool` kullanın
6. **Sağlayıcıları Karşılaştırın**: OpenRouter genellikle rekabetçi fiyatlar sunar
7. **Gizliliği Koruyun**: Hassas veya özel belgeler için yerel bir LLM (Ollama, LM Studio) kullanarak `custom` sağlayıcısını tercih edin. Bu, belge içeriğinizi bulut hizmetlerine göndermek yerine makinenizde tutar.

### Önbellek Stratejisi

renAI, performansı ve güvenilirliği artırmak için SQLite WAL (Write-Ahead Logging) modu ile desteklenen akıllı bir önbellek sistemi kullanır. Meta veri önbellek girdileri sağlayıcı-model çiftlerine göre kapsamlıdır, bu nedenle sağlayıcı veya model değiştirmek otomatik olarak farklı bir önbellek girişi kullanır.

| Önbellek Türü           | Anahtar Bileşenleri                               | Geçersiz Kılma                                                                     |
| :---------------------- | :------------------------------------------------ | :--------------------------------------------------------------------------------- |
| **Metin Önbelleği**     | Dosya hash + OCR ayarları                         | **Dosya İçeriği**: Dosyanın SHA256 karması değiştiğinde önbellek geçersiz kılınır. |
| **Meta Veri Önbelleği** | Dosya hash + sağlayıcı + model + şema doğrulaması | Dosya içeriği, sağlayıcı, model veya şema değiştiğinde                             |

**Nasıl çalışır:**

- Her dosya SHA256 hash'i ile tanımlanır
- Meta veri önbellek anahtarları sağlayıcı-model çiftinin hash'ini içerir
- Sağlayıcı veya model değiştirmek eski önbelleği otomatik olarak geçersiz kılar
- `--update-metadata` gerektiğinde taze çıkarma zorlar
- SQLite WAL modu, eşzamanlı okuma/yazma işlemlerini güvenli ve performanslı hale getirir

**Örnek:**

```bash
# DeepInfra ile ilk çalıştırma - meta verileri önbelleğe alır
uv run renai process "C:/Kitaplar" --provider deepinfra

# OpenRouter ile aynı dosyalar - YENİ önbellek girişi (eski korunur)
uv run renai process "C:/Kitaplar" --provider openrouter

# Aynı sağlayıcı/model - önbelleğe alınmış meta verileri kullanır
uv run renai process "C:/Kitaplar" --provider deepinfra

# Zorla yenileme - önbelleği atlar
uv run renai process "C:/Kitaplar" --provider deepinfra --update-metadata
```

**Önbellek Konumu:** `platformdirs` kullanılarak işletim sistemi tarafından otomatik yönetilir (ör. Linux'ta `~/.cache/renAI`, macOS'te `~/Library/Caches/renAI`, Windows'ta `%LOCALAPPDATA%\renAI\Cache`). İçerde SQLite WAL modu ile eşzamanlı güvenli, yüksek performanslı önbellekleme kullanılır.

---

### Gizlilik Değerlendirmeleri

| Senaryo               | Önerilen Sağlayıcı                                             |
| :-------------------- | :------------------------------------------------------------- |
| Genel belgeler        | Herhangi bir bulut sağlayıcısı (DeepInfra, OpenRouter, OpenAI) |
| Özel/kişisel dosyalar | Yerel LLM ile `custom` sağlayıcısı                             |
| Hassas iş belgeleri   | Yerel LLM ile `custom` sağlayıcısı                             |
| Test/deneme           | Kolaylık için bulut sağlayıcıları                              |

Detaylı gizlilik rehberi, veri saklama politikaları ve uyumluluk konuları için bkz. [PRIVACY.md](docs/PRIVACY.md).

### Maliyet Tahmini

| Dosya  | Tahmini Maliyet (~$0.001/dosya) |
| :----- | :------------------------------ |
| 100    | 0,10$                           |
| 1.000  | 1,00$                           |
| 10.000 | 10,00$                          |

Gerçek maliyetler sağlayıcı ve modele göre değişir.

---

## 🔧 Yapılandırma

### İşletim Sisteminden Bağımsız (OS-Agnostic) Dosya Yönetimi

renAI artık `platformdirs` kütüphanesi aracılığıyla modern ve işletim sisteminden bağımsız (OS-agnostic) bir yol yönetimi kullanmaktadır.

- **Önbellek Verileri**: Windows'ta `%LOCALAPPDATA%\renAI\Cache` veya Linux'ta `~/.cache/renAI` dizininde yönetilir.
- **Log ve Hata Ayıklama Verileri**: İşletim sisteminin kullanıcı veri dizininde yönetilir.
- **Yapılandırma**: Düz işletim sistemi yapılandırma dizinindeki (iç içe klasörler olmadan) `settings.toml`, `secrets.toml` ve `categories.toml` dosyaları aracılığıyla yönetilir.

### Özel Kategori Tanımlama

Kendi işletim sisteminizin `renAI` için ayrılmış yapılandırma dizininde (örneğin Linux'ta `~/.config/renAI/categories.toml`, Windows'ta `%APPDATA%\renAI\categories.toml`) bir `categories.toml` dosyası oluşturun:

```toml
[en]
categories = [
    "Article", "Fiction", "Non-Fiction", "Textbook", "Report",
    "Gov", "Presentation", "Manual", "Thesis", "Research",
    "Other", "Dictionary", "Magazine"
]

[tr]
categories = [
    "Makale", "Kurgu", "Kurgu Dışı", "Ders Kitabı", "Rapor",
    "Devlet", "Sunum", "Kılavuz", "Tez", "Araştırma",
    "Diğer", "Sözlük", "Dergi"
]
```

### Komut Seçenekleri

| Seçenek                | Açıklama                                                                 |
| :--------------------- | :----------------------------------------------------------------------- |
| `--provider`           | `deepinfra`, `openrouter`, `openai` veya `custom`                        |
| `--model`              | Metin işleme veya vision modunda kullanılacak model adı                  |
| `--workers`            | Eşzamanlı çalışanlar (varsayılan: 4)                                     |
| `--ocr-enhanced`       | Geliştirilmiş OCR'yi etkinleştir                                         |
| `--use-mutool`         | Çıkarma için mutool kullan                                               |
| `--debug-ocr`          | Hata ayıklama görüntülerini kaydet                                       |
| `--recursive`          | Alt dizinleri işle (varsayılan: True)                                    |
| `--text-length`        | Hedef metin uzunluğu (varsayılan: 10000)                                 |
| `--text-extraction`    | Metin okuma modu (`head-tail` veya `regular`) (varsayılan: `head-tail`)  |
| `--extraction-pages`   | Doküman başına işlenecek sayfa sayısı                                    |
| `--update-metadata`    | Önbelleği zorla yenile                                                   |
| `--fallback-mode`      | Çıkarma başarısız olduğundaki strateji (`none`, `ocr`, `vision`, `move`) |
| `--move-scan-failures` | Başarısız dosyaları `Scan_Failures` klasörüne taşı                       |
| `--benchmark`          | Tespit edilen tüm dosyalarda PDF çıkarıcılarını karşılaştır              |
| `--vision-max-pages`   | Vision PDF çıkarma için örneklenecek maksimum sayfa (varsayılan: 3)       |
| `--vision-provider`    | Vision işlemleri için LLM sağlayıcısı (varsayılan: ayarlardan)            |
| `--vision-model`       | Vision işlemleri için vision destekli model (varsayılan: ayarlardan)      |
| `--dry-run`            | Dosyaları yeniden adlandırmadan veya taşımadan işlemi simüle eder         |
| `undo` / `undo-last`   | En son yeniden adlandırma veya düzenleme işlemlerini geri alır            |
| `cache`                | Önbelleği temizleme komutu                                               |
| `init`                 | İnteraktif kurulum sihirbazını başlatma komutu                            |
| `providers`            | Yapılandırılmış sağlayıcıları listeleme komutu                            |
| `--version`            | renAI versiyonunu göster                                                 |

---

## 🏗️ Mimari

````text
├── src/
│   └── renai/
│       ├── cli.py               # CLI giriş noktası (Typer)
│       ├── config/              # Yapılandırma ve Pydantic ayarları yükleme
│       │   ├── settings.py      # Pydantic ayarları yönetimi
│       │   ├── settings.default.toml
│       │   ├── providers.py     # Sağlayıcı kaydı
│       │   └── constants.py     # Önbellek ve uygulama dizini sabitleri
│       ├── cache/               # [YENİ] Atomik yazma ile önbellek yönetimi
│       │   ├── manager.py       # Metin ve metadata önbellek işlemleri
│       │   └── __init__.py
│       ├── messages/            # [YENİ] Merkezi kullanıcı mesajları
│       │   ├── cli.py           # CLI uyarıları, komut istemleri, yardım metinleri
│       │   ├── orchestrator.py  # İşlem hattı durumu, ilerleme, özet
│       │   ├── setup.py         # Sihirbaz, uçuş öncesi, kayıt mesajları
│       │   └── errors.py        # İstisna analizi (neden/nasıl düzeltilir)
│       ├── prompts/             # Modüler .txt prompt şablonları
│       ├── models/              # SSoT Pydantic Şemaları (Book, Image)
│       ├── engines/             # Görüş ve OCR motorları
│       ├── extractors/          # Belge ayrıştırıcıları
│       ├── processors/          # Çekirdek işleme mantığı
│       │   ├── book.py
│       │   ├── orchestrator.py  # Pipeline orkestrasyonu
│       │   ├── setup.py         # [YENİ] İnteraktif Kurulum Sihirbazı
│       │   ├── text_extractor.py
│       │   └── visual.py
│       └── ...
├── tests/                       # Asenkron birim testleri (pytest-asyncio)
├── pyproject.toml               # Paket meta verileri ve bağımlılıklar
└── ...
```text

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

### Gelişmiş Yapılandırma (Katmanlı Mimari)

renAI, maksimum esneklik için katmanlı bir yapılandırma hiyerarşisi kullanır:

1. **CLI Parametreleri**: En yüksek öncelik (örn: `--workers 8`).
2. **Ortam Değişkenleri**: `RENAI_` ön eki ile (örn: `RENAI_PROVIDER=openai`).
3. **Kullanıcı Ayarları**: `settings.toml` (İşletim sistemi ayar klasörünüzde saklanır).
4. **Yerel Varsayılanlar**: `settings.default.toml` (`init` sırasında oluşturulan fiziksel kopya).
5. **Fabrika Varsayılanları**: Paket içindeki gömülü temel ayarlar.

Yeniden adlandırma davranışları, OCR ayarları ve diğer birçok parametre `settings.toml` dosyasından ayarlanabilir. Mevcut tüm ayarların listesi ve açıklamaları için yapılandırma dizininizdeki `settings.default.toml` dosyasına bakabilirsiniz.

---

## 🖼️ Gelişmiş Görüş İş Akışları

Görüntü modu, renAI'nin metin çıkarmadan multimodal yapay zeka modelleri kullanarak belge içeriğini "görmesini" sağlar.

### Doğrudan Resim Yeniden Adlandırma

Yapay zeka kullanarak resim dosyalarını (fotoğraflar, ekran görüntüleri) yeniden adlandırın:

```bash
renai process "C:/Fotograflar" --mode rename --fallback-mode vision --provider deepinfra --model meta-llama/Llama-3.2-90B-Vision-Instruct --rename-images
````

Bu komut her resmi analiz edecek ve görsel içeriğine göre yeniden adlandıracaktır.

### Karışık Klasörleri İşleme Stratejisi

Hem PDF hem de resim içeren klasörleri işleyin:

```bash
renai process "C:/Belgeler" --mode rename --fallback-mode vision --provider deepinfra --model meta-llama/Llama-3.2-90B-Vision-Instruct --rename-images
```

Bu komut şunları işler:

- PDF'leri görüntü modeli ile (OCR gerekmez)
- Resimleri görüntü modeli ile

### EXIF Tarih Desteği

Resimleri işlerken, dosya adlarına EXIF tarih bilgisi ekleyebilirsiniz:

```bash
renai process "C:/Fotograflar" --mode rename --fallback-mode vision --provider deepinfra --model meta-llama/Llama-3.2-90B-Vision-Instruct --rename-images --exif
```

Bu komut, mevcutsa fotoğrafın EXIF meta verilerinden tarihi dosya adına ekler.

### Tavsiye Edilen Görüş Modelleri

| Model                                      | Sağlayıcı  | Notlar                              |
| :----------------------------------------- | :--------- | :---------------------------------- |
| `meta-llama/Llama-3.2-90B-Vision-Instruct` | DeepInfra  | Kalite ve hız arasında en iyi denge |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | DeepInfra  | Daha hızlı, büyük partiler için iyi |
| `openai/gpt-4o`                            | OpenAI     | En yüksek kalite, daha pahalı       |
| `google/gemini-1.5-pro`                    | OpenRouter | İyi alternatif                      |

### Görüş Modu ve OCR Karşılaştırması

| Özellik     | Görüntü Modu                   | OCR Modu                 |
| :---------- | :----------------------------- | :----------------------- |
| Hız         | Daha hızlı (metin çıkarma yok) | Daha yavaş               |
| Doğruluk    | Modele bağlı                   | Temiz metin için iyi     |
| Maliyet     | Dosya başına API çağrısı       | Dosya başına API çağrısı |
| En iyi için | Taranmış PDF'ler, resimler     | Metin tabanlı PDF'ler    |

---

## 📧 Destek

- **Sorunlar**: GitHub'da sorun açın
- **Tartışmalar**: GitHub Discussions kullanın
- **E-posta**: Bakımcılarla iletişime geçin

---

---

## 🛠️ Geliştirme

```bash
# Linting & Formatlama
uv run ruff check .
uv run ruff format .

# Tip Kontrolü
uv run pyright

# Testleri Çalıştır
uv run pytest tests/ -v
```

---

renAI - Dijital kütüphanenizi yapay zeka ile düzenleyin
