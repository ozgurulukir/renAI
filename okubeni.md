# renAI: Yapay Zeka Güçlü Kitap Yeniden Adlandırıcı ve Düzenleyici

`renAI`, PDF, EPUB, MOBI ve diğer belge formatlarının içeriğinden doğrudan meta verileri çıkararak dijital kitap koleksiyonunuzu otomatik olarak yeniden adlandırmak ve düzenlemek için tasarlanmış güçlü bir Python aracıdır. LLM'ler (Büyük Dil Modelleri) ve OCR (Optik Karakter Tanıma) kullanır.

---

## ⚙️ Özel Kategoriler Yapılandırması

Proje kök dizininde `categories.txt` dosyası oluşturarak özel kategoriler tanımlayabilirsiniz.

### Sözlük Yapısı:
Kategoriler, dil kodlarını anahtar olarak ve kategori listelerini değer olarak içeren bir sözlük olarak depolanır:
```json
{
  "en": ["Article", "Fiction", "Non-Fiction", ...],
  "tr": ["Makale", "Kurgu", "Kurgu Dışı", ...]
}
```

### Dosya Formatı:
- `#` ile başlayan satırlar yorum satırlarıdır
- Boş satırlar yoksayılır
- Dil belirtmek için `LANG: dil_kodu` kullanın (tr Türkçe, en İngilizce için)
- Kategoriler ilgili dillerin altında listelenir

### Örnek `categories.txt`:
```
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
Biography
Memoir
Science
History
Philosophy
Psychology

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
Biyografi
Öykü
Ansiklopedi
Bilim
Tarih
Felsefe
Psikoloji
Ekonomi
Teknoloji
Programlama
Aşçılık
Seyahat
Sağlık
Kişisel Gelişim
```

`categories.txt` dosyası yoksa araç varsayılan kategorileri kullanır.

---

## ⚠️ Önemli Uyarılar ve En İyi Uygulamalar

### Organize Modu İçin Dizin Yapısı Uyarısı
**UYARI:** `--mode organize` kullanırken, araç dosyaları kategori klasörlerine taşıyacak ve yeniden adlandıracaktır. Bu, mevcut dizin yapınızı bozabilir:
- Tüm alt dizinlerdeki dosyalar kök seviyedeki kategori klasörlerine taşınır (örn. `Kurgu/`, `Makale/`)
- Dosyalar taşındıktan sonra boş alt dizinler kalabilir
- Kategori klasörleri kök seviyede oluşturulur, alt dizinlerin içinde değil
- Organize modunu birden fazla kez çalıştırmak, dosyaları yeniden değerlendirecek ve klasör yapısını değiştirebilir

**Öneri:** Organize modunu çalıştırmadan önce her zaman dosyalarınızın yedeğini alın ve önce küçük bir dizinle test edin.

### Veri Kaybını Önleme
- **Her zaman dosyalarınızın yedeğini alın** renAI'yi `rename` veya `organize` modunda çalıştırmadan önce. Araç dosyaları yeniden adlandırır ve taşır; güvenlik kontrolleri içerse de beklenmeyen sorunlar oluşabilir.
- **İsimlendirme kuralını inceleyin** koleksiyonunuzda uygulamadan önce. Format: `Başlık -- Alt Başlık -- Yazar (Yıl).ext`
- **Önce küçük bir dizinle test edin** aracın belirli dosyalarınızla nasıl davranacağını doğrulamak için.

### Token Kullanımı ve Bütçe İle İlgili Hususlar
- **LLM API maliyetleri geçerlidir**: İşlenen her dosya metni bir LLM API'ye gönderir. API kullanımınızı düzenli olarak izleyin.
- **Metin uzunluğu sınırları**: Her dosya için LLM'e en fazla 8000 karakter gönderilir. Büyük belgeler kesilebilir.
- **Önbellekleme maliyetleri azaltır**: Çıkarılan metin ve meta veriler önbelleğe alınır. Yalnızca gerekli olduğunda `--update-metadata` kullanın.
- **Bütçenizi tahmin edin**: 1000 dosya için yaklaşık $0.001/filo maliyetle, ~$1 API maliyeti bekleyin. Gerçek maliyetler sağlayıcı ve modele göre değişir.
- **Oran sınırları**: Bazı sağlayıcılar oran sınırları uygular. Eşzamanlılığı kontrol etmek için `--workers` kullanın.

### Önerilen En İyi Uygulamalar
1. **Düzenli Yedekler**: İşlemeden önce orijinal dosya koleksiyonunuzun en az bir yedeğini tutun.
2. **Küçük Başla**: Meta veri çıkarma kalitesini doğrulamak için önce 5-10 dosya test edin.
3. **API Kullanımını İzleyin**: Beklenmeyen ücretlerden kaçınmak için LLM sağlayıcınızda kullanım uyarıları ayarlayın.
4. **Önbellek Yönetimi**: Modelleri veya çıkarma yöntemlerini değiştirdiğinde `--clear-cache` kullanarak önbellek verilerini sıfırlayın.
5. **OCR Kalitesi**: Taranmış belgelerle çalışırken daha iyi sonuçlar için `--ocr-enhanced --use-mutool` kullanın (ama daha fazla metin nedeniyle daha yüksek API maliyetleri bekleyin).
6. **Sağlayıcı Seçimi**: DeepInfra, OpenRouter ve OpenAI arasında maliyetleri karşılaştırın. OpenRouter genellikle rekabetçi fiyatlar sunar.
7. **Skew Düzeltme**: Döndürülmüş/skew'lenmiş belgeler için gelişmiş OCR modu OpenCV kullanarak otomatik skew düzeltme içerir (`pip install opencv-python` ile yükleyin).

---

## 🚀 Özellikler

- **Çoklu Format Desteği:** PDF, EPUB, MOBI, DOCX, PPTX, XLSX, TXT ve MD dosyalarını işler.
- **Çoklu PDF Çıkarıcılar:** Metin çıkarma için PyMuPDF, pdfplumber, pypdf veya pypdfium2'den birini seçin.
- **Yapay Zeka Destekli Meta Veri Çıkarma:** Başlıkları, yazarları, yılları ve kategorileri tanımlamak için DeepInfra, OpenRouter (OpenAI uyumlu) veya OpenAI kullanır.
- **Gelişmiş OCR:** Düşük kaliteli taramaları temizlemek ve kesmek için "Gelişmiş Mod" içeren Tesseract OCR entegrasyonu.
- **mutool Entegrasyonu:** Sabit düzenli PDF'lerden yüksek kaliteli görüntüleri çıkarmak için `mutool` desteği.
- **Akıllı Önbellekleme:** API maliyetlerini ve zamanı tasarruf etmek için MD5 hash kullanarak çıkarılan metni ve meta verileri önbelleğe alır.
- **Esnek Modlar:**
  - `rename`: Çıkarılan meta verilere göre dosyaları yeniden adlandırır.
  - `organize`: Yeniden adlandırılmış dosyaları otomatik olarak oluşturulmuş kategori klasörlerine taşır.
  - `benchmark`: Farklı PDF çıkarma kütüphanelerini karşılaştırır (`PyMuPDF`, `pdfplumber`, vb.).
  - `eval-model`: Meta veri doğruluğu için farklı LLM modellerini değerlendirir. (Modeller `eval_models_{provider}.txt` dosyaları aracılığıyla yönetilir.)

---

## 🛠️ Kurulum

### 1. Ön Gereksinimler
- **Python 3.10+**
- **Tesseract OCR:** [İndir ve yükle](https://github.com/UB-Mannheim/tesseract/wiki/Tesseract-at-UB-Mannheim).
- **mutool (İsteğe bağlı ama önerilir):** [MuPDF](https://mupdf.com/releases/index.html) parçası.

### 2. API Anahtarları
API anahtarlarınızı alın:
- **DeepInfra:** [https://deepinfra.com](https://deepinfra.com)
- **OpenRouter:** [https://openrouter.ai](https://openrouter.ai)
- **OpenAI:** [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### 3. Depoyu Klonlayın
```bash
git clone https://github.com/ozgurulukir/renAI.git
cd renAI
```

### 4. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

---

## ⚙️ Yapılandırma

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

`config.py` dosyasında varsayılan modeller ve Tesseract yolları gibi ayarları da özelleştirebilirsiniz.

---

## 📖 Kullanım

### Temel Yeniden Adlandırma
```bash
python main.py "C:/Path/To/Your/Books" --mode rename
```

### Klasörler Düzenleme
```bash
python main.py "C:/Path/To/Your/Books" --mode organize
```

### Yüksek Kaliteli OCR (taramalar için)
```bash
python main.py "C:/Path/To/Your/Books" --ocr-enhanced --use-mutool
```

### Farklı Yapay Zeka Sağlayıcıları Kullanma

#### DeepInfra (varsayılan)
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --provider deepinfra
```

#### OpenRouter
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --provider openrouter
```

#### OpenAI
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --provider openai
```

### Farklı PDF Çıkarıcılar Kullanma

#### Varsayılan PyMuPDF yerine pdfplumber kullanma
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --extractor pdfplumber
```

#### pypdf kullanma
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --extractor pypdf
```

#### pypdfium2 kullanma
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --extractor pypdfium2
```

### Özel Model Seçimi

#### DeepInfra için özel model belirtme
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --model Qwen/Qwen2.5-72B-Instruct
```

#### OpenRouter için özel model belirtme
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --provider openrouter --model google/gemini-2.0-flash-exp:free
```

#### OpenAI için özel model belirtme
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --provider openai --model gpt-4o
```

### Birden Fazla Dosyayı Eşzamanlı İşleme

#### Daha hızlı işlemek için çalışan sayısını artırma
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --workers 8
```

### Meta Verileri Zorla Yenileme

#### Önbelleği atla ve meta verileri yeniden çıkar
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --update-metadata
```

### OCR İşleme Hatasını Giderme

#### Sorun giderme için OCR hata ayıklama görüntüleri kaydet
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --debug-ocr
```

### PDF Çıkarıcıları Benchmarklama

#### Farklı çıkarıcıların performansını karşılaştır
```bash
python main.py "C:/Path/To/Your/Books" --benchmark
```

### LLM Modellerini Değerlendirme

#### Meta veri doğruluğu için birden fazla model test et
```bash
python main.py "C:/Path/To/Your/Books" --eval-model --provider deepinfra
```

#### OpenAI modellerini değerlendir
```bash
python main.py "C:/Path/To/Your/Books" --eval-model --provider openai
```

### Gelişmiş Seçenekler
| Seçenek | Açıklama |
| :--- | :--- |
| `--provider` | `deepinfra`, `openrouter` veya `openai` arasında seçim yapın. |
| `--extractor` | PDF çıkarıcı seçin (`pymupdf`, `pypdf`, `pdfplumber`, `pypdfium2`). |
| `--workers` | Eşzamanlı işlenecek dosya sayısı (varsayılan: 4). |
| `--update-metadata` | Önbelleği atlayarak yeni çıkarma ve LLM çağrısı yapın. |
| `--ocr-enhanced` | Görüntü artırma ile yüksek kaliteli OCR'yi etkinleştirin. |
| `--use-mutool` | Daha yüksek kaliteli PDF görüntü çıkarmak için mutool kullanın. |
| `--debug-ocr` | OCR hata ayıklama görüntülerini `.ocr_debug/` klasörüne kaydedin. |
| `--benchmark` | İlk 3 dosyada çıkarma benchmark'ı çalıştırın. |
| `--eval-model` | Seçilen sağlayıcı için birden fazla modeli değerlendirin. |
| `--clear-cache` | Tüm önbelleği veya belirtilen dizindeki dosyaların önbelleğini temizleyin (metin ve meta verileri siler). |
| `--recursive` | Alt dizinleri özyinelemeli işler (varsayılan: True). |
| `--no-recursive` | Sadece kök dizini işler (özyineleme yok). |
| `--move-ocr-failures` | OCR gerektiren dosyaları OCR_Failures klasörüne taşıyın. |

### Önbelleği Temizleme

#### Tüm önbelleklenmiş metin ve meta verileri temizle (dizine ihtiyaç yoktur)
```bash
python main.py --clear-cache
```

#### Bir dizinin önbelleğini temizle (işlenecek dosyaların metin ve meta verilerini siler)
```bash
python main.py "C:/Path/To/Your/Books" --clear-cache
```

#### Tek bir dosyanın önbelleğini temizle
```bash
python main.py "C:/Path/To/Your/Books/book.pdf" --clear-cache
```

Bu önbellek dosyalarını siler:
- Önbelleklenmiş çıkarılan metin dosyaları (örn. `hash_enhanced.txt`, `hash_basic.txt`)
- Önbelleklenmiş meta veri dosyaları (örn. `hash_metadata.json`)

---

### Özyinelemeli İşleme

#### Sadece kök dizini işleyin (özyineleme yok)
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --no-recursive
```

#### Özyinelemeli işleyin (varsayılan davranış)
```bash
python main.py "C:/Path/To/Your/Books" --mode rename --recursive
```

`organize` modu için özyineleme varsayılan olarak etkindir, çünkü kök seviyedeki kategori klasörlerine tüm alt dizinlerden dosyaları düzenlemek için.

#### Organize Modu İçin Yeniden İşleme Davranışı:
Aynı dizin üzerinde organize modunun tekrar çalıştırılması desteklenir. Kategori klasörlerindeki dosyalar yeniden değerlendirilir ve gerekirse yeniden adlandırılıp taşınabilir. Bu, şu durumlarda faydalıdır:
+- Farklı bir LLM sağlayıcısı veya modeline geçiş yapmak
+- `categories.txt` dosyasındaki kategori tanımlarını güncellemek
+- Manuel olarak taşınmış dosyaları düzeltmek
+- Farklı ayarlarla yeniden işlemek
+
+Önbellekleme mekanizması hala geçerlidir - bir dosya için önbelleklenmiş sonuçlar varsa metin ve meta veri çıkarma atlanır.
+
+#### OCR Hataları Klasörü
+`--move-ocr-failures` kullanıldığında, işlenemeyen dosyalar (hiçbir metin çıkartılamadıysa, OCR geri dönüşü de dahil) hedef dizindeki `OCR_Failures` klasörüne taşınır. Bu, manuel inceleme gerektirebilecek sorunlu dosyaları belirlemeye yardımcı olur.

---

## 🏗️ Teknik Mimari

- **`main.py`:** Giriş noktası ve CLI yöneticisi.
- **`config.py`:** API anahtarları, yolları ve değerlendirme modelleri için yapılandırma.
- **`utils.py`:** Hashleme, temizleme ve metin parçalama için yardımcı fonksiyonlar.
- **`file_processor.py`:** Akışı yönetir: çıkar → LLM → yeniden adlandır/taşı.
- **`ocr_engine.py`:** Görüntü ön işleme ve Tesseract entegrasyonunu yönetir.
- **`llm_engine.py`:** Yapay zeka sağlayıcıları ile iletişim kurar (DeepInfra, OpenRouter, OpenAI) ve JSON çıktısını doğrular.
- **`extractors/`:** Çeşitli belge çözümleme motorları için modüler paket.

---

## 📝 Lisans

[MIT Lisansı](LICENSE)

---

## 🤝 Katkıda Bulunma

Katkılar memnuniyetle karşılanır! Lütfen bir Pull Request göndermekten çekinmeyin.

1. Depoyu fork edin
2. Özellik dalı oluşturun (`git checkout -b feature/GuzelOzellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Bazi Guzel Ozellik Ekle'`)
4. Dala push edin (`git push origin feature/GuzelOzellik`)
5. Bir Pull Request açın

---

## 📧 Destek

Destek için lütfen GitHub deposunda bir issue açın veya bakımcılarla iletişime geçin.