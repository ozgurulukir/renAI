# Head-Tail Sampling Refaktör Planı

## Özet

`proportional_chunk_trim` chunking yaklaşımı, metadata çıkarma için suboptimaldir.
Head-tail sampling stratejisiyle değiştirilecek. Terminoloji ve parametreler
tutarlı hale getirilecek. Double-truncation kaldırılacak. LLM'e gönderilen
segmentler head/tail marker ile işaretlenecek.

---

## Faz 0: Procedural Head-Tail Page Extraction

### Sorun

Mevcut extractor'lar **sadece ilk N sayfayı** linear okuyor.
PDF'in son sayfaları (referanslar, ISBN, kolofon) hiç okunmuyor.
Büyük PDF'lerde (300+ sayfa) tüm sayfaları okumak zaman kaybı.

### Çözüm: Procedural sayfa-sayfa accumulation

Sabir sayfa okumak yerine, hedef karakter sayısına ulaşana kadar sayfa sayfa oku:

```
Head extraction (ilk sayfadan ileriye):
  target = text_length × head_ratio (örn: 10000 × 0.7 = 7000 char)
  page 0 → char_count ekle → toplam < 7000? → devam
  page 1 → char_count ekle → toplam < 7000? → devam
  page N → toplam >= 7000 → son sayfayı trimle → dur

Tail extraction (son sayfadan geriye):
  target = text_length × (1 - head_ratio) (örn: 10000 × 0.3 = 3000 char)
  page -1 → char_count ekle → toplam < 3000? → devam
  page -2 → char_count ekle → toplam < 3000? → devam
  page -N → toplam >= 3000 → son sayfayı trimle → dur
  ⚠️ Sayfalar sondan başa okunur ama text doğru sırada birleştirilir
```

**Avantaj:** 300 sayfalık PDF'de sadece ~5-7 head + ~2-3 tail sayfa okunur.
Metadata sayfaları sparse olduğu için (başlık, copyright sayfası) 7000 char
için gerçekte 5-8 sayfa yeterli — sabit 10 sayfa okumaktan daha verimli.

### Algoritma detayı

```python
def _extract_head(self, doc, target_chars: int) -> str:
    """İlk sayfadan başlayıp target_chars'a ulaşana kadar oku."""
    parts = []
    accumulated = 0
    for i in range(len(doc)):
        page_text = doc[i].get_text() or ""
        if not page_text.strip():
            continue
        remaining = target_chars - accumulated
        if remaining <= 0:
            break
        if accumulated + len(page_text) > target_chars:
            parts.append(page_text[:remaining])
            accumulated += remaining
            break
        parts.append(page_text)
        accumulated += len(page_text)
    return "".join(parts)

def _extract_tail(self, doc, target_chars: int, head_end_page: int) -> str:
    """Son sayfadan başlayıp geriye doğru target_chars'a ulaşana kadar oku.
    
    Sayfalar sondan başa okunur ama result doğru sırada (head→tail) olmalı.
    """
    parts = []
    accumulated = 0
    total_pages = len(doc)
    for i in range(total_pages - 1, head_end_page - 1, -1):
        page_text = doc[i].get_text() or ""
        if not page_text.strip():
            continue
        remaining = target_chars - accumulated
        if remaining <= 0:
            break
        if accumulated + len(page_text) > target_chars:
            trimmed = page_text[-remaining:]  # sondan trimle
            parts.append(trimmed)
            accumulated += remaining
            break
        parts.append(page_text)
        accumulated += len(page_text)
    # Sondan başa okuduk, doğru sıraya çevir
    parts.reverse()
    return "".join(parts)
```

### Extractor API değişikliği

**Dosya:** `src/renai/extractors/base.py`

```python
# Eski:
def extract_text(self, file_path: Path, pages: int = 10) -> str | None:

# Yeni:
def extract_text(
    self,
    file_path: Path,
    target_chars: int = 10000,
    head_ratio: float = 0.7,
) -> str | None:
```

`pages` parametresi kaldırılıyor, yerine `target_chars` + `head_ratio` geliyor.
Artık sabit sayfa sayısı değil, hedef karakter sayısı driving force.

### Değişecek dosyalar

- [ ] `src/renai/extractors/base.py` — `extract_text` signature + docstring güncelle
- [ ] `src/renai/extractors/pdf_extractors.py` — tüm 4 extractor:
  - `PyMuPDFExtractor.extract_text()` — pymupdf ile direkt sayfa erişimi hızlı
  - `PDFPlumberExtractor.extract_text()`
  - `PyPDFExtractor.extract_text()`
  - `PyPdfium2Extractor.extract_text()`
- [ ] `_extract_head` / `_extract_tail` her extractor'a ekle
  - VEYA `base.py`'de yardımcı metod olarak tanımla (DRY)
  - **Karar:** Base class'ta abstract `get_page_text(doc, page_idx)` tanımla,
    `extract_text` logic'i base class'ta olsun. Subclasses sadece `get_page_text`
    ve `open_doc` implement etsin. Bu sayede algoritma tek yerde.
- [ ] `extraction_pages` ayarı güvenlik limiti olarak kalır (bkz yukarıdaki karar)
- [ ] Test senaryoları:
  - Kısa PDF (5 sayfa, head+tail overlap → tüm sayfalar okunur)
  - Uzun PDF (300 sayfa, head ~5 sayfa, tail ~2 sayfa)
  - 1 sayfalık PDF
  - Boş sayfaları olan PDF (skip blank pages)
  - Sparse metadata pages (copyright sayfası ~100 char)

### Overlap handling

Kısa PDF'lerde head ve tail aynı sayfaları okuyabilir.
Head extraction `head_end_page` döndürür, tail extraction bu sayfadan
başlayarak geriye okur. Eğer `head_end_page >= total_pages` → tail boş döner.

### `extraction_pages` ayarı kaldırılacak mı?

**Hayır** — backward compat için `extraction_pages` Max sayfa limiti olarak kalabilir.
Procedural extraction `target_chars`'a ulaştığında durur ama `extraction_pages`
hard upper bound olarak güvenlik sağlar:
```
max_pages = extraction_pages  # güvenlik limiti (default: 10)
# Head: min(target_chars' ulaş, extraction_pages sayfa)
# Tail: min(target_chars' ulaş, extraction_pages sayfa)
```
Bu sayede aşırı uzun sparse PDF'lerde sonsuz döngü önlenir.

---

## Faz 1: Core Logic — `utils/__init__.py`

### 1.1 `proportional_chunk_trim` → `head_tail_trim`

**Dosya:** `src/renai/utils/__init__.py:144-176`

**Not:** Faz 0'da procedural page-level extraction yapıldıktan sonra,
PDF'ler zaten head+tail olarak döner. Bu fonksiyon sadece **non-PDF** dosyalar
(Office, plain text) ve **regular → head-tail migration** için gerekli.

- [ ] `proportional_chunk_trim` fonksiyonunu kaldır
- [ ] `head_tail_trim(text: str, target_length: int, head_ratio: float = 0.7) -> str` ekle
- [ ] Mantık:
  ```
  if len(text) <= target_length:
      return text
  head_len = int(target_length * head_ratio)
  tail_len = target_length - head_len
  return text[:head_len] + "\n...\n" + text[-tail_len:]
  ```
- [ ] `head_ratio` default: `0.7` (endüstri standardı: %70 head / %30 tail)
- [ ] Edge cases:
  - `text = ""` → `""`
  - `target_length = 0` → `""`
  - `target_length >= len(text)` → text olduğu gibi döner
  - `head_ratio < 0 or > 1` → ValueError
  - `target_length < 3` → sadece head al ("\n...\n" sığmaz)
- [ ] `__all__` ve export'ları güncelle

### 1.2 `__init__.py` exports

- [ ] `proportional_chunk_trim` export'u kaldır
- [ ] `head_tail_trim` export'u ekle

---

## Faz 2: LLM Head/Tail Marker

### 2.1 Veri akışı (PDF)

Faz 0'dan sonra PDF extractor procedural olarak head+tail döndürüyor.
Artık data flow şu şekilde:

```
PDF Extractor (Faz 0):
  _extract_head(doc, 7000)  → head_text
  _extract_tail(doc, 3000)  → tail_text
  return head_text + "\n---\n" + tail_text     # separator ile birleşik
       ↓
TextExtractor._extract_from_document():
  full_text = extractor.extract_text(...)      # zaten head+tail
  return full_text                             # trim gerekmez (zaten target_chars'a göre)
       ↓
LLMEngine.get_book_metadata(text):
  # text zaten head+tail separator ile ayrılmış
  # marker'ları ekleyip LLM'e gönder
```

### 2.2 LLM marker ekleme

**Dosya:** `src/renai/engines/llm.py:102-116`

- [ ] `get_book_metadata()` metodunu güncelle:
  - Double-truncation (`text[:app_settings.text_length]`) kaldır (satır 108)
  - Text'teki `"\n---\n"` separator'ı bul ve marker'larla değiştir

**Mevcut prompt (llm.py:113):**
```python
f"<document_content>\n{truncated_text}\n</document_content>"
```

**Yeni prompt:**
```python
# text_extractor'dan gelen text: "head_content\n---\ntail_content"
if "\n---\n" in text:
    head_text, tail_text = text.split("\n---\n", 1)
    prompt = (
        f"<document_content>\n"
        f"<document_head>\n{head_text}\n</document_head>\n"
        f"<document_tail>\n{tail_text}\n</document_tail>\n"
        f"</document_content>"
    )
else:
    # Non-PDF veya regular mode — marker yok
    prompt = f"<document_content>\n{text}\n</document_content>"
```

### 2.3 Prompt şablonu güncelle

**Dosya:** `src/renai/prompts/book_extractor.txt`

- [ ] Prompt'a head/tail segment açıklaması ekle:
  ```
  The document content may be split into two segments:
  - <document_head>: Beginning of the document (title, author, publisher, copyright)
  - <document_tail>: End of the document (references, ISBN, colophon, publication info)
  Extract metadata from both segments when present.
  ```

---

## Faz 3: TextExtractor + OCR Fallback Entegrasyonu

### 3.1 `_apply_mixed_mode_or_trim` güncelle

**Dosya:** `src/renai/processors/text_extractor.py:143-147`

- [ ] Fonksiyonu `_apply_extraction_mode` olarak yeniden adlandır
- [ ] İçerik:
  ```python
  def _apply_extraction_mode(self, text: str, target_length: int) -> str:
      if self.text_extraction_mode == "head-tail":
          return head_tail_trim(text, target_length, app_settings.head_ratio)
      return text[:target_length]  # regular mode
  ```
- [ ] Import: `proportional_chunk_trim` → `head_tail_trim`

### 3.2 Çağrı noktaları (4 yer)

Tüm `_apply_mixed_mode_or_trim` çağrılarını `_apply_extraction_mode` ile değiştir:

- [ ] `_extract_from_office` (satır 194)
- [ ] `_extract_from_plain_text` (satır 217)
- [ ] `_extract_from_document` (satır 242) — **not:** PDF'ler Faz 0'da procedural
  head-tail ile geliyor, burada `_apply_extraction_mode` çağrısı sadece regular
  mode veya non-PDF dosyalar için aktif olur
- [ ] `_fallback_to_ocr` (satır 252) — bkz 3.3

### 3.3 OCR Fallback güncellemesi

**Dosya:** `src/renai/engines/ocr.py:274-283`

Mevcut OCR page sampling (`_get_page_sampling`):
```python
# İlk 15 sayfa + orta 2 sayfa + son 2 sayfa
pages_to_extract = list(range(min(15, total_pages)))
if total_pages > 30:
    mid = total_pages // 2
    pages_to_extract += [mid, mid + 1]
    pages_to_extract += [total_pages - 1, total_pages]
```

**Sorunlar:**
1. İlk 15 sayfa çok agresif — metadata için gereksiz, yavaş
2. `total_pages - 1, total_pages` — son index `total_pages` out of bounds olabilir
3. Orta sayfalar metadata için değersiz
4. OCR zaten yavaş bir işlem, daha az sayfa = daha hızlı

**Güncelleme:**
- [ ] `_get_page_sampling` → head-tail aware yap:
  ```python
  def _get_page_sampling(self, total_pages: int) -> list[int]:
      HEAD_PAGES = 5   # İlk 5 sayfa (başlık, yazar, telif)
      TAIL_PAGES = 3   # Son 3 sayfa (referanslar, ISBN)
      head = list(range(min(HEAD_PAGES, total_pages)))
      tail_start = max(HEAD_PAGES, total_pages - TAIL_PAGES)
      tail = list(range(tail_start, total_pages))
      return sorted(set(head + tail))
  ```
- [ ] `MAX_OCR_CHARS = 10000` korunsun ama `head_ratio` ile bölünsün:
  ```python
  # OCR extract_text() sonrası text_extractor'da:
  # _fallback_to_ocr → _apply_extraction_mode ile trim
  # OCR metni zaten head+tail sayfalardan geliyor
  # head_tail_trim sadece boyut limiti için gerekli
  ```
- [ ] `_fallback_to_ocr` akışı:
  ```
  OCREngine.extract_text(file_path)
    → _get_page_sampling() → head+tail sayfaları
    → OCR each page → concatenate → return
       ↓
  text_extractor._fallback_to_ocr()
    → _apply_extraction_mode(ocr_text, target_length)
    → head-tail mode: head_tail_trim(ocr_text, target_length, head_ratio)
    → regular mode: ocr_text[:target_length]
  ```
- [ ] Out of bounds fix: `total_pages - 1` yerine `total_pages` değil,
  `range(tail_start, total_pages)` zaten doğru index verir
- [ ] OCR `extract_text`'te `MAX_OCR_CHARS` kontrolü (satır 524) —
  bu de `target_chars` parametresi ile uyumlu hale getirilmeli:
  ```python
  # Eski:
  MAX_OCR_CHARS = 10000
  if total_chars > MAX_OCR_CHARS:
      break

  # Yeni: target_chars parametresi al
  async def extract_text(self, file_path: Path, target_chars: int = 10000) -> str:
      ...
      if total_chars > target_chars:
          break
  ```

---

## Faz 4: Ayarlar ve Terminoloji

### 4.1 Parametre yeniden adlandırma

**Dosya:** `src/renai/config/settings.py`

| Eski | Yeni | Açıklama |
|---|---|---|
| `text_extraction: str = "mixed"` | `text_extraction: str = "head-tail"` | Mod adı güncellendi |
| `mixed_chunk_count: int = 10` | `head_ratio: float = Field(default=0.7, ge=0.3, le=0.9)` | Chunk count → ratio |
| `text_length: int = 10000` | `text_length: int = 10000` | Aynı kalır ama review (bkz 4.2) |

**Not:** `text_extraction` değerleri artık `"head-tail"` veya `"regular"` olacak.
`"mixed"` geriye dönük uyumluluk için geçici olarak kabul edilebilir (alias).

### 4.2 `text_length` parametresi review

Mevcut: `text_length = 10000`, range `100-100000`.

- [ ] Default değer korunacak (10000 karakter ≈ ~2500-3000 token, makul)
- [ ] CLI `--text-length` bayrağı korunacak
- [ ] Range korunacak, ancak Pydantic validator eklenebilir:
  - Uyarı: 50000+ karakter LLM context'i aşabilir, log warning ekle
- [ ] Help text güncelle: `"Target text length for LLM"` →
      `"Max characters sent to LLM (split into head+tail segments in head-tail mode)"`

### 4.3 `settings.default.toml` güncelle

**Dosya:** `src/renai/config/settings.default.toml`

```toml
# Eski:
text_extraction = "regular"
mixed_chunk_count = 10

# Yeni:
text_extraction = "head-tail"
head_ratio = 0.7
```

### 4.4 Backward compatibility

- [ ] `text_extraction = "mixed"` → otomatik olarak `"head-tail"` olarak yorumla
  - `@model_validator` ekle: `"mixed"` değerini `"head-tail"` olarak normalize et
- [ ] `mixed_chunk_count` TOML/env'de varsa → ignore et (veya log warning)
- [ ] `proportional_chunk_trim` import'u kaldır ama `DeprecationWarning` ile 1 sürüm
  daha yaşatma seçeneğini değerlendir — **karar:** Temiz kesim, kaldır.

---

## Faz 5: CLI Güncellemeleri

### 5.1 CLI parametreleri

**Dosya:** `src/renai/cli.py`

- [ ] `--text-extraction` help text: `"Extraction mode: head-tail or regular"`
- [ ] `--mixed-chunk-count` → kaldır (artık kullanılmıyor)
- [ ] Eğer `--head-ratio` CLI flag'i isteniyorsa ekle (opsiyonel, advanced)
  - `--head-ratio: float = 0.7, help="Head segment ratio (0.3-0.9)"`

### 5.2 ProcessingConfig model

**Dosya:** `src/renai/models/schemas.py`

- [ ] `ProcessingConfig` içinde `text_extraction` alanı varsa kontrol et
- [ ] `mixed_chunk_count` referansı varsa kaldır

---

## Faz 6: Testler

### 6.1 Silinecek testler (chunking-specific)

- [ ] `tests/test_weighted_preservation.py` → tamamen kaldır (chunk-based)
- [ ] `tests/test_trimming.py` → tamamen kaldır (chunk-based)

### 6.2 Güncellenecek testler

- [ ] `tests/test_utils.py`:
  - `test_proportional_chunk_trim_basic` → `test_head_tail_trim_basic`
  - `test_proportional_chunk_trim_edge_cases` → `test_head_tail_trim_edge_cases`
  - `test_proportional_chunk_trim_rounding_behavior` → kaldır (yuvarlama sorunu artık yok)
  - Yeni testler:
    - Head-tail split doğru oranda
    - `"\n...\n"` separator mevcut
    - Head metni text'in başından, tail metni text'in sonundan
    - `head_ratio` sınır değerleri (0.3, 0.9)
    - `target_length < 3` durumunda separator olmadan sadece head

### 6.3 Güncellenecek testler (referanslar)

- [ ] `tests/test_cli_process.py` — `text_extraction="regular"` korunabilir
- [ ] `tests/test_progress_bar.py` — `text_extraction="mixed"` → `"head-tail"`
- [ ] `tests/test_ocr_skip.py` — `text_extraction="regular"` korunabilir

### 6.4 Yeni testler

- [ ] `tests/test_head_tail_trim.py` — kapsamlı head-tail testi:
  - Metadata preservation: başlık başta, ISBN sonda
  - Orta kısım atlanıyor doğrulama
  - Farklı `head_ratio` değerleri
  - Marker'ların LLM'e doğru aktarımı

---

## Faz 7: Dökümantasyon

- [ ] `README.md` — `--text-extraction` açıklaması güncelle
- [ ] `okubeni.md` — Türkçe dökümantasyon güncelle
- [ ] `CHANGELOG.md` — `[Unreleased]` bölümüne ekle
- [ ] `docs/CACHING_STRATEGY.md` — etkilenmez, kontrol
- [ ] `AGENTS.md` — `mixed_chunk_count` referansı varsa güncelle

---

## Faz 8: Geçiş Kontrol Listesi

### Kod akışı (değişiklik sonrası)

```
TextExtractor.extract(file_path)
  ├─ text = raw extraction (PDF/EPUB/Office/...)
  ├─ _apply_extraction_mode(text, target_length)
  │   ├─ "head-tail" → head_tail_trim(text, target_length, head_ratio=0.7)
  │   │   → "AAAA...\n...\n...BBBB"  (≤ target_length karakter)
  │   └─ "regular" → text[:target_length]
  └─ return trimmed_text

LLMEngine.get_book_metadata(text)
  ├─ # DOUBLE TRUNCATION KALDIRILDI — text zaten trimlenmiş
  ├─ prompt = "<document_content>
  │     <document_head>{head}</document_head>
  │     <document_tail>{tail}</document_tail>
  │   </document_content>"
  └─ LLM call
```

### Dosya değişiklik özeti

| Dosya | Değişiklik |
|---|---|
| `src/renai/extractors/pdf_extractors.py` | Procedural head-tail page extraction |
| `src/renai/extractors/base.py` | Signature: `pages` → `target_chars` + `head_ratio` |
| `src/renai/engines/ocr.py` | `_get_page_sampling` head-tail aware, `target_chars` param |
| `src/renai/utils/__init__.py` | `proportional_chunk_trim` → `head_tail_trim` |
| `src/renai/engines/llm.py` | Double truncation kaldır, marker ekle |
| `src/renai/processors/text_extractor.py` | Import + metot adı güncelle |
| `src/renai/config/settings.py` | `mixed_chunk_count` → `head_ratio`, değer `"mixed"` → `"head-tail"` |
| `src/renai/config/settings.default.toml` | Parametreler güncelle |
| `src/renai/cli.py` | Help text + gereksiz parametre kaldır |
| `src/renai/models/schemas.py` | ProcessingConfig güncelle |
| `src/renai/prompts/book_extractor.txt` | Head/tail açıklaması ekle |
| `tests/test_utils.py` | Testleri güncelle |
| `tests/test_trimming.py` | Sil |
| `tests/test_weighted_preservation.py` | Sil |
| `tests/test_head_tail_trim.py` | Yeni test dosyası |

### Geriye dönük uyumluluk

- `settings.toml`'de `text_extraction = "mixed"` → model_validator ile `"head-tail"` normalize
- `settings.toml`'de `mixed_chunk_count = 10` → sessizce ignore
- CLI `--text-extraction mixed` → `"head-tail"` normalize
- Cache: metin cache'leri SHA256-keyed, trimming değişse bile cache geçerli
  (text cache raw extraction, metadata cache provider/model keyed)
- Metadata cache: etkilenmez (provider+model keyed)

---

## Uygulama Sırası

1. **Faz 0** — Extractor head-tail page sampling (kritik — olmadan head-tail çalışmaz)
2. **Faz 1** — Core `head_tail_trim` fonksiyonu
3. **Faz 6.4** — Yeni testler (TDD)
4. **Faz 3** — TextExtractor entegrasyonu
5. **Faz 2** — LLM marker + double truncation kaldır
6. **Faz 4** — Settings + backward compat
7. **Faz 5** — CLI güncellemeleri
8. **Faz 6.1-6.3** — Eski testleri temizle/güncelle
9. **Faz 7** — Dökümantasyon
10. **Faz 8** — Geçiş kontrolü + ruff + pyright + pytest
