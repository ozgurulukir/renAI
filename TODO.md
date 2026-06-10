# Head-Tail Sampling Refaktör Planı

## Özet

`proportional_chunk_trim` chunking yaklaşımı, metadata çıkarma için suboptimaldir.
Head-tail sampling stratejisiyle değiştirilecek. Terminoloji ve parametreler
tutarlı hale getirilecek. Double-truncation kaldırılacak. LLM'e gönderilen
segmentler head/tail marker ile işaretlenecek.

---

## Faz 0: Kritik Keşif — Extractor Head-Tail Page Sampling

### Sorun

Mevcut extractor'lar (`PyMuPDFExtractor`, `PDFPlumberExtractor`, vb.) **sadece
ilk N sayfayı** linear okuyor:

```python
# pdf_extractors.py:15
for i in range(min(pages, len(doc))):
    page_text = doc[i].get_text()
```

Bu demek ki `extraction_pages=10` ile sadece ilk 10 sayfa okunuyor.
PDF'in **son sayfaları** (referanslar, ISBN, kolofon) hiç extraction'a girmiyor.
Head-tail trimming ancak **tüm metin elimizdeyse** anlamlıdır.

### Çözüm: Page-level head-tail sampling

Extractor'lar sadece ilk N sayfa değil, **ilk N sayfa + son M sayfa** okumalı:

```python
def extract_text(self, file_path: Path, pages: int = 10) -> str | None:
    with pymupdf.open(str(file_path)) as doc:
        total = len(doc)
        # Head: ilk `pages` sayfa
        head_end = min(pages, total)
        head_pages = range(0, head_end)

        # Tail: son `pages` sayfa (head ile overlap varsa atla)
        tail_start = max(head_end, total - pages)
        tail_pages = range(tail_start, total)

        # Deduplicate (kısa PDF'lerde head ve tail overlap edebilir)
        sampled = sorted(set(head_pages) | set(tail_pages))

        text_parts = [doc[i].get_text() for i in sampled]
        return "".join(text_parts).strip()
```

**Sonuç:** 300 sayfalık bir PDF'de `pages=10` ile ilk 10 + son 10 sayfa okunur.
Kısa bir 8 sayfalık PDF'de tüm sayfalar okunur (overlap merge edilir).

### Değişecek dosyalar

- [ ] `src/renai/extractors/pdf_extractors.py` — tüm 4 extractor güncellenecek:
  - `PyMuPDFExtractor.extract_text()`
  - `PDFPlumberExtractor.extract_text()`
  - `PyPDFExtractor.extract_text()`
  - `PyPFDFium2Extractor.extract_text()`
- [ ] `src/renai/extractors/base.py` — `extract_text` docstring güncelle
- [ ] `extraction_pages` parametresi artık "head için N sayfa, tail için N sayfa"
  anlamına gelecek. Help text güncelle.
- [ ] Head ve tail sayfaları arasında marker ekle:
  ```python
  head_text = "".join(doc[i].get_text() for i in head_pages)
  tail_text = "".join(doc[i].get_text() for i in tail_pages)
  return head_text + "\n--- TAIL PAGES ---\n" + tail_text
  ```
  VEYA sadece concatenate et (trimming phase marker'ı ekleyecek).
- [ ] Test: kısa PDF (overlap), uzun PDF (no overlap), 1 sayfalık PDF

### Performans notu

Head-tail page sampling ile okunan sayfa sayısı ikiye çıkar (10→20).
Ama sadece metadata için okunuyor ve `text_length` truncation zaten
gereksiz metni atacak. 20 sayfa ≈ 60.000 karakter → 10.000'e trim.

---

## Faz 1: Core Logic — `utils/__init__.py`

### 1.1 `proportional_chunk_trim` → `head_tail_trim`

**Dosya:** `src/renai/utils/__init__.py:144-176`

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
  - PDF/EPUB başlık, yazar, yayıncı → ilk sayfalarda
  - Referanslar, ISBN, kolofon → son sayfalarda
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

### 2.1 Segment marker ekleme

**Dosya:** `src/renai/engines/llm.py:102-116`

- [ ] `get_book_metadata()` metodunu güncelle:
  - Head-tail trimming `text_extractor` tarafında yapılıyor (Faz 3)
  - LLM'e zaten trimlenmiş metin geliyor
  - Double-truncation (`text[:app_settings.text_length]`) kaldırılacak (satır 108)
  - `<document_content>` prompt'u head/tail aware hale getirilecek

**Mevcut prompt (llm.py:113):**
```python
f"<document_content>\n{truncated_text}\n</document_content>"
```

**Yeni prompt:**
```python
f"<document_content>\n<document_head>\n{head_text}\n</document_head>\n<document_tail>\n{tail_text}\n</document_tail>\n</document_content>"
```

- [ ] `head_tail_trim` çağrısını LLM tarafında yap, text_extractor'dan raw text al
  - VEYA text_extractor head-tail yapıp marker'lı metin dönsün
  - **Karar:** TextExtractor sadece trimming yapar (marker yok). LLMEngine,
    marker'ları ekler ve LLM'e gönderir. Separation of concerns korunur.

- [ ] `llm.py`'de `text_length` double-truncation'ı kaldır (satır 108)

### 2.2 Prompt şablonu güncelle

**Dosya:** `src/renai/prompts/book_extractor.txt`

- [ ] Prompt'a head/tail segment açıklaması ekle:
  ```
  The document content is split into two segments:
  - <document_head>: Beginning of the document (title, author, publisher, copyright)
  - <document_tail>: End of the document (references, ISBN, colophon, publication info)
  Extract metadata from both segments.
  ```

---

## Faz 3: TextExtractor Entegrasyonu

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
- [ ] `_extract_from_document` (satır 242)
- [ ] `_fallback_to_ocr` (satır 252)

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
| `src/renai/extractors/pdf_extractors.py` | Head-tail page sampling (ilk N + son N sayfa) |
| `src/renai/extractors/base.py` | Docstring güncelle |
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
