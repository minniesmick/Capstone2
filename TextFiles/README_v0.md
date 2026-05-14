# 🗑️ Garbage Classification — Hibrit AI Arena

> **Local CNN vs Vision Language Model** — Eğitilmiş atık sınıflandırma modellerini OpenWebUI üzerinden gerçek zamanlı karşılaştırın.

---

## 📌 Proje Özeti

Bu proje, **6 kategorili atık sınıflandırması** için Custom CNN ve Transfer Learning modellerini eğitir; ardından bu modelleri Ollama üzerinden çalışan bir VLM (Vision Language Model) ile kıyaslayan bir **"Model Arena"** ortamı sunar.

| Kategori | Açıklama |
|----------|----------|
| `cardboard` | Karton / mukavva |
| `glass` | Cam |
| `metal` | Metal |
| `paper` | Kağıt |
| `plastic` | Plastik |
| `trash` | Diğer çöp |

---

## 🏗️ Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────┐
│                    KULLANICI KATMANI                    │
│              OpenWebUI  (localhost:3000)                 │
│          Browser / macOS izleme / Sunum                  │
└────────────────────┬────────────────────────────────────┘
                     │  HTTP / Tools API
          ┌──────────┴──────────┐
          │                     │
┌─────────▼──────────┐  ┌──────▼──────────────────────────┐
│   FastAPI Servisi  │  │        Ollama Servisi            │
│  localhost:8000    │  │       localhost:11434            │
│  (api_service.py)  │  │  Moondream / LLaVA / llama-vision│
└─────────┬──────────┘  └──────────────────────────────────┘
          │
┌─────────▼──────────────────────────────────────────────┐
│              Yerel Model Havuzu (.pth dosyaları)        │
│  custom_cnn  │  resnet50  │  efficientnet_b0  │  mobilenet_v3 │
│            NVIDIA RTX 3060 Ti (CUDA 12.x)              │
└────────────────────────────────────────────────────────┘
```

### Dosya Yapısı

```
Capstone2/
├── config.py              # ⚙️  Platform-aware merkezi yapılandırma
├── models.py              # 🧠  Paylaşılan model mimarileri
├── api_service.py         # 🚀  FastAPI çıkarım servisi
├── arena_battle.py        # 🏟️  CNN vs VLM terminal karşılaştırması
├── openwebui_tool.py      # 🔌  OpenWebUI Tools entegrasyonu
├── requirements.txt       # 📦  Bağımlılıklar
├── garbage_classification_complete.py  # 📊  Eğitim & analiz (referans)
└── results/
    ├── models/            # 💾  .pth checkpoint dosyaları
    ├── analysis/          # 📈  Veri seti istatistikleri
    └── visualizations/    # 🖼️  Grafik ve karşılaştırma görselleri
```

---

## ⚡ Hızlı Başlangıç

### 1. Ortam Kurulumu

```bash
# Python 3.11 sanal ortamı (Windows)
python -m venv venv
.\venv\Scripts\activate

# CUDA 12.1 için PyTorch (RTX 3060 Ti)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Diğer bağımlılıklar
pip install -r requirements.txt
```

### 2. Model Yollarını Kontrol Et

```bash
python config.py
```

Örnek çıktı:
```
OS           : Windows
PROJECT_ROOT : C:\Users\alper\PROJELER\Capstone2
MODELS_DIR   : C:\Users\alper\PROJELER\Capstone2\results\models
  ✅  custom_cnn           → ...models\custom_cnn_best.pth
  ✅  resnet50             → ...models\resnet50_best.pth
  ❌  efficientnet_b0      → ...models\efficientnet_b0_best.pth
```

### 3. FastAPI Servisini Başlat

```bash
python api_service.py
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### 4. Arena Karşılaştırması Çalıştır

```bash
# Varsayılan görsel, tüm modeller + Moondream
python arena_battle.py

# Belirli görsel
python arena_battle.py --image C:\Users\alper\Desktop\test_image.png

# Sadece CNN (VLM atla)
python arena_battle.py --no-vlm

# Belirli CNN modelleri
python arena_battle.py --models custom_cnn resnet50

# Sonuçları kaydet
python arena_battle.py --save results/arena_results.json
```

---

## 🔌 OpenWebUI Entegrasyonu

### Tool Kurulumu

1. OpenWebUI'yu açın → **Settings** → **Tools** → **"+ New Tool"**
2. `openwebui_tool.py` dosyasının tüm içeriğini yapıştırın
3. **Save** → Sohbet ekranına dönün → araç simgesini aktifleştirin

### Desteklenen Komutlar

| Komut (doğal dil) | Çağrılan Fonksiyon |
|-------------------|-------------------|
| `classify my test image` | `classify_default_image()` |
| `classify image at C:\path\img.jpg` | `classify_image_by_path()` |
| `arena compare` | `arena_compare()` |
| `check api status` | `check_api_status()` |

### Ortam Değişkeni (opsiyonel)

```bash
# API farklı bir adreste çalışıyorsa
set GARBAGE_API_URL=http://192.168.1.100:8000   # Windows
export GARBAGE_API_URL=http://192.168.1.100:8000  # macOS/Linux
```

---

## 🌐 API Referansı

Servis çalışırken `http://localhost:8000/docs` adresinden tam Swagger dokümantasyonuna erişebilirsiniz.

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/health` | Servis & GPU durumu |
| GET | `/models` | Yüklü model listesi |
| GET | `/predict?model=X` | Varsayılan görsel tahmini |
| GET | `/predict/path?img=PATH&model=X` | Yerel yol ile tahmin |
| POST | `/predict/upload?model=X` | Dosya yükleme ile tahmin |
| POST | `/predict/all` | Tüm modeller ile tahmin |

### Örnek Yanıt (`/predict`)

```json
{
  "model": "custom_cnn",
  "prediction": "plastic",
  "confidence": 94.37,
  "top3": [
    {"class": "plastic",   "confidence": 94.37},
    {"class": "metal",     "confidence":  3.21},
    {"class": "cardboard", "confidence":  1.44}
  ],
  "device": "cuda",
  "image_path": "C:\\Users\\alper\\Desktop\\test_image.png"
}
```

---

## 🤖 Model Bilgileri

| Model | Mimari | Parametreler | Notlar |
|-------|--------|-------------|--------|
| `custom_cnn` | Sıfırdan 4-katmanlı CNN | ~13M | Hızlı, tam kontrol |
| `resnet50` | ResNet-50 (fine-tuned) | ~25M | Güçlü feature extraction |
| `efficientnet_b0` | EfficientNet-B0 (timm) | ~5M | Verimli, hafif |
| `mobilenet_v3` | MobileNetV3-Small | ~2.5M | Edge deployment için |
| `moondream` | VLM (Ollama) | ~1.8B | Görsel-dil anlama |

### CustomCNN Mimarisi

```
Input (3×224×224)
    │
    ├─ Conv1 (3→32)  + BN + ReLU + MaxPool → 32×112×112
    ├─ Conv2 (32→64) + BN + ReLU + MaxPool → 64×56×56
    ├─ Conv3 (64→128)+ BN + ReLU + MaxPool → 128×28×28
    ├─ Conv4 (128→256)+BN + ReLU + MaxPool → 256×14×14
    │
    └─ FC: Flatten(50176) → 512 → Dropout(0.5) → 6
```

---

## 🖥️ Platform Uyumluluğu

`config.py` işletim sistemini otomatik algılar:

| İşletim Sistemi | `PROJECT_ROOT` | Test Görseli |
|-----------------|----------------|--------------|
| Windows 11 | `~\PROJELER\Capstone2` | `~\Desktop\test_image.png` |
| macOS | `~/Projects/Capstone2` | `~/Desktop/test_image.png` |
| Linux / Docker | `~/Capstone2` | `~/test_image.png` |

Ortam değişkeni ile özelleştirebilirsiniz:
```bash
set CAPSTONE2_ROOT=D:\MyProjects\Capstone2   # Windows (mklink /J ile D: sürücüsü)
export CAPSTONE2_ROOT=/mnt/data/Capstone2    # Linux
```

---

## 🔧 Sorun Giderme

**CUDA bulunamadı:**
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# False çıkıyorsa: CUDA toolkit ve doğru PyTorch sürümünü yeniden yükleyin
```

**Model yüklenemiyor (`size mismatch`):**
```
FC katmanı Dropout sırası training kodunda farklı olabilir.
models.py içindeki CustomCNN._fc tanımını checkpoint ile karşılaştırın.
```

**Ollama bağlantı hatası:**
```bash
ollama serve          # Ollama servisini başlat
ollama pull moondream # Modeli indir
ollama list           # Yüklü modelleri listele
```

**OpenWebUI API'ye erişemiyor:**
```
GARBAGE_API_URL ortam değişkenini kontrol edin.
api_service.py'nin 0.0.0.0 üzerinde dinlediğini doğrulayın.
Windows Güvenlik Duvarı'nda 8000 portunu açın.
```

---

## 📊 Beklenen Performans (RTX 3060 Ti)

| Model | Doğruluk | Çıkarım Süresi |
|-------|----------|----------------|
| custom_cnn | ~88% | ~15ms |
| resnet50 | ~93% | ~25ms |
| efficientnet_b0 | ~91% | ~20ms |
| mobilenet_v3 | ~89% | ~10ms |
| moondream (VLM) | nitel | ~800ms |

---

## 📄 Lisans

MIT — Eğitim ve araştırma amaçlı serbestçe kullanılabilir.
