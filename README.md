# 🗑️ Garbage Classification — Hybrid AI Arena

> **Local CNN vs Vision Language Model** — Eğitilmiş atık sınıflandırma modellerini FastAPI + OpenWebUI üzerinden gerçek zamanlı karşılaştırın. Dört CNN mimarisi + LLaVA VLM ile **Model Arena** deneyimi.

---

## 📌 Proje Özeti

Bu proje iki ana katmandan oluşur:

1. **Eğitim Katmanı** — 6 kategorili atık sınıflandırması için Custom CNN ve Transfer Learning modelleri eğitilir ve `.pth` checkpoint dosyaları üretilir.
2. **Servis & Arena Katmanı** — Eğitilen dört model bir FastAPI sunucusu üzerinden canlı endpointler olarak yayınlanır; her tahmin isteğinde LLaVA Vision-Language Model hakem rolüne girerek modelleri sıralar ve bir **Kompozit Skor** hesaplanır.

| Kategori | Türkçe Karşılığı |
|---|---|
| `cardboard` | Karton / mukavva |
| `glass` | Cam |
| `metal` | Metal |
| `paper` | Kağıt |
| `plastic` | Plastik |
| `trash` | Diğer çöp |

---

## 🏗️ Sistem Mimarisi

```
┌──────────────────────────────────────────────────────────────┐
│                      KULLANICI KATMANI                       │
│                OpenWebUI  (localhost:3000)                    │
│            Tarayıcı / Sunum / Mobil İzleme                   │
└──────────────────────┬───────────────────────────────────────┘
                       │  HTTP / Tools API  (openwebui_tool.py)
           ┌───────────┴────────────┐
           │                        │
┌──────────▼─────────┐   ┌──────────▼──────────────────────────┐
│   FastAPI Servisi   │   │         Ollama Servisi              │
│   localhost:8000    │   │        localhost:11434              │
│  (api_service.py)   │   │    LLaVA / llava-llama3 / bakllava  │
│                     │   │          (GPU — VRAM)               │
│  GET  /health       │   └─────────────────────────────────────┘
│  GET  /predict      │
│  GET  /predict/all  │  ← Base64 JSON pipeline (Docker-safe)
│  POST /predict/upload│
│  POST /predict/all/ │
│        upload       │
└──────────┬──────────┘
           │  map=cpu  (VRAM, LLaVA'ya bırakıldı)
┌──────────▼───────────────────────────────────────────────────┐
│                   Yerel Model Havuzu (.pth)                  │
│   custom_cnn  │  resnet50  │  efficientnet_b0  │  mobilenet_v3│
│                  CPU inference (RAM)                          │
│              ← NVIDIA RTX 3060 Ti → LLaVA (GPU) →           │
└──────────────────────────────────────────────────────────────┘
```

### Veri Akışı — Base64 Pipeline

```
Kullanıcı isteği
      │
      ▼
FastAPI /predict/all
      │
      ├─► CNN x4  →  { prediction, confidence, inference_ms }
      │
      ├─► Görsel  →  base64.b64encode()  →  JSON içine gömülür
      │
      ▼
openwebui_tool.py
      │
      ├─► Ollama /api/generate  (images: [base64_str])
      │         └─► LLaVA  →  CoT açıklama + sınıf tahmini
      │
      └─► Ollama /api/generate  (Hakem sıralaması)
                └─► LLaVA  →  Model ranking listesi
                      │
                      ▼
              Composite Score = CNN_conf × 0.60 + VLM_rank × 0.40
                      │
                      ▼
              🏆 Arena Sonuç Tablosu  →  OpenWebUI Sohbet
```

> **Neden Base64?** Docker konteynerlerinde her süreç kendi dosya sistemi ad alanında çalışır. `openwebui_tool.py` FastAPI konteynerin diske yazdığı görsele erişemez. Görüntüyü doğrudan JSON gövdesine Base64 olarak gömerek bu izolasyon problemi tamamen ortadan kaldırılır — paylaşımlı volume, ortam değişkeni veya ağ dosya sistemi gerekmez.

---

## 📁 Dosya Yapısı

```
Capstone2/
│
├── config.py                        # ⚙️  Platform-aware merkezi yapılandırma
│                                    #     (Windows / macOS / Linux / Docker)
│
├── models.py                        # 🧠  Paylaşılan model mimarileri
│                                    #     CustomCNN, ResNet50, EfficientNet, MobileNet
│
├── api_service.py                   # 🚀  FastAPI çıkarım & arena sunucusu
│                                    #     Startup'ta 4 model RAM'e yüklenir
│
├── arena_battle.py                  # 🏟️  Terminal tabanlı CNN vs VLM karşılaştırması
│
├── openwebui_tool.py                # 🔌  OpenWebUI Tools entegrasyon sınıfı
│                                    #     LLaVA CoT + Composite Score hesaplama
│
├── garbage_classification_complete.py  # 📊  Eğitim scripti & veri analizi
│
├── requirements.txt                 # 📦  Python bağımlılıkları
│
└── results/
    ├── models/                      # 💾  .pth checkpoint dosyaları
    │   ├── custom_cnn_best.pth
    │   ├── resnet50_best.pth
    │   ├── efficientnet_b0_best.pth
    │   └── mobilenet_v3_best.pth
    ├── analysis/                    # 📈  Veri seti istatistikleri & raporlar
    └── visualizations/              # 🖼️  Eğitim grafikleri & karşılaştırma görselleri
```

---

## 🤖 Model Bilgileri

### CNN Modelleri

| Model Adı | Mimari | Parametreler | Eğitim | Notlar |
|---|---|---|---|---|
| `custom_cnn` | 4-blok özel CNN (Alper-CNN) | ~7.5M | Sıfırdan | Baseline model |
| `resnet50` | ResNet-50 (fine-tuned) | ~23.5M | Transfer | En yüksek doğruluk |
| `efficientnet_b0` | EfficientNet-B0 (timm) | ~5.3M | Transfer | En verimli oran |
| `mobilenet_v3` | MobileNetV3-Small | ~2.5M | Transfer | Hız odaklı |

### VLM (Vision-Language Model)

| Model | Sağlayıcı | Çalışma Yeri | Görev |
|---|---|---|---|
| `llava` | Ollama (yerel) | GPU (RTX 3060 Ti) | CoT açıklama + hakem sıralama |

### Alper-CNN (CustomCNN) Mimarisi

```
Input  (3 × 224 × 224)
   │
   ├─ Conv2d(3→32,  k=3, bias=True) + BN + ReLU + MaxPool2d(2) → 32×112×112
   ├─ Conv2d(32→64, k=3, bias=True) + BN + ReLU + MaxPool2d(2) → 64×56×56
   ├─ Conv2d(64→128,k=3, bias=True) + BN + ReLU + MaxPool2d(2) → 128×28×28
   ├─ Conv2d(128→256,k=3,bias=True) + BN + ReLU + MaxPool2d(2) → 256×14×14
   │
   └─ Flatten(256×14×14=50176) → Linear(50176→512) → ReLU → Dropout(0.5)
                                → Linear(512→6)
```

> ⚠️ **Kritik:** `Conv2d` bloklarında `bias=True` **zorunludur**. Checkpoint bu parametrelerle kaydedilmiştir; `bias=False` ile oluşturulan model `unexpected key` hatası verir.

---

## ⚡ Hızlı Başlangıç

### Ön Koşullar

- Python 3.11+
- NVIDIA GPU (CUDA 12.x destekli) — RTX 3060 Ti önerilir
- [Ollama](https://ollama.com) kurulu ve çalışıyor olmalı
- [OpenWebUI](https://github.com/open-webui/open-webui) kurulu (isteğe bağlı, Arena UI için)

### 1. Python Ortamı Kurulumu

```bash
# Sanal ortam oluştur
python -m venv venv

# Aktifleştir
.\venv\Scripts\activate      # Windows
source venv/bin/activate     # macOS / Linux

# CUDA 12.1 için PyTorch (RTX 3060 Ti)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Diğer bağımlılıklar
pip install -r requirements.txt
```

### 2. Yapılandırmayı Doğrula

```bash
python config.py
```

Beklenen çıktı:
```
OS           : Windows
PROJECT_ROOT : C:\Users\<user>\PROJELER\Capstone2
MODELS_DIR   : C:\Users\<user>\PROJELER\Capstone2\results\models
  ✅  custom_cnn           → ...\models\custom_cnn_best.pth
  ✅  resnet50             → ...\models\resnet50_best.pth
  ✅  efficientnet_b0      → ...\models\efficientnet_b0_best.pth
  ✅  mobilenet_v3         → ...\models\mobilenet_v3_best.pth
```

### 3. Modelleri Eğit (İlk Kurulum)

```bash
python garbage_classification_complete.py
# → Eğitim tamamlandığında results/models/ altında .pth dosyaları oluşur
```

### 4. FastAPI Sunucusunu Başlat

```bash
python api_service.py
# → http://localhost:8000         (API)
# → http://localhost:8000/docs    (Swagger UI — otomatik dokümantasyon)
# → http://localhost:8000/health  (Sağlık kontrolü)
```

Başarılı başlatma logu:
```
INFO  🚀 Hesaplama cihazı : cpu
INFO  🎮 GPU (Sistemde)   : NVIDIA GeForce RTX 3060 Ti
INFO  💾 VRAM (Toplam)    : 8.0 GB
INFO  ℹ️  Not: GPU mevcut ancak modeller CPU'ya yönlendirildi (VRAM tasarrufu).
INFO  📦 Modeller yükleniyor...
INFO    ✅ custom_cnn          → custom_cnn_best.pth
INFO    ✅ resnet50            → resnet50_best.pth
INFO    ✅ efficientnet_b0     → efficientnet_b0_best.pth
INFO    ✅ mobilenet_v3        → mobilenet_v3_best.pth
INFO  ✔️  Yüklenen model sayısı: 4/4
```

### 5. Ollama & LLaVA Kurulumu

```bash
# Ollama servisini başlat (zaten kuruluysa)
ollama serve

# LLaVA modelini indir (ilk seferinde)
ollama pull llava

# Alternatifler:
# ollama pull llava-llama3   # Daha güçlü, daha yavaş
# ollama pull bakllava       # Alternatif VLM

# Yüklü modelleri kontrol et
ollama list
```

### 6. Terminal Arena'yı Çalıştır

```bash
# Tüm modeller + LLaVA, varsayılan test görseli
python arena_battle.py

# Belirli bir görsel ile
python arena_battle.py --image C:\Users\<user>\Desktop\test_image.png

# Sadece CNN (VLM atla)
python arena_battle.py --no-vlm

# Belirli CNN modelleri
python arena_battle.py --models custom_cnn resnet50

# Sonuçları JSON dosyasına kaydet
python arena_battle.py --save results/arena_results.json
```

---

## 🔌 OpenWebUI Entegrasyonu

### Tool Kurulumu

1. OpenWebUI'yu açın (`http://localhost:3000`)
2. **Settings** → **Tools** → **"+ New Tool"**
3. `openwebui_tool.py` dosyasının tüm içeriğini yapıştırın
4. Dosyanın üst kısmındaki URL'leri kendi sunucu adresinize göre güncelleyin:
   ```python
   self.API_BASE_URL    = "http://<YOUR_SERVER_IP>:8000"
   self.OLLAMA_BASE_URL = "http://<YOUR_SERVER_IP>:11434"
   self.OLLAMA_MODEL    = "llava"   # veya llava-llama3
   ```
5. **Save** → Sohbet ekranına dönün → 🔧 araç simgesini aktifleştirin

### Desteklenen Doğal Dil Komutları

| Kullanıcı Komutu | Çağrılan Fonksiyon | Açıklama |
|---|---|---|
| `arena karşılaştır` / `arena battle` | `arena_battle()` | Tüm CNN'ler + LLaVA hakem |
| `sağlık kontrolü` / `check api` | `arena_battle()` → `/health` | Servis & model durumu |

### Tool Çıktısı Örneği

```markdown
## 🏆 Model Arena — Kesin Sonuçlar
**Görsel:** `/home/user/test_image.png`

| # | Model | Tahmin | Güven | Rank Sk | Kompozit | Süre |
|---|-------|--------|-------|---------|----------|------|
| 1 | **efficientnet_b0** | 🧴 plastic | %91.4 | 100 | **94.8** | 18.3 ms |
| 2 | **resnet50**        | 🧴 plastic | %88.2 |  67 | **79.6** | 52.1 ms |
| 3 | **mobilenet_v3**    | 🧴 plastic | %79.6 |  33 | **61.1** |  9.7 ms |
| 4 | **custom_cnn**      | 🔩 metal   | %61.3 |   0 | **36.8** | 24.4 ms |

---
### 🌙 LLaVA Hakem
**Gerekçe:** This appears to be a clear plastic bottle with a smooth, 
transparent surface — typical of PET plastic containers.
**Tahmini:** 🧴 **PLASTIC**
```

---

## 🌐 API Referansı

Sunucu çalışırken `http://localhost:8000/docs` adresinden interaktif Swagger UI'ya erişin.

### Endpoint Tablosu

| Method | Path | Açıklama |
|---|---|---|
| `GET` | `/health` | Servis durumu, yüklü modeller, CUDA bilgisi |
| `GET` | `/predict` | Tek model — varsayılan test görseli |
| `GET` | `/predict/all` | Tüm 4 model (Arena) — varsayılan test görseli |
| `POST` | `/predict/upload` | Tek model — yüklenen görsel |
| `POST` | `/predict/all/upload` | Tüm 4 model (Arena) — yüklenen görsel |

### `/health` Yanıt Örneği

```json
{
  "status": "ok",
  "device": "cpu",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3060 Ti",
  "loaded_models": ["custom_cnn", "resnet50", "efficientnet_b0", "mobilenet_v3"],
  "missing_models": []
}
```

### `/predict` Sorgu Parametreleri

| Parametre | Tip | Varsayılan | Açıklama |
|---|---|---|---|
| `model` | `string` | `custom_cnn` | `custom_cnn`, `resnet50`, `efficientnet_b0`, `mobilenet_v3` |

### `/predict` Yanıt Örneği

```json
{
  "model": "efficientnet_b0",
  "image": "/home/user/test_image.png",
  "prediction": "plastic",
  "confidence": 91.37,
  "inference_ms": 18.34,
  "device": "cpu"
}
```

### `/predict/all` Yanıt Yapısı

```json
{
  "image": "/home/user/test_image.png",
  "base64_image": "<base64_encoded_string>",
  "device": "cpu",
  "results": {
    "custom_cnn":      { "prediction": "metal",   "confidence": 61.30, "inference_ms": 24.4 },
    "resnet50":        { "prediction": "plastic",  "confidence": 88.21, "inference_ms": 52.1 },
    "efficientnet_b0": { "prediction": "plastic",  "confidence": 91.37, "inference_ms": 18.3 },
    "mobilenet_v3":    { "prediction": "plastic",  "confidence": 79.64, "inference_ms":  9.7 }
  },
  "winner": "efficientnet_b0",
  "winner_prediction": "plastic",
  "winner_confidence": 91.37
}
```

> `base64_image` alanı, OpenWebUI Tool tarafından LLaVA'ya doğrudan iletilir. Bu alan sayesinde herhangi bir dosya yolu paylaşımına gerek kalmadan görsel, JSON içinden VLM'e ulaşır.

---

## 🏟️ Model Arena & Kompozit Skor

### Puanlama Mekanizması

Arena sonucu iki bileşenin ağırlıklı toplamıyla belirlenir:

```
Composite Score(m) = CNN_Confidence(m) × 0.60 + VLM_Rank_Score(m) × 0.40
```

**VLM Rank Score** hesaplaması:

```
VLM_Rank_Score(m) = 100 × (1 - (rank(m) - 1) / (N - 1))
```

| Sıralama | VLM Rank Score |
|---|---|
| 1. (En iyi) | 100.0 |
| 2. | 66.7 |
| 3. | 33.3 |
| 4. (En kötü) | 0.0 |

**Ağırlık Gerekçesi:**
- CNN confidence (%60) daha güvenilir ve deterministik bir sinyal olduğu için baskındır.
- VLM rank skoru (%40) yeterince ağırdır ki, hafif üstün ama hatalı bir CNN'i doğru sıralama ile geçersiz kılabilsin — bu, tek modelli sisteme kıyasla önemli bir hata denetimi sağlar.

### LLaVA — Chain-of-Thought (CoT) Prompting

LLaVA iki ayrı istekle sorgulanır:

**1. Sınıflandırma & Açıklama:**
```
"Describe this waste item in one short sentence. 
Is it cardboard, glass, metal, paper, plastic, or trash?"
```

**2. Hakem Sıralaması:**
```
"Rank these models from best to worst based on the image: 
[custom_cnn, resnet50, efficientnet_b0, mobilenet_v3]. 
Reply with names only."
```

LLaVA'nın yanıtından geçerli sınıf adı (`cardboard`, `glass`, ... ) regex ile çıkarılır. Eşleşme yoksa `"bilinmiyor"` döner.

---

## ⚙️ Yapılandırma (`config.py`)

`config.py` işletim sistemini otomatik algılar; `CAPSTONE2_ROOT` ortam değişkeniyle CI/CD veya Docker için özelleştirilebilir.

| Değişken | Açıklama | Örnek Değer |
|---|---|---|
| `PROJECT_ROOT` | Proje kök dizini | `~/PROJELER/Capstone2` |
| `DATA_DIR` | Ham görsel veri seti | `PROJECT_ROOT/Garbage classification` |
| `MODELS_DIR` | .pth checkpoint dosyaları | `PROJECT_ROOT/results/models` |
| `DEFAULT_TEST_IMAGE` | Varsayılan test görseli | `~/Desktop/test_image.png` |
| `CLASS_NAMES` | Sınıf etiket listesi | `["cardboard", ..., "trash"]` |
| `API_HOST` / `API_PORT` | FastAPI bağlama adresi | `0.0.0.0` / `8000` |
| `OLLAMA_VLM_MODEL` | Aktif VLM model adı | `"llava"` |

### Platform Yol Davranışı

| İşletim Sistemi | `PROJECT_ROOT` | Test Görseli |
|---|---|---|
| Windows | `~\PROJELER\Capstone2` | `~\Desktop\test_image.png` |
| macOS | `~/Projects/Capstone2` | `~/Desktop/test_image.png` |
| Linux / Docker | `~/Capstone2` | `~/test_image.png` |

```bash
# Ortam değişkeniyle özelleştir (Docker / CI)
export CAPSTONE2_ROOT=/mnt/data/Capstone2    # Linux
set    CAPSTONE2_ROOT=D:\MyProjects\Capstone2 # Windows
```

---

## 📊 Eğitim Detayları

### Hiperparametreler

| Parametre | Değer |
|---|---|
| Görsel boyutu | 224 × 224 px |
| Batch size | 32 |
| Maksimum epoch | 30 |
| Başlangıç öğrenme hızı | 0.001 |
| Optimizatör | Adam (β₁=0.9, β₂=0.999) |
| Loss fonksiyonu | CrossEntropyLoss |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early stopping | Patience = 7 epoch |

### Veri Artırma (Augmentation) — Yalnızca Eğitim

| Teknik | Parametre |
|---|---|
| Random Horizontal Flip | p = 0.5 |
| Random Rotation | ±15° |
| Color Jitter | brightness/contrast/saturation ±0.2 |
| Random Affine | Translation ±10% |
| Normalize (ImageNet) | mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] |

### Veri Seti Dağılımı

| Sınıf | Görüntü Sayısı | Oran |
|---|---|---|
| cardboard | 403 | %15.9 |
| glass | 501 | %19.8 |
| metal | 410 | %16.2 |
| paper | 594 | %23.5 |
| plastic | 482 | %19.1 |
| trash | 137 | %5.4 ⚠️ |
| **Toplam** | **2,527** | **%100** |

> ⚠️ `trash` sınıfı dengesiz veri nedeniyle tüm modellerde daha düşük recall gösterir.

---

## 📈 Beklenen Performans

### Doğruluk & Hız Karşılaştırması

| Model | Doğruluk | Precision | Recall | F1-Score | Çıkarım Süresi (CPU) |
|---|---|---|---|---|---|
| `custom_cnn` | ~82.5% | ~0.824 | ~0.825 | ~0.823 | ~20–35 ms |
| `resnet50` | ~91.2% | ~0.913 | ~0.912 | ~0.912 | ~40–80 ms |
| `efficientnet_b0` | ~90.1% | ~0.902 | ~0.900 | ~0.901 | ~15–25 ms |
| `mobilenet_v3` | ~87.6% | ~0.877 | ~0.875 | ~0.876 | ~8–15 ms |
| `llava` (VLM) | nitel | — | — | — | ~2,000–8,000 ms (GPU) |

> **Not:** CNN modelleri VRAM tasarrufu amacıyla kasıtlı olarak CPU'ya alınmıştır; GPU'ya taşınırlarsa çıkarım süreleri 3–5× azalır.

### Deployment Senaryosu Önerileri

| Senaryo | Önerilen Model |
|---|---|
| En yüksek doğruluk | `resnet50` |
| Verimlilik (doğruluk/parametre) | `efficientnet_b0` |
| Gerçek zamanlı / edge cihaz | `mobilenet_v3` |
| Açıklanabilirlik (XAI) | `llava` + herhangi CNN |
| Eğitim / baseline | `custom_cnn` |

---

## 🔧 Sorun Giderme

### CUDA Bulunamadı

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# False → CUDA toolkit ve PyTorch sürümünü yeniden kontrol edin
```

### Model Yüklenemedi — `size mismatch`

```
FC katmanı Dropout sırası checkpoint ile uyumsuz olabilir.
models.py içindeki CustomCNN.fc tanımını checkpoint state_dict ile karşılaştırın.
bias=True parametresinin Conv2d bloklarında set edildiğinden emin olun.
```

### Model Yüklenemedi — `unexpected key`

```
Checkpoint bias tensörleriyle kaydedilmiş; model bias=False ile tanımlanmış.
models.py içinde Conv2d(..., bias=True) olduğunu doğrulayın.
```

### Ollama Bağlantı Hatası

```bash
ollama serve              # Servisi başlat
ollama pull llava         # Modeli indir (ilk kullanımda)
ollama list               # Kurulu modelleri listele
curl http://localhost:11434/api/tags  # HTTP ile test
```

### OpenWebUI FastAPI'ye Erişemiyor

```
1. openwebui_tool.py içindeki API_BASE_URL adresini kontrol edin.
2. api_service.py'nin 0.0.0.0 üzerinde dinlediğini doğrulayın (localhost değil).
3. Güvenlik duvarında 8000 portunu açın.
4. Docker kullanıyorsanız port yönlendirmesini (-p 8000:8000) kontrol edin.
```

### `timm` Bulunamadı (EfficientNet)

```bash
pip install timm
```

### Docker'da Dosya Yolu Hatası

```
Bu proje Base64 pipeline ile bu sorunu çözmüştür.
/predict/all endpoint'i görseli diske yazmak yerine JSON içine gömer.
openwebui_tool.py dosyasını güncel (base64_image destekli) versiyonla kullandığınızdan emin olun.
```

---

## 🖥️ Gereksinimler

```
torch>=2.0.0
torchvision>=0.15.0
fastapi>=0.100.0
uvicorn>=0.23.0
pillow>=9.0.0
timm>=0.9.0
python-multipart>=0.0.6
```

**Opsiyonel (eğitim & analiz için):**
```
scikit-learn
matplotlib
seaborn
pandas
numpy
```

---

## 📄 Lisans

MIT — Eğitim ve araştırma amaçlı serbestçe kullanılabilir.

---

*Bu proje, Capstone II dersi kapsamında geliştirilmiştir. Tüm modeller yerel olarak eğitilmiş ve yerel olarak dağıtılmıştır; harici API veya bulut servisi kullanılmamıştır.*
