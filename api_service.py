"""
api_service.py — FastAPI tabanlı çıkarım servisi.
RTX 3060 Ti / CUDA öncelikli. 4 model startup'ta GPU'ya yüklenir.

Endpointler:
  GET  /health           → Servis durumu + yüklü modeller
  GET  /predict          → Tek model, varsayılan test görseli
  GET  /predict/all      → 4 model yarışması, varsayılan test görseli
  POST /predict/upload   → Tek model, yüklenen görsel
  POST /predict/all/upload → 4 model yarışması, yüklenen görsel
"""

# ─── CUDA Kontrolü ve Cihaz Ayarı (DÜZELTİLDİ) ───────────────────────────────
import torch
# Değişkenleri en başta tanımlıyoruz ki NameError almayalım
_CUDA_AVAILABLE = torch.cuda.is_available()

# LLaVA/Ollama GPU'yu rahat kullansın diye CNN modellerini CPU'ya zorluyoruz
DEVICE = torch.device("cpu") 

# ─── Standart kütüphaneler ────────────────────────────────────────────────────
import io
import logging
import time
from pathlib import Path

# ─── Üçüncü taraf ────────────────────────────────────────────────────────────
import uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

# ─── Proje modülleri ──────────────────────────────────────────────────────────
from config import CLASS_NAMES, DEFAULT_TEST_IMAGE, API_HOST, API_PORT, get_model_path
from models import SUPPORTED_MODELS, load_trained_model

# ─── Loglama ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("api_service")

# Başlangıç loglarını DEVICE ve _CUDA_AVAILABLE üzerinden güvenle yazıyoruz
log.info(f"🚀 Hesaplama cihazı : {DEVICE}")
if _CUDA_AVAILABLE:
    # CUDA mevcutsa GPU ismini ve VRAM bilgisini hala loglayabiliriz (bilgi amaçlı)
    log.info(f"🎮 GPU (Sistemde)   : {torch.cuda.get_device_name(0)}")
    log.info(f"💾 VRAM (Toplam)    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log.info("ℹ️  Not: GPU mevcut ancak modeller CPU'ya yönlendirildi (VRAM tasarrufu).")
else:
    log.warning("⚠️  CUDA bulunamadı — CPU kullanılıyor (performans düşük olacak)")

# ─── FastAPI uygulaması ───────────────────────────────────────────────────────
app = FastAPI(
    title="Alper Garbage AI — Model Arena API",
    version="2.0.0",
    description="4 model + Moondream VLM ile çöp sınıflandırma yarışması"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model havuzu ─────────────────────────────────────────────────────────────
_model_registry: dict = {}

# ─── Görüntü ön işleme ───────────────────────────────────────────────────────
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def _image_to_tensor(img: Image.Image) -> torch.Tensor:
    return _TRANSFORM(img.convert("RGB")).unsqueeze(0).to(DEVICE)

def _run_inference(model_name: str, tensor: torch.Tensor) -> dict:
    """Tek model üzerinde çıkarım yapar; süreyi ms cinsinden döndürür."""
    net = _model_registry[model_name]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = net(tensor)
        prob = torch.nn.functional.softmax(out, dim=1)
        conf, idx = torch.max(prob, 1)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "prediction": CLASS_NAMES[idx.item()],
        "confidence": round(float(conf.item() * 100), 2),
        "inference_ms": round(elapsed_ms, 2),
    }


# ─── Startup: modelleri GPU'ya (Bu projede CPU'ya) yükle ────────────────────
@app.on_event("startup")
def load_models():
    log.info("📦 Modeller yükleniyor...")
    for name in SUPPORTED_MODELS:
        try:
            path = get_model_path(name)
            if path.exists():
                # Buradaki DEVICE artık güvenle 'cpu' olarak gidiyor
                _model_registry[name] = load_trained_model(name, path, DEVICE)
                log.info(f"  ✅ {name:20s} → {path.name}")
            else:
                log.warning(f"  ❌ {name:20s} → dosya bulunamadı: {path}")
        except Exception as exc:
            log.error(f"  💥 {name} yüklenemedi: {exc}")
    log.info(f"✔️  Yüklenen model sayısı: {len(_model_registry)}/{len(SUPPORTED_MODELS)}")


# ─── /health ──────────────────────────────────────────────────────────────────
@app.get("/health", summary="Servis durumu")
async def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "cuda_available": _CUDA_AVAILABLE,
        "gpu_name": torch.cuda.get_device_name(0) if _CUDA_AVAILABLE else None,
        "loaded_models": list(_model_registry.keys()),
        "missing_models": [m for m in SUPPORTED_MODELS if m not in _model_registry],
    }


# ─── /predict (tek model, varsayılan görsel) ──────────────────────────────────
@app.get("/predict", summary="Tek model — varsayılan test görseli")
async def predict(model: str = Query(default="custom_cnn", description="Model adı")):
    if model not in _model_registry:
        raise HTTPException(status_code=404, detail=f"Model yüklü değil: '{model}'. Mevcut: {list(_model_registry)}")

    img_path = Path(DEFAULT_TEST_IMAGE)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Test görseli bulunamadı: {img_path}")

    tensor = _image_to_tensor(Image.open(img_path))
    result = _run_inference(model, tensor)
    return {"model": model, "image": str(img_path), **result, "device": str(DEVICE)}


# ─── /predict/all (tüm modeller, varsayılan görsel) ───────────────────────────
import base64 # Dosyanın en üstüne import base64 eklemeyi unutma

# ─── /predict/all (tüm modeller, varsayılan görsel) ───────────────────────────
@app.get("/predict/all", summary="Tüm modeller — varsayılan test görseli (Arena)")
async def predict_all():
    if not _model_registry:
        raise HTTPException(status_code=503, detail="Hiç model yüklenmedi")

    img_path = Path(DEFAULT_TEST_IMAGE)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Test görseli bulunamadı: {img_path}")

    # Görüntüyü tensora çevirip CNN'lere gönder
    tensor = _image_to_tensor(Image.open(img_path))
    results = {}
    for name in _model_registry:
        results[name] = _run_inference(name, tensor)

    winner = max(results, key=lambda k: results[k]["confidence"])

    # YENİ EKLENEN KISIM: Resmi Base64'e çevirip JSON ile yolla
    with open(img_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode('utf-8')

    return {
        "image": str(img_path),
        "base64_image": base64_data, # Tool artık bu veriyi kullanacak!
        "device": str(DEVICE),
        "results": results,
        "winner": winner,
        "winner_prediction": results[winner]["prediction"],
        "winner_confidence": results[winner]["confidence"],
    }


# ─── /predict/upload (tek model, yüklenen görsel) ────────────────────────────
@app.post("/predict/upload", summary="Tek model — yüklenen görsel")
async def predict_upload(
    model: str = Query(default="custom_cnn"),
    file: UploadFile = File(...),
):
    if model not in _model_registry:
        raise HTTPException(status_code=404, detail=f"Model yüklü değil: '{model}'")

    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    tensor = _image_to_tensor(img)
    result = _run_inference(model, tensor)
    return {"model": model, "filename": file.filename, **result, "device": str(DEVICE)}


# ─── /predict/all/upload (tüm modeller, yüklenen görsel) ─────────────────────
@app.post("/predict/all/upload", summary="Tüm modeller — yüklenen görsel (Arena)")
async def predict_all_upload(file: UploadFile = File(...)):
    if not _model_registry:
        raise HTTPException(status_code=503, detail="Hiç model yüklenmedi")

    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    tensor = _image_to_tensor(img)

    results = {}
    for name in _model_registry:
        results[name] = _run_inference(name, tensor)

    winner = max(results, key=lambda k: results[k]["confidence"])

    return {
        "filename": file.filename,
        "device": str(DEVICE),
        "results": results,
        "winner": winner,
        "winner_prediction": results[winner]["prediction"],
        "winner_confidence": results[winner]["confidence"],
    }


# ─── Giriş noktası ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)