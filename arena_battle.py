"""
arena_battle.py — Terminal hızlı test scripti (v3.0 — CoT + Bug Fixes)
=======================================================================
Düzeltmeler (v2 → v3):
  1. _ollama_post: NDJSON parsing düzeltildi (tek JSON veya satır-satır JSON)
  2. _ollama_post: Hata durumunda ham yanıt ekrana basılıyor (debug)
  3. run_moondream_cot: 3 adımlı Chain of Thought (betimleme → sınıf → gerekçe)
  4. run_moondream_judge: CoT betimlemesini bağlam olarak kullanıyor
  5. _normalize_model_name: Moondream'in farklı isim varyantlarını eşleştiriyor
  6. compute_composite_scores: CNN güveni (%60) + Moondream sıralaması (%40)
  7. print_scoreboard: CoT çıktısı + kompozit skor + renk kodlaması

Kullanım:
  python arena_battle.py
  python arena_battle.py --image C:\\Users\\alper\\Desktop\\test.png
  python arena_battle.py --no-vlm        # Moondream'i atla
  python arena_battle.py --debug-ollama  # Ham Ollama yanıtını göster
"""

import argparse
import base64
import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ─── Cihaz ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")

# ─── Proje modülleri ──────────────────────────────────────────────────────────
from config import (
    CLASS_NAMES, DEFAULT_TEST_IMAGE, OLLAMA_VLM_MODEL,
    MODEL_FILES, get_model_path,
)
from models import load_trained_model

# ─── Sabitler ─────────────────────────────────────────────────────────────────
VALID_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
OLLAMA_URL    = "http://localhost:11434/api/generate"

# Debug flag — argparse ile set edilir
_DEBUG_OLLAMA = False

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─── Model ismi normalizasyon tablosu ─────────────────────────────────────────
# Moondream hakem yanıtında farklı varyantlar kullanabilir; hepsini iç isme çevirir.
_NAME_MAP: dict[str, str] = {
    # custom_cnn
    "custom_cnn": "custom_cnn", "customcnn": "custom_cnn",
    "alpercnn":   "custom_cnn", "alper":     "custom_cnn",
    "alper-cnn":  "custom_cnn", "alpercnn":  "custom_cnn",
    "custom":     "custom_cnn",
    # resnet50
    "resnet50":   "resnet50",   "resnet":    "resnet50",
    "resnet-50":  "resnet50",   "res50":     "resnet50",
    # efficientnet_b0
    "efficientnet_b0": "efficientnet_b0", "efficientnet": "efficientnet_b0",
    "efficient":       "efficientnet_b0", "effnet":        "efficientnet_b0",
    "efficientnetb0":  "efficientnet_b0", "efficientnet-b0": "efficientnet_b0",
    # mobilenet_v3
    "mobilenet_v3": "mobilenet_v3", "mobilenet": "mobilenet_v3",
    "mobile":       "mobilenet_v3", "mobilenetv3": "mobilenet_v3",
    "mobilenet-v3": "mobilenet_v3",
}


def _normalize_model_name(raw: str) -> str | None:
    """
    Moondream'in hakem yanıtındaki model adını iç isme çevirir.
    Bilinmeyen isimler None döndürür.
    """
    key = re.sub(r"[^a-z0-9]", "", raw.lower())
    return _NAME_MAP.get(key)


# ─── Görsel base64 kodlama ────────────────────────────────────────────────────
def _encode_image(image_path: Path) -> str:
    """Görseli base64 string olarak döndürür."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ─── CNN çıkarım ─────────────────────────────────────────────────────────────
def run_cnn(model_name: str, image_path: Path) -> dict:
    path = get_model_path(model_name)
    if not path.exists():
        return {"model": model_name, "prediction": "N/A", "confidence": 0.0,
                "ms": 0.0, "error": True}

    model  = load_trained_model(model_name, path, DEVICE)
    img    = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
    ms = (time.perf_counter() - t0) * 1000

    return {
        "model":      model_name,
        "prediction": CLASS_NAMES[idx.item()],
        "confidence": conf.item() * 100,
        "ms":         ms,
        "error":      False,
    }


# ─── Ollama HTTP yardımcısı ───────────────────────────────────────────────────
def _ollama_post(payload: dict, timeout: int = 90) -> dict | None:
    """
    Ollama REST API'sine POST gönderir.

    DÜZELTME: Ollama bazı versiyonlarda 'stream: false' olsa bile NDJSON
    (satır-satır JSON) döndürebilir. Bu fonksiyon her iki formatı da işler:
      1. Tek JSON nesnesi  → doğrudan parse et
      2. NDJSON           → done=true olan satırı bul, yoksa son geçerli satırı al
    """
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        print(f"  ⚠️  Ollama bağlantı hatası: {exc}")
        print(f"       → Ollama'nın çalıştığını kontrol edin: ollama serve")
        return None

    if _DEBUG_OLLAMA:
        preview = raw[:500].replace("\n", "\\n")
        print(f"  [DEBUG] Ollama ham yanıt ({len(raw)} byte): {preview}")

    # ── Önce tek JSON olarak dene (en yaygın durum) ───────────────────────────
    stripped = raw.strip()
    if stripped:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass  # NDJSON olabilir, aşağıda dene

    # ── NDJSON: satır satır parse et ──────────────────────────────────────────
    last_valid: dict | None = None
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            last_valid = obj
            if obj.get("done") is True:  # tamamlanmış satır bulundu
                return obj
        except json.JSONDecodeError:
            continue  # bozuk satırı atla

    if last_valid is not None:
        return last_valid

    # ── Hiçbir şey çözülemediyse ham içeriği logla ───────────────────────────
    print(f"  ⚠️  Ollama yanıtı parse edilemedi.")
    print(f"       Ham içerik (ilk 300 karakter): {raw[:300]!r}")
    print(f"       İpucu: --debug-ollama bayrağı ile tam yanıtı görebilirsiniz.")
    return None


# ─── Moondream: Chain of Thought (3 Adım) ────────────────────────────────────
def run_moondream_cot(image_path: Path) -> tuple[str, str, str]:
    """
    Moondream'i 3 adımlı Chain of Thought ile çalıştırır.

    Adım 1 — Betimleme  : "Bu görseldeki ana nesneyi bir cümleyle tanımla."
    Adım 2 — Sınıflandırma : "Betimene dayanarak SINIFLAR'dan birini seç."
    Adım 3 — Gerekçe    : "Neden bu sınıf? Kısa bir açıklama."

    Küçük VLM'ler (Moondream ~1.6B) tek büyük prompt'ta tutarsız yanıt verebilir.
    Adım adım gitmek her adımda odağı daraltır ve doğruluğu artırır.

    Döndürür:
        (description, prediction, reasoning)
    """
    img_b64 = _encode_image(image_path)

    # ── Adım 1: Görsel Betimleme ──────────────────────────────────────────────
    print("  🌙  [CoT 1/3] Görsel betimleniyor...")
    step1 = _ollama_post({
        "model":  OLLAMA_VLM_MODEL,
        "prompt": (
            "Look at this image. In ONE short sentence, describe the main object: "
            "what it is, what material it appears to be made of, and its color. "
            "Be specific and factual."
        ),
        "images": [img_b64],
        "stream": False,
    }, timeout=60)

    description = "Betimlenemedi"
    if step1 and "response" in step1:
        description = step1["response"].strip()
        # 200 karakteri aşan yanıtı kısalt (Moondream bazen çok uzun yazar)
        if len(description) > 200:
            description = description[:197] + "..."

    # ── Adım 2: Sınıflandırma (betim bağlamıyla) ─────────────────────────────
    print("  🌙  [CoT 2/3] Sınıflandırılıyor...")
    step2 = _ollama_post({
        "model":  OLLAMA_VLM_MODEL,
        "prompt": (
            f"You described this image as: \"{description}\"\n\n"
            "Now classify the waste material in the image. "
            "Choose EXACTLY ONE word from this list: "
            "cardboard, glass, metal, paper, plastic, trash\n\n"
            "Reply with ONE WORD ONLY. No sentences, no punctuation."
        ),
        "images": [img_b64],
        "stream": False,
    }, timeout=60)

    raw_class  = ""
    prediction = "bilinmiyor"
    if step2 and "response" in step2:
        raw_class = step2["response"].strip().lower()
        words = re.sub(r"[^a-z]", " ", raw_class).split()
        for w in words:
            if w in VALID_CLASSES:
                prediction = w
                break
        if prediction == "bilinmiyor":
            prediction = f"bilinmiyor ({raw_class[:25]})"

    # ── Adım 3: Kısa Gerekçe ─────────────────────────────────────────────────
    print("  🌙  [CoT 3/3] Gerekçe oluşturuluyor...")
    step3 = _ollama_post({
        "model":  OLLAMA_VLM_MODEL,
        "prompt": (
            f"You described: \"{description}\"\n"
            f"You classified it as: {prediction}\n\n"
            "In ONE short sentence, explain WHY you chose this category "
            "based on visual clues (texture, shape, color, transparency, etc.)."
        ),
        "images": [img_b64],
        "stream": False,
    }, timeout=60)

    reasoning = "Gerekçe alınamadı"
    if step3 and "response" in step3:
        r = step3["response"].strip()
        reasoning = r[:200] + "..." if len(r) > 200 else r

    return description, prediction, reasoning


# ─── Moondream: Hakem / Sıralayıcı (CoT bağlamlı) ───────────────────────────
def run_moondream_judge(
    image_path: Path,
    cnn_results: list,
    cot_description: str = "",
) -> tuple[str, list[str]]:
    """
    CNN sonuçlarını Moondream'e gösterir; CoT betimini bağlam olarak ekler.
    Görsele bakarak hangi modelin en doğru tahmin yaptığını sıralar.

    Döndürür:
        (ham_yanıt, normalize_edilmiş_sıralama_listesi)
    """
    print("  ⚖️  Moondream hakem olarak sıralama yapıyor...")
    img_b64 = _encode_image(image_path)

    model_lines = "\n".join(
        f"- {r['model']}: {r['prediction']} (confidence {r['confidence']:.1f}%)"
        for r in cnn_results if not r.get("error")
    )

    # CoT betimini bağlam olarak ekle (varsa)
    context_line = ""
    if cot_description and cot_description != "Betimlenemedi":
        context_line = (
            f"You previously described this image as: \"{cot_description}\"\n\n"
        )

    prompt = (
        f"{context_line}"
        "You are judging an image classification contest. "
        "Look at the image carefully.\n\n"
        "These AI models gave the following predictions:\n"
        f"{model_lines}\n\n"
        "Based on what you ACTUALLY SEE in the image, rank these models "
        "from MOST correct to LEAST correct.\n"
        "IMPORTANT: Reply with ONLY the model names separated by commas. "
        "Use these EXACT names: custom_cnn, resnet50, efficientnet_b0, mobilenet_v3\n"
        "Example output: resnet50, efficientnet_b0, mobilenet_v3, custom_cnn\n"
        "No other text. Just the comma-separated names."
    )

    result = _ollama_post({
        "model":  OLLAMA_VLM_MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
    }, timeout=120)

    if result is None:
        return "bağlantı hatası", []

    raw = result.get("response", "").strip()

    # Model isimlerini normalize et
    ranked: list[str] = []
    seen:   set[str]  = set()
    for token in re.split(r"[,\n;]+", raw):
        token = token.strip()
        if not token:
            continue
        normalized = _normalize_model_name(token)
        if normalized and normalized not in seen:
            ranked.append(normalized)
            seen.add(normalized)

    # Sıralamada eksik kalan modelleri sona ekle
    all_models = [r["model"] for r in cnn_results if not r.get("error")]
    for m in all_models:
        if m not in seen:
            ranked.append(m)

    return raw, ranked


# ─── Kompozit Skor Hesaplama ──────────────────────────────────────────────────
def compute_composite_scores(
    cnn_results: list,
    ranked_models: list[str],
    weight_confidence: float = 0.60,
    weight_rank:       float = 0.40,
) -> list[dict]:
    """
    Kompozit skor = CNN Güveni × %60 + Moondream Sıralama Skoru × %40

    Sıralama skoru (4 model için):
        1. sıra → 100 puan
        2. sıra → 66  puan
        3. sıra → 33  puan
        4. sıra →  0  puan

    Moondream sıralama yapamadıysa rank skoru 50 (nötr) olarak atanır.
    """
    n = len(ranked_models) if ranked_models else 1
    rank_score_map: dict[str, float] = {}
    for i, name in enumerate(ranked_models):
        # Lineer: 1. = 100, son = 0
        rank_score_map[name] = 100.0 * (1 - i / max(n - 1, 1)) if n > 1 else 100.0

    scored = []
    for r in cnn_results:
        if r.get("error"):
            scored.append({**r, "rank_score": 0.0, "composite": 0.0, "rank_pos": 99})
            continue

        model    = r["model"]
        conf     = r["confidence"]
        rk_score = rank_score_map.get(model, 50.0)  # sıralama yoksa nötr

        composite = conf * weight_confidence + rk_score * weight_rank

        pos = ranked_models.index(model) + 1 if model in ranked_models else 99

        scored.append({
            **r,
            "rank_score": round(rk_score, 1),
            "composite":  round(composite, 2),
            "rank_pos":   pos,
        })

    return sorted(scored, key=lambda x: x["composite"], reverse=True)


# ─── Terminal Scoreboard ──────────────────────────────────────────────────────
def print_scoreboard(
    cnn_results:     list,
    scored_results:  list,
    cot_description: str,
    cot_prediction:  str,
    cot_reasoning:   str,
    judge_raw:       str,
    ranked_models:   list[str],
    image_path:      Path,
):
    SEP  = "=" * 72
    SEP2 = "─" * 72
    medals = ["🥇", "🥈", "🥉", "4️⃣"]

    print(f"\n{SEP}")
    print(f"  🏆  MODEL ARENA — {image_path.name}")
    print(f"  🖥️  Cihaz : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  🎮  GPU   : {torch.cuda.get_device_name(0)}")
    print(SEP)

    # ── CNN Tablosu (kompozit sıraya göre) ────────────────────────────────────
    print(f"\n  {'#':<3} {'Model':<20} {'Tahmin':<14} {'Güven':>7}  "
          f"{'Rank Sk':>7}  {'Komposit':>9}  {'Süre':>8}")
    print(f"  {SEP2}")

    for i, r in enumerate(scored_results):
        if r.get("error"):
            print(f"  {'❌':<3} {r['model']:<20} .pth bulunamadı")
            continue
        medal = medals[i] if i < len(medals) else "  "
        print(
            f"  {medal}  {r['model']:<20} {r['prediction']:<14}"
            f" %{r['confidence']:5.1f}  "
            f" {r['rank_score']:6.1f}  "
            f" {r['composite']:8.2f}  "
            f" {r['ms']:6.1f} ms"
        )

    print(f"\n  Skor formülü: CNN Güven × %60  +  Moondream Sıralama × %40")

    # ── Moondream CoT Bölümü ──────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print(f"  🌙  MOONDREAM ({OLLAMA_VLM_MODEL.upper()}) — Chain of Thought")
    print(SEP2)

    print(f"\n  📝 Betimleme   : {cot_description}")
    print(f"  🔍 Gerekçe     : {cot_reasoning}")

    pred_ok = cot_prediction in VALID_CLASSES
    icon    = "✅" if pred_ok else "⚠️"
    print(f"  🎯 Tahmini     : {icon}  {cot_prediction.upper()}")

    # Hakem sıralaması
    print(f"\n  ⚖️  Hakem Sıralaması:")
    if ranked_models:
        for i, name in enumerate(ranked_models[:4]):
            m = medals[i] if i < len(medals) else "  "
            print(f"      {m}  {name}")
    else:
        print(f"      ⚠️  Sıralama alınamadı. Ham yanıt: {judge_raw[:80]!r}")

    # ── Mutabakat ──────────────────────────────────────────────────────────────
    all_preds = [r["prediction"] for r in cnn_results if not r.get("error")]
    if pred_ok:
        all_preds.append(cot_prediction)

    print(f"\n{SEP2}")
    if all_preds:
        consensus = max(set(all_preds), key=all_preds.count)
        count     = all_preds.count(consensus)
        total     = len(all_preds)
        print(f"  🤝  Mutabakat : {consensus.upper()}  ({count}/{total} model aynı fikirde)")

    if scored_results:
        winner = scored_results[0]
        if not winner.get("error"):
            print(
                f"  🏅  Kompozit Kazanan : {winner['model']}  "
                f"({winner['prediction'].upper()}, "
                f"skor={winner['composite']:.1f})"
            )
    print(f"\n{SEP}\n")


# ─── Ana Fonksiyon ────────────────────────────────────────────────────────────
def main():
    global _DEBUG_OLLAMA

    parser = argparse.ArgumentParser(description="Model Arena — Terminal v3.0")
    parser.add_argument("--image",        type=str,  default=None,
                        help="Test görseli yolu (varsayılan: config.py'deki yol)")
    parser.add_argument("--no-vlm",       action="store_true",
                        help="Moondream adımlarını atla")
    parser.add_argument("--debug-ollama", action="store_true",
                        help="Ham Ollama yanıtını ekrana bas")
    args = parser.parse_args()

    _DEBUG_OLLAMA = args.debug_ollama

    img_path = Path(args.image) if args.image else Path(DEFAULT_TEST_IMAGE)
    if not img_path.exists():
        print(f"❌  Görsel bulunamadı: {img_path}")
        sys.exit(1)

    print(f"\n📸  Görsel : {img_path}")
    print(f"🖥️  Cihaz  : {DEVICE}")
    if torch.cuda.is_available():
        print(f"🎮  GPU    : {torch.cuda.get_device_name(0)}\n")

    # 1. CNN Modelleri
    print("── CNN Modelleri ──────────────────────────────────────────────")
    cnn_results: list[dict] = []
    for name in MODEL_FILES:
        print(f"  ⚙️  {name} yükleniyor + çalıştırılıyor...", end="\r")
        cnn_results.append(run_cnn(name, img_path))
    print(" " * 60, end="\r")
    print(f"  ✅  {len([r for r in cnn_results if not r.get('error')])} CNN modeli tamamlandı.\n")

    # 2. Moondream CoT + Hakem
    cot_description = "atlandı"
    cot_prediction  = "atlandı"
    cot_reasoning   = "atlandı"
    judge_raw       = "atlandı"
    ranked_models   = []

    if not args.no_vlm:
        print("── Moondream Chain of Thought ─────────────────────────────────")
        cot_description, cot_prediction, cot_reasoning = run_moondream_cot(img_path)
        print(f"  ✅  CoT tamamlandı → {cot_prediction.upper()}\n")

        print("── Moondream Hakemlik ─────────────────────────────────────────")
        judge_raw, ranked_models = run_moondream_judge(
            img_path, cnn_results, cot_description
        )
        print(f"  ✅  Sıralama: {' > '.join(ranked_models)}\n")

    # 3. Kompozit Skorlama
    scored = compute_composite_scores(cnn_results, ranked_models)

    # 4. Çıktı
    print_scoreboard(
        cnn_results, scored,
        cot_description, cot_prediction, cot_reasoning,
        judge_raw, ranked_models,
        img_path,
    )


if __name__ == "__main__":
    main()
