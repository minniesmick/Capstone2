"""
config.py — Merkezi yapılandırma modülü.
Windows, macOS ve Linux yollarını otomatik olarak algılar (pathlib).
"""

import os
import platform
from pathlib import Path

# ─── Platform algılama ────────────────────────────────────────────────────────
_OS = platform.system()  # "Windows" | "Darwin" | "Linux"


def _resolve_project_root() -> Path:
    """
    Projenin kök dizinini işletim sistemine göre döndürür.
    Ortam değişkeni CAPSTONE2_ROOT tanımlıysa onu kullanır (CI/CD için).
    """
    env_root = os.environ.get("CAPSTONE2_ROOT")
    if env_root:
        return Path(env_root)

    if _OS == "Windows":
        return Path.home() / "PROJELER" / "Capstone2"
    elif _OS == "Darwin":          # macOS
        return Path.home() / "Projects" / "Capstone2"
    else:                          # Linux / Docker
        return Path.home() / "Capstone2"


PROJECT_ROOT: Path = _resolve_project_root()

# ─── Veri ve model yolları ────────────────────────────────────────────────────
DATA_DIR:     Path = PROJECT_ROOT / "Garbage classification"
RESULTS_DIR:  Path = PROJECT_ROOT / "results"
MODELS_DIR:   Path = RESULTS_DIR  / "models"
ANALYSIS_DIR: Path = RESULTS_DIR  / "analysis"
VIS_DIR:      Path = RESULTS_DIR  / "visualizations"

# ─── Varsayılan test görseli ──────────────────────────────────────────────────
if _OS in ("Windows", "Darwin"):
    DEFAULT_TEST_IMAGE: Path = Path.home() / "Desktop" / "test_image.png"
else:
    DEFAULT_TEST_IMAGE: Path = Path.home() / "test_image.png"

# ─── Model dosya adları ───────────────────────────────────────────────────────
MODEL_FILES: dict[str, str] = {
    "custom_cnn":       "custom_cnn_best.pth",
    "resnet50":         "resnet50_best.pth",
    "efficientnet_b0":  "efficientnet_b0_best.pth",
    "mobilenet_v3":     "mobilenet_v3_best.pth",
}

# arena_battle.py bu listeyi doğrudan import eder
SUPPORTED_MODELS: list[str] = list(MODEL_FILES.keys())


def get_model_path(model_name: str) -> Path:
    filename = MODEL_FILES.get(model_name)
    if filename is None:
        raise ValueError(
            f"Bilinmeyen model: '{model_name}'. "
            f"Geçerli seçenekler: {list(MODEL_FILES)}"
        )
    return MODELS_DIR / filename


# ─── Sınıf isimleri ───────────────────────────────────────────────────────────
CLASS_NAMES: list[str] = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# ─── Eğitim hiper-parametreleri (sadece referans; eğitim kodu burada yoktur) ──
IMAGE_SIZE:    int   = 224
BATCH_SIZE:    int   = 32
EPOCHS:        int   = 30
LEARNING_RATE: float = 0.001

# ─── API servisi ──────────────────────────────────────────────────────────────
API_HOST: str = "0.0.0.0"
API_PORT: int = 8000

# ─── Ollama VLM ───────────────────────────────────────────────────────────────
OLLAMA_VLM_MODEL: str = "llava"   # alternatifler: llava, llava-llama3, bakllava

# ─── Özet (doğrudan çalıştırıldığında) ───────────────────────────────────────
if __name__ == "__main__":
    print(f"OS           : {_OS}")
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"DATA_DIR     : {DATA_DIR}")
    print(f"MODELS_DIR   : {MODELS_DIR}")
    print(f"Test image   : {DEFAULT_TEST_IMAGE}")
    for name in MODEL_FILES:
        p = get_model_path(name)
        status = "✅" if p.exists() else "❌"
        print(f"  {status}  {name:20s} → {p}")