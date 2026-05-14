"""
models.py — Model mimarileri ve yükleme yardımcıları.
CustomCNN'de bias=True zorunludur; aksi hâlde .pth yüklemesi 'unexpected key' hatası verir.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from pathlib import Path

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


# ─── CustomCNN ────────────────────────────────────────────────────────────────

class CustomCNN(nn.Module):
    """
    4 katmanlı konvolüsyonel ağ.
    Conv2d'de bias=True — state_dict uyumluluğu için zorunlu.
    """
    def __init__(self, num_classes: int = 6):
        super().__init__()

        def _block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),  # bias=True ZORUNLU
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.conv1 = _block(3,   32)
        self.conv2 = _block(32,  64)
        self.conv3 = _block(64,  128)
        self.conv4 = _block(128, 256)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.fc(x)


# ─── Model kayıt defteri ──────────────────────────────────────────────────────

def _build_resnet50(nc: int) -> nn.Module:
    return tv_models.resnet50(num_classes=nc)

def _build_efficientnet_b0(nc: int) -> nn.Module:
    if not _TIMM_AVAILABLE:
        raise ImportError("timm paketi yüklü değil: pip install timm")
    return timm.create_model("efficientnet_b0", pretrained=False, num_classes=nc)

def _build_mobilenet_v3(nc: int) -> nn.Module:
    return tv_models.mobilenet_v3_small(num_classes=nc)

_REGISTRY: dict = {
    "custom_cnn":       lambda nc: CustomCNN(nc),
    "resnet50":         _build_resnet50,
    "efficientnet_b0":  _build_efficientnet_b0,
    "mobilenet_v3":     _build_mobilenet_v3,
}

SUPPORTED_MODELS: list[str] = list(_REGISTRY.keys())


# ─── Yardımcı fonksiyonlar ────────────────────────────────────────────────────

def build_model(name: str, num_classes: int = 6) -> nn.Module:
    """Boş model oluştur (ağırlık yüklemez)."""
    if name not in _REGISTRY:
        raise ValueError(f"Bilinmeyen model '{name}'. Geçerliler: {SUPPORTED_MODELS}")
    return _REGISTRY[name](num_classes)


def load_trained_model(name: str, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """
    Kayıtlı .pth ağırlıklarını yükler, modeli eval moduna alır.
    weights_only=True: torch 2.x güvenli yükleme için zorunlu.
    """
    model = build_model(name)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model