import json
import re
import urllib.request
import urllib.error

class Tools:
    def __init__(self):
        self.API_BASE_URL:    str = "http://192.168.1.105:8000"
        self.OLLAMA_BASE_URL: str = "http://192.168.1.105:11434"
        self.OLLAMA_MODEL:    str = "llava"
        self.VALID_CLASSES:   list = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

    def _normalize(self, raw: str) -> str:
        key = re.sub(r"[^a-z0-9]", "", raw.lower())
        name_map = {"customcnn": "custom_cnn", "alper": "custom_cnn", "resnet50": "resnet50", "efficientnetb0": "efficientnet_b0", "mobilenetv3": "mobilenet_v3"}
        return name_map.get(key, key)

    def _emoji(self, pred: str) -> str:
        emojis = {"cardboard": "📦", "glass": "🫙", "metal": "🔩", "paper": "📄", "plastic": "🧴", "trash": "🗑️"}
        return emojis.get(pred, "❓")

    def _request(self, url: str, method: str = "GET", payload: dict = None, timeout: int = 60):
        try:
            data = json.dumps(payload).encode("utf-8") if payload else None
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            try: return json.loads(raw)
            except: return json.loads([l for l in raw.splitlines() if l.strip()][-1])
        except Exception as e:
            return {"error": str(e)}

    def arena_battle(self, image_path: str = "") -> str:
        # 1. API'den CNN sonuçlarını ve HAZIR BASE64 verisini çek
        cnn_data = self._request(f"{self.API_BASE_URL}/predict/all")
        if "error" in cnn_data:
            return f"❌ API Hatası: {cnn_data['error']}"

        # 2. Docker/Windows Dosya Yolu Sorunu ÇÖZÜLDÜ!
        # Artık dosyayı okumaya çalışmıyoruz, API'nin gönderdiği ham veriyi alıyoruz.
        img_b64 = cnn_data.get("base64_image", "")
        if not img_b64:
            return "⚠️ API'den Base64 görsel verisi alınamadı. api_service.py'yi güncellediğine emin ol."

        # 3. LLaVA (VLM) Analizi
        cot_resp = self._request(f"{self.OLLAMA_BASE_URL}/api/generate", "POST", {
            "model": self.OLLAMA_MODEL,
            "prompt": "Describe this waste item in one short sentence. Is it cardboard, glass, metal, paper, plastic, or trash?",
            "images": [img_b64], "stream": False
        })
        vlm_raw = cot_resp.get("response", "Analiz yapılamadı.")
        vlm_pred = next((w for w in self.VALID_CLASSES if w in vlm_raw.lower()), "bilinmiyor")

        # 4. Hakem Sıralaması
        judge_resp = self._request(f"{self.OLLAMA_BASE_URL}/api/generate", "POST", {
            "model": self.OLLAMA_MODEL,
            "prompt": f"Rank these models from best to worst based on the image: {list(cnn_data['results'].keys())}. Reply with names only.",
            "images": [img_b64], "stream": False
        })
        ranked = [self._normalize(t) for t in re.split(r"[,\s]+", judge_resp.get("response", "")) if self._normalize(t)]

        # 5. Skorlama
        results = cnn_data['results']
        n = len(results)
        rank_map = {name: 100 * (1 - i/(n-1)) if n > 1 else 100 for i, name in enumerate(ranked)}
        
        scored = []
        for k, v in results.items():
            rk = rank_map.get(k, 50.0)
            scored.append({"name": k, "pred": v['prediction'], "conf": v['confidence'], "ms": v['inference_ms'], "rk": rk, "comp": round(v['confidence'] * 0.60 + rk * 0.40, 2)})
        
        scored = sorted(scored, key=lambda x: x['comp'], reverse=True)

        # 6. Tablo Çıktısı
        lines = ["## 🏆 Model Arena — Kesin Sonuçlar", f"**Görsel:** `{cnn_data.get('image')}`", "", 
                 "| # | Model | Tahmin | Güven | Rank Sk | Kompozit | Süre |", "|---|-------|--------|-------|---------|----------|------|"]
        for i, r in enumerate(scored):
            lines.append(f"| {i+1} | **{r['name']}** | {self._emoji(r['pred'])} {r['pred']} | %{r['conf']:.1f} | {r['rk']:.0f} | **{r['comp']:.1f}** | {r['ms']:.1f} ms |")

        lines += ["", "---", f"### 🌙 LLaVA Hakem", f"**Gerekçe:** {vlm_raw}", f"**Tahmini:** {self._emoji(vlm_pred)} **{vlm_pred.upper()}**"]
        return "\n".join(lines)