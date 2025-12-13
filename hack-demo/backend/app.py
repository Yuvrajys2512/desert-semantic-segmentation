# app.py (mock inference; no PyTorch needed)
import io, time, base64
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- palette for nice colors (adjust later if you want) ---
PALETTE: Dict[int, Tuple[int,int,int]] = {
    100: (34,139,34), 200:(60,179,113), 300:(189,183,107),
    500:(210,180,140), 550:(128,128,128), 600:(255,182,193),
    700:(139,69,19), 800:(105,105,105), 7100:(222,184,135), 10000:(135,206,235)
}
DEFAULT_COLOR = (0,0,0)

def colorize_mask(mask: np.ndarray) -> Image.Image:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, col in PALETTE.items():
        rgb[mask == cid] = col
    missing = (rgb == 0).all(axis=2)
    rgb[missing] = DEFAULT_COLOR
    return Image.fromarray(rgb, "RGB")

def blend_overlay(rgb: Image.Image, color: Image.Image, alpha: float) -> Image.Image:
    base = rgb.convert("RGBA")
    layr = color.convert("RGBA")
    layr.putalpha(int(alpha * 255))
    return Image.alpha_composite(base, layr)

def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# Mock model: 3 vertical stripes = 3 fake classes
def predict_pil_mock(pil: Image.Image) -> np.ndarray:
    w, h = pil.size
    mask = np.zeros((h, w), dtype=np.int32)
    mask[:, : w//3] = 100
    mask[:, w//3: 2*w//3] = 300
    mask[:, 2*w//3:] = 800
    return mask

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/infer")
async def infer(file: UploadFile = File(...), overlay_alpha: float = 0.5):
    try:
        raw = await file.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")

        t0 = time.time()
        mask = predict_pil_mock(pil)
        latency_ms = int((time.time() - t0) * 1000)

        color = colorize_mask(mask)
        overlay = blend_overlay(pil, color, alpha=float(overlay_alpha))
        b64 = pil_to_base64_png(overlay)

        return {"overlay_png_b64": b64, "latency_ms": latency_ms}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
