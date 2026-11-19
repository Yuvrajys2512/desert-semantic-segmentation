# Desert Semantic Segmentation (hackathon demo)

The demo shell for a desert-terrain semantic segmentation project: upload an off-road desert image, and the app returns a colour-coded class mask over it (trees, bushes, dry grass, rocks, logs, sky and so on).

> **Note:** `app.py` uses **mock inference** (it splits the image into fixed stripes as stand-in classes) so the UI can be demoed without PyTorch or model weights. A trained model can be swapped in at `predict_pil_mock()`; the rest of the pipeline (palette, overlay, stats) stays the same.

## Structure

- `hack-demo/backend/app.py`: FastAPI. `POST` an image and get back the mask, a coloured overlay (adjustable alpha) and per-class pixel shares as base64 PNG + JSON
- `hack-demo/frontend/index.html`: single-page upload and preview UI

## Run it

```bash
cd hack-demo/backend
pip install fastapi uvicorn pillow numpy python-multipart
uvicorn app:app --reload
```

Then open `hack-demo/frontend/index.html` in a browser.
