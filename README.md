# Plant Disease Detection (LeafSense)

Plant disease detection app: **Flask backend** (PyTorch model) + **React front end** (LeafSense UI).

## Project layout (mapping)

| Part | Location | Role |
|------|----------|------|
| **Backend (API)** | `app.py` (project root) | Flask app: loads EfficientNet model, exposes `POST /predict`, serves built front end from `leaf-doctor-frontend-main/dist` |
| **Front end (UI)** | `leaf-doctor-frontend-main/` | React + Vite + TypeScript + shadcn/ui. Build output: `leaf-doctor-frontend-main/dist/` |
| **Old UI (deprecated)** | `templates/index.html`, `templates/result.html` | Legacy HTML forms; only used if the React app is not built |

## API contract (front end ↔ backend)

- **Endpoint:** `POST /predict`
- **Request:** `multipart/form-data` with field **`image`** (file). Max size **10 MB** (configurable via `MAX_CONTENT_MB`). Allowed types: JPG, PNG, GIF, WEBP.
- **Response (JSON):**
  - Success: `{ "class": "Healthy" | "Diseased", "confidence": number, "message", "recommendation", "confidence_tier", "nutrient_score" }`
  - Error: `{ "error": "message" }` with HTTP 4xx/5xx.
- **Health check:** `GET /health` returns `{ "status": "ok", "model_loaded": true }` with 200 when the app and model are ready (for load balancers / deployments).
- **Rate limit:** 30 requests per 60 seconds per IP (configurable via env). Exceeding returns **429** and `{ "error": "Too many requests. Please try again later." }`.

Front end source: `leaf-doctor-frontend-main/src/lib/api.ts` (`predictImage()`).

## How to run

### Option A – Single server (Flask serves UI + API)

1. Build the front end:
   ```bash
   cd leaf-doctor-frontend-main
   npm install
   npm run build
   cd ..
   ```
2. Run the backend (serves the built React app at `/` and API at `/predict`):
   ```bash
   python app.py
   ```
3. Open **http://localhost:5000**

### Option B – Dev: React dev server + Flask API

1. Start Flask (API only; old template at `/` if dist not present):
   ```bash
   python app.py
   ```
2. In another terminal, start the React dev server:
   ```bash
   cd leaf-doctor-frontend-main
   npm run dev
   ```
3. Open **http://localhost:8080** (React). The UI calls the API at `http://localhost:5000` (set in `leaf-doctor-frontend-main/src/lib/api.ts` via `VITE_API_BASE_URL` or default).

## Backend requirements

- Python 3 with: `flask`, `torch`, `torchvision`, `PIL` (Pillow), `timm` (see `requirements.txt`)
- Model file: `efficientnet_plantdoc.pth` in the project root (same folder as `app.py`)

## Environment variables (backend)

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_DEBUG` | `false` | Set to `true`/`1` for debug mode (do not use in production). |
| `CORS_ORIGIN` | `*` | Allowed CORS origin; set to your front-end URL in production (e.g. `https://yourdomain.com`). |
| `MAX_CONTENT_MB` | `10` | Max upload size in MB for `/predict`. |
| `MIN_PLANT_CONFIDENCE` | `0.5` | Minimum confidence (0–1) to accept a prediction; below this returns a low-confidence error. |
| `RATE_LIMIT_REQUESTS` | `30` | Max requests per IP per window. |
| `RATE_LIMIT_WINDOW_SEC` | `60` | Rate limit window in seconds. |
| `RATE_LIMIT_MAX_IPS` | `10000` | Max number of IPs to track (older entries evicted). |

## Summary of changes made

- **Backend:** Uses uploaded file (field `image`), saves to a temp file, runs prediction, returns JSON, then deletes the temp file. CORS enabled for dev. Serves React build from `leaf-doctor-frontend-main/dist` at `/` when present; fallback to `templates/index.html` if not built.
- **Front end:** Hero section no longer depends on a missing `hero-bg.jpg` (uses CSS gradient). Production build uses relative `/predict` when served from Flask (`.env.production`).
- **Old templates:** Kept for fallback only; primary UI is the React app in `leaf-doctor-frontend-main`.
