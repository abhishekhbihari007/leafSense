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
- **Request:** `multipart/form-data` with field **`image`** (file).
- **Response (JSON):**
  - Success: `{ "class": "Healthy" | "Diseased", "confidence": number, "nutrient_score": number, "random_confidence_score": number }`
  - Error: `{ "error": "message" }` with HTTP 4xx/5xx.

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

- Python 3 with: `flask`, `torch`, `torchvision`, `PIL` (Pillow), `timm`
- Model file: `efficientnet_plantdoc.pth` in the project root (same folder as `app.py`)

## Summary of changes made

- **Backend:** Uses uploaded file (field `image`), saves to a temp file, runs prediction, returns JSON, then deletes the temp file. CORS enabled for dev. Serves React build from `leaf-doctor-frontend-main/dist` at `/` when present; fallback to `templates/index.html` if not built.
- **Front end:** Hero section no longer depends on a missing `hero-bg.jpg` (uses CSS gradient). Production build uses relative `/predict` when served from Flask (`.env.production`).
- **Old templates:** Kept for fallback only; primary UI is the React app in `leaf-doctor-frontend-main`.
