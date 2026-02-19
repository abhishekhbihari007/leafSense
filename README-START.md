# How to Run LeafSense

## Quick Start (Both Servers)

### Option 1: Use the PowerShell Script (Easiest)
```powershell
cd "C:\Users\bihar\Desktop\PLANT FRONT END"
.\start-dev.ps1
```

This will start both backend and frontend automatically.

---

### Option 2: Manual Start (Two Terminals)

#### Terminal 1 - Backend (Flask API)
```powershell
cd "C:\Users\bihar\Desktop\PLANT FRONT END"
py -3.12 app.py
```
**Wait for:** `Running on http://127.0.0.1:5000`

#### Terminal 2 - Frontend (React)
```powershell
cd "C:\Users\bihar\Desktop\PLANT FRONT END"
npm run dev
```
**Wait for:** `Local: http://localhost:5173/`

---

## Then Open in Browser

**Open:** http://localhost:5173

The frontend will call the backend API at `http://localhost:5000/predict` automatically.

---

## Troubleshooting

### "Failed to fetch" Error
- **Make sure Flask backend is running** on port 5000
- Check terminal 1 shows: `Running on http://127.0.0.1:5000`
- If port 5000 is busy, change it in `app.py` (last line: `app.run(debug=True, port=5000)`)

### Frontend Shows Blank Page
- Make sure you're opening **http://localhost:5173** (not 3000 or 5174)
- Check terminal 2 shows Vite is running

### Model Not Loading
- Make sure `efficientnet_plantdoc.pth` exists in the project root
- Check backend terminal for errors
