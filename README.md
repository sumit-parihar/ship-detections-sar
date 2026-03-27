# 🛳️ ShipScan — Maritime Detection Engine

GPU-accelerated ship detection on SAR imagery using YOLO + GeoTIFF support.

**Model:** [`sumit3142857/ship-detection-yolo`](https://huggingface.co/sumit3142857/ship-detection-yolo) on HuggingFace  
**Framework:** Streamlit + Ultralytics YOLO  

---

## 🚀 Deploy on Streamlit Community Cloud (Free)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit — ShipScan"
git remote add origin https://github.com/<your-username>/shipscan.git
git push -u origin main
```

### Step 2 — Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **"New app"**
3. Select your repo, branch (`main`), and set **Main file path** to `app.py`
4. Click **Deploy**

Streamlit Cloud will automatically install `requirements.txt` and `packages.txt`.  
The YOLO model is downloaded from HuggingFace on first run and cached.

---

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

> **GPU note:** For GPU acceleration locally, replace the torch install:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```
> Streamlit Cloud runs on CPU — inference works but is slower on large images.

---

## 📁 Project Structure

```
shipscan/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── packages.txt            # System (apt) dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme + server config
└── .gitignore
```

---

## ⚙️ Configuration

Edit `.streamlit/config.toml` to change upload limits or theme:
```toml
[server]
maxUploadSize = 2000   # Max upload in MB
```
