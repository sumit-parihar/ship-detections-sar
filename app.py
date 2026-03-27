"""
ShipScan — Maritime Detection Engine v3
GPU-accelerated tiled YOLO · native GeoTIFF GPS · Tile Explorer tab
"""
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np
import io, time, cv2, os, tempfile, math
from concurrent.futures import ThreadPoolExecutor, as_completed

Image.MAX_IMAGE_PIXELS = None

st.set_page_config(
    page_title="ShipScan · Maritime AI",
    layout="wide",
    page_icon="🛳️",
    initial_sidebar_state="expanded",
)

# ─── PREMIUM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
  --bg:#07111f; --surf:#0c1a2e; --surf2:#101f33; --surf3:#0e1929;
  --b:#1a2e48; --b2:#203a5a;
  --acc:#00d4ff; --acc2:#00ff99; --acc3:#7c72ff;
  --warn:#ffb400; --danger:#ff3d3d; --txt:#c5d5e8; --dim:#3d5a7a;
  --fn:'Inter',sans-serif; --mono:'JetBrains Mono',monospace;
  --r:10px;
}

html,body,[class*="css"]{ font-family:var(--fn)!important; background:var(--bg)!important; color:var(--txt)!important; }
.stApp{ background:var(--bg)!important; }

/* ══ Scrollbar ══ */
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:var(--surf)}
::-webkit-scrollbar-thumb{background:var(--b2);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--dim)}

/* ══ Sidebar — FIXED, always visible, no toggle ══ */
[data-testid="stSidebar"]{
  background:var(--surf)!important;
  border-right:1px solid var(--b)!important;
  width:300px!important;
  position:fixed!important;
  top:0!important;
  left:0!important;
  height:100vh!important;
  z-index:999!important;
  transform:translateX(0px)!important;
  transition:none!important;
  min-width:300px!important;
  max-width:300px!important;
}
[data-testid="stSidebarContent"]{
  padding:18px 16px!important;
  overflow-y:auto!important;
  height:100vh!important;
}
/* Hide the sidebar collapse/toggle button */
[data-testid="stSidebarCollapseButton"],
button[kind="header"],
[data-testid="collapsedControl"],
.st-emotion-cache-1rtdyuf,
.st-emotion-cache-pkbazv,
[data-testid="stSidebar"] > div:first-child > div > button {
  display:none!important;
  visibility:hidden!important;
  pointer-events:none!important;
}
/* Push main content right to account for fixed sidebar */
section[data-testid="stMain"],
.main,
[data-testid="stAppViewContainer"] > section:not([data-testid="stSidebar"]) {
  margin-left:300px!important;
  width:calc(100% - 300px)!important;
  min-width:0!important;
  overflow-x:hidden!important;
}
.main .block-container,
[data-testid="stMainBlockContainer"] {
  max-width:100%!important;
  padding-left:24px!important;
  padding-right:24px!important;
  box-sizing:border-box!important;
  width:100%!important;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not([data-testid]),
[data-testid="stSidebar"] small{ color:var(--txt)!important; }

/* ── Slider ── */
[data-testid="stSlider"]{ padding:4px 8px 18px 8px!important; }
[data-testid="stSlider"] > label{
  font-family:var(--mono)!important; font-size:0.68rem!important;
  letter-spacing:.05em!important; color:var(--acc)!important;
  text-transform:uppercase!important; margin-bottom:6px!important; display:block!important;
}
[data-baseweb="slider"] [class*="StyledSliderBar"]{ background:var(--b)!important; height:4px!important; border-radius:2px!important; }
[data-baseweb="slider"] [class*="StyledSliderInnerBar"]{ background:var(--acc)!important; }
[data-baseweb="slider"] [class*="StyledSliderThumb"],
[data-baseweb="slider"] div[role="slider"]{
  background:var(--acc)!important; border:2px solid #fff!important;
  width:16px!important; height:16px!important; border-radius:50%!important;
  box-shadow:0 0 10px var(--acc)!important; outline:none!important;
}
[data-baseweb="slider"] [class*="StyledSliderHandleTooltip"],
[data-baseweb="slider"] [data-value]{
  background:var(--surf2)!important; border:1px solid var(--b)!important;
  color:var(--acc)!important; font-family:var(--mono)!important;
  font-size:.65rem!important; border-radius:4px!important; padding:2px 6px!important;
}
[data-testid="stSidebar"] hr{ border-color:var(--b)!important; margin:12px 0!important; }

/* ══ Tabs ══ */
[data-testid="stTabs"] [role="tablist"]{
  background:var(--surf)!important;
  border-bottom:1px solid var(--b)!important;
  border-radius:var(--r) var(--r) 0 0!important;
  padding:0 6px!important;
  gap:2px!important;
}
[data-testid="stTabs"] button[role="tab"]{
  font-family:var(--mono)!important;
  font-size:.72rem!important;
  letter-spacing:.06em!important;
  text-transform:uppercase!important;
  color:var(--dim)!important;
  padding:10px 18px!important;
  border-radius:var(--r) var(--r) 0 0!important;
  border:none!important;
  background:transparent!important;
  transition:all .2s ease!important;
}
[data-testid="stTabs"] button[role="tab"]:hover{ color:var(--txt)!important; background:var(--b)!important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{
  color:var(--acc)!important;
  background:var(--surf2)!important;
  border-bottom:2px solid var(--acc)!important;
}
[data-testid="stTabs"] [data-testid="stTabContent"]{
  background:var(--surf2)!important;
  border:1px solid var(--b)!important;
  border-top:none!important;
  border-radius:0 0 var(--r) var(--r)!important;
  padding:20px!important;
}

/* ══ Header ══ */
.hdr{display:flex;align-items:center;gap:16px;padding:14px 0 12px;border-bottom:1px solid var(--b);margin-bottom:20px}
.hdr-ico{width:48px;height:48px;border:1px solid var(--acc);border-radius:12px;
         display:flex;align-items:center;justify-content:center;font-size:24px;
         background:linear-gradient(135deg,#00d4ff18,transparent);
         box-shadow:0 0 24px #00d4ff25;animation:pulse 3s ease-in-out infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 24px #00d4ff25}50%{box-shadow:0 0 36px #00d4ff50}}
.hdr h1{font-size:1.8rem!important;font-weight:800!important;color:#fff!important;margin:0!important;letter-spacing:-.03em;
         background:linear-gradient(90deg,#fff,#00d4ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hdr p{font-family:var(--mono)!important;font-size:.6rem!important;color:var(--acc)!important;
       letter-spacing:.18em!important;margin:3px 0 0!important;text-transform:uppercase}

/* ══ Metric grid ══ */
.mgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}
.mc{background:var(--surf);border:1px solid var(--b);border-radius:var(--r);
    padding:16px 18px;position:relative;overflow:hidden;
    transition:border-color .2s,box-shadow .2s}
.mc:hover{border-color:var(--b2);box-shadow:0 4px 20px #00000040}
.mc::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
            background:linear-gradient(90deg,var(--acc),transparent)}
.mc.g::before{background:linear-gradient(90deg,var(--acc2),transparent)}
.mc.w::before{background:linear-gradient(90deg,var(--warn),transparent)}
.mc.r::before{background:linear-gradient(90deg,var(--danger),transparent)}
.mc.p::before{background:linear-gradient(90deg,var(--acc3),transparent)}
.mc .lbl{font-family:var(--mono)!important;font-size:.58rem;color:var(--dim);
          text-transform:uppercase;letter-spacing:.12em;margin-bottom:6px}
.mc .val{font-size:1.8rem;font-weight:700;color:#fff;line-height:1.1}
.mc .sub{font-family:var(--mono)!important;font-size:.58rem;color:var(--dim);margin-top:4px}

/* ══ Image panels ══ */
.ipanel{background:var(--surf);border:1px solid var(--b);border-radius:var(--r);overflow:hidden}
.ipanel-hdr{padding:8px 14px;font-family:var(--mono)!important;font-size:.6rem;
            color:var(--acc);text-transform:uppercase;letter-spacing:.1em;
            background:var(--surf2);border-bottom:1px solid var(--b)}

/* ══ Tile card ══ */
.tile-card{
  background:var(--surf);border:1px solid var(--b);border-radius:var(--r);
  overflow:hidden;transition:border-color .2s,box-shadow .2s;margin-bottom:14px;
}
.tile-card:hover{border-color:var(--acc);box-shadow:0 0 20px #00d4ff18}
.tile-hdr{
  padding:8px 12px;display:flex;align-items:center;justify-content:space-between;
  background:var(--surf2);border-bottom:1px solid var(--b);
  font-family:var(--mono)!important;font-size:.6rem;
}
.tile-num{color:var(--acc);font-weight:700;font-size:.75rem}
.tile-ships{color:var(--acc2)}
.tile-coord{color:var(--dim);font-size:.57rem}
.conf-bar-wrap{width:100%;background:var(--b);border-radius:3px;height:4px;margin:6px 0 2px}
.conf-bar{height:4px;border-radius:3px;background:linear-gradient(90deg,#00ff99,#00d4ff)}

/* ══ Badge ══ */
.badge{
  display:inline-flex;align-items:center;gap:5px;
  border-radius:5px;padding:3px 9px;font-family:var(--mono)!important;
  font-size:.62rem;font-weight:600;letter-spacing:.04em;
}
.badge-green{background:rgba(0,255,153,.12);border:1px solid rgba(0,255,153,.3);color:var(--acc2)}
.badge-blue{background:rgba(0,212,255,.12);border:1px solid rgba(0,212,255,.3);color:var(--acc)}
.badge-yellow{background:rgba(255,180,0,.12);border:1px solid rgba(255,180,0,.3);color:var(--warn)}
.badge-red{background:rgba(255,61,61,.12);border:1px solid rgba(255,61,61,.3);color:var(--danger)}
.badge-purple{background:rgba(124,114,255,.12);border:1px solid rgba(124,114,255,.3);color:var(--acc3)}

/* ══ Geo badge ══ */
.geo-badge{display:inline-flex;align-items:center;gap:8px;
           background:rgba(0,255,153,.08);border:1px solid rgba(0,255,153,.3);
           border-radius:8px;padding:8px 16px;font-family:var(--mono)!important;
           font-size:.7rem;color:var(--acc2);margin:8px 0}

/* ══ Ship detail card ══ */
.ship-card{background:var(--surf3);border:1px solid var(--b);border-radius:8px;
           padding:10px 14px;margin:6px 0;position:relative}
.ship-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;
                   background:var(--acc2);border-radius:3px 0 0 3px}
.ship-card h4{margin:0 0 5px;font-size:.82rem;font-weight:600;color:#fff}
.ship-card .coords{font-family:var(--mono)!important;font-size:.68rem;color:var(--acc)}
.ship-card .meta{font-family:var(--mono)!important;font-size:.6rem;color:var(--dim);margin-top:4px}

/* ══ Table ══ */
.det-table{width:100%;border-collapse:collapse;font-family:var(--mono)!important;font-size:.72rem}
.det-table th{background:var(--surf2);color:var(--acc);text-align:left;
              padding:8px 12px;border-bottom:1px solid var(--b);
              font-size:.6rem;letter-spacing:.1em;text-transform:uppercase}
.det-table td{padding:7px 12px;border-bottom:1px solid var(--b);vertical-align:middle}
.det-table tr:hover td{background:rgba(0,212,255,.04)}

/* ══ Section label ══ */
.slbl{font-family:var(--mono)!important;font-size:.6rem;color:var(--dim);
      text-transform:uppercase;letter-spacing:.15em;margin:20px 0 10px;
      display:flex;align-items:center;gap:10px}
.slbl::after{content:'';flex:1;height:1px;background:var(--b)}

/* ══ Upload hint ══ */
.upload-hint{display:flex;flex-direction:column;align-items:center;justify-content:center;
             padding:70px 20px;text-align:center;gap:14px}

/* ══ Custom Spinner ══ */
.shipscan-spinner-overlay {
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:16px;padding:32px;
}
.shipscan-spinner {
  position:relative;width:56px;height:56px;
}
.shipscan-spinner .ring {
  position:absolute;inset:0;border-radius:50%;border:2px solid transparent;
  animation:spin-ring 1.4s cubic-bezier(0.65,0.05,0.36,0.95) infinite;
}
.shipscan-spinner .ring-1 {
  border-top-color:var(--acc);border-right-color:var(--acc);
  animation-duration:1.4s;
}
.shipscan-spinner .ring-2 {
  inset:6px;border-bottom-color:var(--acc2);border-left-color:var(--acc2);
  animation-duration:1.0s;animation-direction:reverse;
}
.shipscan-spinner .ring-3 {
  inset:14px;border-top-color:var(--acc3);
  animation-duration:0.7s;
}
.shipscan-spinner .dot {
  position:absolute;inset:24px;background:var(--acc);border-radius:50%;
  box-shadow:0 0 8px var(--acc);animation:dot-pulse 1.4s ease-in-out infinite;
}
@keyframes spin-ring {
  0%  { transform:rotate(0deg); }
  100%{ transform:rotate(360deg); }
}
@keyframes dot-pulse {
  0%,100%{ opacity:1;transform:scale(1); }
  50%     { opacity:0.4;transform:scale(0.6); }
}
.spinner-label {
  font-family:var(--mono)!important;font-size:.68rem;color:var(--acc);
  letter-spacing:.14em;text-transform:uppercase;
  animation:label-fade 1.4s ease-in-out infinite;
}
@keyframes label-fade {
  0%,100%{ opacity:1; }
  50%     { opacity:0.5; }
}

/* Override Streamlit default spinner */
[data-testid="stSpinner"] > div {
  display:none!important;
}

/* ══ Misc ══ */
.stProgress>div>div>div>div{background:var(--acc)!important}
.stAlert{background:var(--surf)!important;border:1px solid var(--b)!important;border-radius:8px!important}
[data-testid="stExpanderDetails"]{background:var(--surf2)!important;border:1px solid var(--b)!important}
[data-testid="stExpander"] summary{font-family:var(--mono)!important;font-size:.75rem!important;color:var(--txt)!important}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:6px!important;}
[data-testid="stImage"] p{display:none}
[data-testid="stToggle"] label{font-family:var(--mono)!important;font-size:.7rem!important;color:var(--txt)!important;}
[data-testid="stFileUploader"]{background:var(--surf)!important;border:1px dashed var(--b2)!important;border-radius:var(--r)!important;padding:16px!important}
.stDownloadButton>button{
  font-family:var(--mono)!important;font-size:.7rem!important;
  background:var(--surf)!important;border:1px solid var(--b2)!important;
  color:var(--txt)!important;border-radius:6px!important;padding:6px 14px!important;
  transition:all .2s!important;
}
.stDownloadButton>button:hover{border-color:var(--acc)!important;color:var(--acc)!important;background:var(--surf2)!important}
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
HF_REPO = "sumit3142857/ship-detection-yolo"

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Custom spinner helper ─────────────────────────────────────────────────────
def custom_spinner(label="Processing…"):
    """Returns an st.empty() placeholder pre-filled with the ShipScan spinner."""
    placeholder = st.empty()
    placeholder.markdown(f"""
    <div class="shipscan-spinner-overlay">
      <div class="shipscan-spinner">
        <div class="ring ring-1"></div>
        <div class="ring ring-2"></div>
        <div class="ring ring-3"></div>
        <div class="dot"></div>
      </div>
      <div class="spinner-label">{label}</div>
    </div>""", unsafe_allow_html=True)
    return placeholder

# ─── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    path = hf_hub_download(repo_id=HF_REPO, filename="best.pt", repo_type="model")
    mdl  = YOLO(path)
    mdl.to(DEVICE)
    return mdl


# ─── Image helpers ─────────────────────────────────────────────────────────────
def robust_normalize(arr):
    if arr.dtype == np.uint8:
        return arr
    f = arr.astype(np.float32)
    s = f[::10, ::10]
    p2, p98 = np.percentile(s, (2, 98))
    if p98 > p2:
        f = (f - p2) / (p98 - p2) * 255.0
    elif np.max(f) > 0:
        f = f / np.max(f) * 255.0
    return np.clip(f, 0, 255).astype(np.uint8)


def extract_geo(tmp_path):
    """Extract CRS + affine from GeoTIFF. Returns (dict, debug_str)."""
    debug = []
    try:
        import rasterio
        debug.append(f"rasterio {rasterio.__version__} OK")
        with rasterio.open(tmp_path) as src:
            debug.append(f"Size: {src.width}×{src.height} px | bands: {src.count} | driver: {src.driver}")
            debug.append(f"CRS: {src.crs}")
            debug.append(f"Transform: {src.transform}")
            if src.crs:
                t = src.transform
                epsg = None
                try: epsg = src.crs.to_epsg()
                except: pass
                geo = {
                    "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
                    "crs_wkt": src.crs.to_wkt(),
                    "epsg": epsg,
                    "w": src.width, "h": src.height,
                }
                debug.append(f"EPSG: {epsg}")
                debug.append("✅ GeoInfo extracted successfully")
                return geo, "\n".join(debug)
            else:
                debug.append("❌ No CRS in this TIFF")
    except ImportError:
        debug.append("❌ rasterio not installed")
    except Exception as e:
        debug.append(f"❌ {type(e).__name__}: {e}")
    return None, "\n".join(debug)


def pixels_to_latlon(px, py, geo):
    """Pixel centre → (lat, lon) via affine + pyproj."""
    try:
        import pyproj
        a, b, c, d, e, f = geo["transform"]
        mx = a * px + b * py + c
        my = d * px + e * py + f
        tr = pyproj.Transformer.from_crs(geo["crs_wkt"], "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(mx, my)
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return float(lat), float(lon)
    except Exception:
        pass
    return None, None


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def geo_pixel_size_m(geo):
    """Return approximate ground sampling distance (GSD) in metres per pixel."""
    try:
        import pyproj
        a_coef = geo["transform"][0]  # pixel width in CRS units
        e_coef = geo["transform"][4]  # pixel height (negative)
        crs_wkt = geo["crs_wkt"]
        epsg    = geo.get("epsg")
        # For geographic CRS (degrees), convert ~1 deg ≈ 111 km
        if epsg == 4326 or (epsg is None and "GEOGCS" in crs_wkt):
            gsd_x = abs(a_coef) * 111_000
            gsd_y = abs(e_coef) * 111_000
        else:
            gsd_x, gsd_y = abs(a_coef), abs(e_coef)
        return (gsd_x + gsd_y) / 2
    except Exception:
        return None


def render_map_folium(pts_df, geo=None):
    """Render an interactive Folium map with ships, heatmap, and cluster layers."""
    try:
        import folium
        from folium.plugins import HeatMap, MarkerCluster, MeasureControl, Fullscreen
        from streamlit_folium import st_folium
    except ImportError:
        st.error("Install folium and streamlit-folium: `pip install folium streamlit-folium`")
        return

    avg_lat = pts_df["lat"].mean()
    avg_lon = pts_df["lon"].mean()

    # Auto-zoom based on spread
    lat_span = pts_df["lat"].max() - pts_df["lat"].min()
    lon_span = pts_df["lon"].max() - pts_df["lon"].min()
    span = max(lat_span, lon_span, 0.0001)
    zoom = max(8, min(18, int(15 - math.log2(span * 1000))))

    m = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=zoom,
        tiles=None,   # we'll add layers manually
    )

    # ── Tile layer options (all free) ──
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="🛰 Satellite (Esri)",
        overlay=False, control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="🗺 Street Map",
        overlay=False, control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB",
        name="🌑 Dark",
        overlay=False, control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="🌊 Ocean",
        overlay=False, control=True,
    ).add_to(m)

    # ── Heatmap layer ──
    heat_data = [[r["lat"], r["lon"], r["conf"]] for _, r in pts_df.iterrows()]
    HeatMap(
        heat_data, name="🔥 Heatmap",
        min_opacity=0.4, max_zoom=18, radius=30, blur=20,
        gradient={"0.3": "blue", "0.6": "lime", "1.0": "red"},
    ).add_to(m)

    # ── Cluster layer ──
    cluster = MarkerCluster(name="📍 Ship Markers").add_to(m)
    for _, r in pts_df.iterrows():
        conf  = r["conf"]
        color = "green" if conf >= 0.75 else ("orange" if conf >= 0.50 else "red")
        icon  = folium.Icon(color=color, icon="ship", prefix="fa")
        popup_html = f"""
            <div style="font-family:monospace;font-size:12px;min-width:160px">
              <b>🚢 Ship Detection</b><br>
              <b style="color:#00aa44">Confidence: {conf:.1%}</b><br>
              Lat: <code>{r['lat']:.6f}</code><br>
              Lon: <code>{r['lon']:.6f}</code>
            </div>"""
        folium.Marker(
            location=[r["lat"], r["lon"]],
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"🚢 {conf:.0%} confidence",
            icon=icon,
        ).add_to(cluster)

    # ── Image footprint bounding box (if geo available) ──
    if geo:
        try:
            import pyproj
            W, H = geo["w"], geo["h"]
            corners_px = [(0,0),(W,0),(W,H),(0,H)]
            corners_ll = []
            for px, py in corners_px:
                lat2, lon2 = pixels_to_latlon(px, py, geo)
                if lat2: corners_ll.append([lat2, lon2])
            if len(corners_ll) == 4:
                folium.Polygon(
                    locations=corners_ll,
                    color="#00d4ff", weight=1.5,
                    fill=True, fill_color="#00d4ff", fill_opacity=0.05,
                    tooltip="📷 Image Footprint",
                    name="📷 Image Footprint",
                ).add_to(m)
        except Exception:
            pass

    # ── Controls ──
    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    MeasureControl(position="bottomleft", primary_length_unit="kilometers").add_to(m)
    Fullscreen(position="topleft").add_to(m)

    st_folium(m, use_container_width=True, height=520, returned_objects=[])


def make_geojson_url(ship_pts, label="Ship"):
    import json, urllib.parse
    features = []
    for i, p in enumerate(ship_pts):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
            "properties": {
                "name": f"{label} #{i+1}",
                "confidence": f"{p['conf']:.1%}",
                "marker-color": "#00ff99" if p["conf"] >= 0.75 else ("#ffb400" if p["conf"] >= 0.5 else "#ff3d3d"),
                "marker-size": "medium",
                "marker-symbol": "harbor",
            }
        })
    geojson_str = json.dumps({"type": "FeatureCollection", "features": features})
    encoded = urllib.parse.quote(geojson_str)
    return f"https://geojson.io/#data=data:application/json,{encoded}"


def make_google_maps_url(ship_pts):
    if not ship_pts:
        return None
    if len(ship_pts) == 1:
        return f"https://www.google.com/maps?q={ship_pts[0]['lat']},{ship_pts[0]['lon']}&z=16"
    centre = ship_pts[0]
    return f"https://www.google.com/maps/search/?api=1&query={centre['lat']},{centre['lon']}"


def load_image(uploaded_file):
    """Returns (PIL.Image RGB, geo_dict | None, geo_debug_str)."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    geo, geo_debug = None, "Not a TIFF file"
    if suffix in (".tif", ".tiff"):
        geo, geo_debug = extract_geo(tmp_path)

    # 1 · PIL (works for most formats)
    try:
        img = Image.open(tmp_path).convert("RGB")
        img.load()
        try: os.remove(tmp_path)
        except: pass
        return img, geo, geo_debug
    except Exception:
        pass

    # 2 · OpenCV fallback
    try:
        cv_img = cv2.imread(tmp_path, cv2.IMREAD_UNCHANGED)
        if cv_img is not None:
            if len(cv_img.shape) == 2:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
            elif cv_img.shape[2] == 4:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
            else:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(robust_normalize(cv_img))
            try: os.remove(tmp_path)
            except: pass
            return img, geo, geo_debug
    except Exception:
        pass

    # 3 · tifffile (large multi-band TIFF)
    try:
        import tifffile
        raw = tifffile.imread(tmp_path)
        if raw is not None:
            if raw.ndim == 3 and raw.shape[2] > 3: raw = raw[:, :, :3]
            raw = robust_normalize(raw)
            if raw.ndim == 2: raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
            img = Image.fromarray(raw)
            try: os.remove(tmp_path)
            except: pass
            return img, geo, geo_debug
    except Exception:
        pass

    try: os.remove(tmp_path)
    except: pass
    raise ValueError(f"Cannot decode {uploaded_file.name}")


# ─── Detection ─────────────────────────────────────────────────────────────────
def _run_batch(batch_items, model, conf_t, iou_t, tile_sz, show_lbl, show_conf_box):
    """Run inference on one batch of crop dicts. Returns list of per-crop result dicts."""
    imgs = [np.ascontiguousarray(np.array(b["crop"], dtype=np.uint8)) for b in batch_items]
    results = model.predict(
        imgs, conf=conf_t, iou=iou_t, imgsz=tile_sz,
        device=DEVICE, verbose=False, half=(DEVICE == "cuda")
    )
    out = []
    for j, res in enumerate(results):
        info = batch_items[j]
        ox, oy = info["x1"], info["y1"]
        bxs = res.boxes
        if len(bxs) == 0:
            continue
        tile_cv = cv2.cvtColor(np.array(info["crop"]), cv2.COLOR_RGB2BGR)
        boxes_raw, tile_ships = [], []
        for box in bxs:
            bx1, by1, bx2, by2 = [float(v) for v in box.xyxy[0]]
            c   = float(box.conf[0])
            cls = int(box.cls[0])
            gx1, gy1 = bx1 + ox, by1 + oy
            gx2, gy2 = bx2 + ox, by2 + oy
            boxes_raw.append({"xyxy_global": [gx1, gy1, gx2, gy2],
                               "xyxy_local":  [bx1, by1, bx2, by2],
                               "conf": c, "cls": cls})
            tile_ships.append({"global_box": [gx1, gy1, gx2, gy2], "conf": c})
            th = max(1, tile_sz // 500)
            cv2.rectangle(tile_cv, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 210, 80), th)
            if show_lbl or show_conf_box:
                lbl = ("Ship " if show_lbl else "") + (f"{c:.2f}" if show_conf_box else "")
                cv2.putText(tile_cv, lbl.strip(),
                            (int(bx1), max(int(by1) - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.35, tile_sz / 1800),
                            (0, 210, 80), max(1, tile_sz // 1000))
        out.append({
            "image": cv2.cvtColor(tile_cv, cv2.COLOR_BGR2RGB),
            "x1": ox, "y1": oy,
            "ships": tile_ships,
            "boxes_raw": boxes_raw,
            "max_conf": max(s["conf"] for s in tile_ships),
        })
    return out


def detect(image_pil, model, conf_t, iou_t, tile_sz, min_sz,
           show_lbl, show_conf_box, pbar=None):
    """Returns (final_boxes, annotated_np, active_tiles_sorted_desc)."""
    import torchvision

    W, H = image_pil.size
    annotated = np.array(image_pil).copy()

    overlap = 0.15
    stride  = int(tile_sz * (1 - overlap))
    crops   = []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            x2 = min(x + tile_sz, W); y2 = min(y + tile_sz, H)
            x1 = max(0, x2 - tile_sz); y1 = max(0, y2 - tile_sz)
            crops.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                          "crop": image_pil.crop((x1, y1, x2, y2))})

    BATCH  = 12 if DEVICE == "cuda" else 6
    batches = [crops[i*BATCH:(i+1)*BATCH] for i in range(math.ceil(len(crops)/BATCH))]
    total  = len(batches)

    all_boxes, all_scores, all_extra = [], [], []
    active_tiles = []

    futures_map = {}
    with ThreadPoolExecutor(max_workers=1) as exe:
        for bi, batch in enumerate(batches):
            fut = exe.submit(_run_batch, batch, model, conf_t, iou_t, tile_sz, show_lbl, show_conf_box)
            futures_map[fut] = bi

        done_count = 0
        for fut in as_completed(futures_map):
            done_count += 1
            if pbar:
                pbar.progress(done_count / max(1, total),
                              text=f"🔍 Processed batch {done_count}/{total} on {DEVICE.upper()}…")
            try:
                for tile_res in fut.result():
                    active_tiles.append(tile_res)
                    for br in tile_res["boxes_raw"]:
                        gx1,gy1,gx2,gy2 = br["xyxy_global"]
                        w_box = gx2 - gx1; h_box = gy2 - gy1
                        if w_box < min_sz or h_box < min_sz:
                            continue
                        all_boxes.append([gx1, gy1, gx2, gy2])
                        all_scores.append(br["conf"])
                        all_extra.append({"cls": br["cls"], "conf": br["conf"]})
            except Exception:
                pass

    if not all_boxes:
        return [], annotated, []

    bt   = torch.tensor(all_boxes,  dtype=torch.float32)
    st_  = torch.tensor(all_scores, dtype=torch.float32)
    keep = torchvision.ops.nms(bt, st_, iou_t).tolist()

    final_boxes = []
    for i in keep:
        fb   = bt[i].tolist()
        conf = st_[i].item()
        final_boxes.append({"xyxy": fb, "conf": conf, "cls": all_extra[i]["cls"]})
        x1, y1, x2, y2 = map(int, fb)
        th = max(2, W // 800)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 210, 80), th)
        if show_lbl or show_conf_box:
            lbl = ("Ship " if show_lbl else "") + (f"{conf:.2f}" if show_conf_box else "")
            cv2.putText(annotated, lbl.strip(), (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.4, W / 1500),
                        (0, 210, 80), max(1, W // 1000))

    active_tiles.sort(key=lambda t: t["max_conf"], reverse=True)
    return final_boxes, annotated, active_tiles


# ─── Map ───────────────────────────────────────────────────────────────────────
def render_map(df_pts):
    try:
        import pydeck as pdk
        avg_lat = df_pts["lat"].mean()
        avg_lon = df_pts["lon"].mean()
        span = max(df_pts["lat"].max()-df_pts["lat"].min(),
                   df_pts["lon"].max()-df_pts["lon"].min(), 0.001)
        zoom = max(1, min(16, int(math.log2(180/span))))
        layer = pdk.Layer(
            "ScatterplotLayer", df_pts,
            get_position="[lon, lat]",
            get_color="[0, 210, 80, 230]",
            get_radius=max(25, span * 2500),
            pickable=True, stroked=True,
            get_line_color="[0, 255, 120]",
            line_width_min_pixels=2,
        )
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(
                latitude=avg_lat, longitude=avg_lon,
                zoom=zoom, pitch=40),
            map_style="mapbox://styles/mapbox/satellite-streets-v12",
            tooltip={"text": "🚢 Ship\nLat: {lat}\nLon: {lon}\nConf: {conf}"},
        ))
    except Exception:
        st.map(df_pts.rename(columns={"lon": "longitude", "lat": "latitude"}))


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;
      color:#00d4ff;text-transform:uppercase;letter-spacing:.22em;margin-bottom:16px;
      padding:8px 0;border-bottom:1px solid #1a2e48">⚙ Detection Control</div>""",
      unsafe_allow_html=True)

    conf_threshold = st.slider("Confidence Threshold", 0.10, 0.95, 0.35, 0.01,
        help="Minimum YOLO confidence to keep a detection.")
    iou_threshold  = st.slider("IoU / NMS Threshold",  0.10, 0.95, 0.40, 0.01,
        help="Suppress overlapping boxes. Lower = stricter deduplication.")
    min_ship_px    = st.slider("Min Ship Size (px)",    5, 150, 15, 5,
        help="Ignore detections smaller than this width/height.")
    tile_size      = st.select_slider("Tile Resolution (px)",
                                      options=[320, 416, 512, 640, 768, 1024],
                                      value=640,
                                      help="Larger tiles = more context, slower. 640 is optimal.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;
      color:#00d4ff;text-transform:uppercase;letter-spacing:.22em;margin-bottom:12px">
      🖥 Display</div>""", unsafe_allow_html=True)

    show_labels    = st.toggle("Show Labels on Boxes",      value=True)
    show_conf_box  = st.toggle("Show Confidence on Boxes",  value=True)
    show_original  = st.toggle("Show Original Image",       value=True)
    show_detected  = st.toggle("Show Detected Image",       value=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    gpu_txt = (f"🟢 GPU · {torch.cuda.get_device_name(0)}"
               if DEVICE == "cuda" else "🟡 CPU mode (no CUDA detected)")
    st.markdown(f"""<div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#3d5a7a;line-height:1.8">
      Device: <span style="color:#c5d5e8">{gpu_txt}</span><br>
      Model:  <span style="color:#00d4ff">{HF_REPO}</span><br>
      Upload: <span style="color:#c5d5e8">JPG · PNG · TIFF up to 2 GB</span></div>""",
      unsafe_allow_html=True)


# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div class="hdr-ico">🛳️</div>
  <div>
    <h1>ShipScan</h1>
    <p>Maritime Object Detection · GPU YOLO Engine · Native GeoTIFF GPS</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Upload ────────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload satellite imagery  —  GeoTIFF gives real GPS · JPG/PNG gives detection only",
    type=["jpg", "jpeg", "png", "tiff", "tif"],
    accept_multiple_files=True,
    label_visibility="visible",
)

if not uploaded_files:
    st.markdown("""
    <div class="upload-hint">
      <div style="font-size:3.5rem;filter:drop-shadow(0 0 24px #00d4ff60)">🌊</div>
      <h3 style="font-size:1.1rem;color:#c5d5e8;margin:0;font-weight:700">No imagery loaded</h3>
      <p style="font-family:'JetBrains Mono',monospace;font-size:.72rem;margin:0;
         max-width:340px;line-height:1.8;color:#3d5a7a">
        Upload a <b style="color:#00d4ff">GeoTIFF (.tif / .tiff)</b> for real-world GPS ship coordinates.<br>
        Upload <b style="color:#c5d5e8">JPG / PNG</b> for detection without GPS.<br>
        <span style="color:#1a2e48">━━━━━━━━━━━━━━━━━━━━━━</span><br>
        Files up to <b style="color:#00ff99">2 GB</b> supported.
      </p>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Load model with custom spinner ──
spin = custom_spinner("Loading Model…")
model = load_model()
spin.empty()

# ─── Per-file processing ───────────────────────────────────────────────────────
for idx, uf in enumerate(uploaded_files):
    if len(uploaded_files) > 1:
        st.markdown(f'<div class="slbl">File {idx+1} / {len(uploaded_files)} — {uf.name}</div>',
                    unsafe_allow_html=True)

    # Load with custom spinner
    spin = custom_spinner("Loading Image…")
    try:
        image, geo, geo_debug = load_image(uf)
    except Exception as e:
        spin.empty()
        st.error(f"❌ Cannot load **{uf.name}**: {e}")
        continue
    spin.empty()

    W, H = image.size

    # Run inference
    pbar = st.progress(0.0, text="Initialising…")
    t0   = time.perf_counter()
    boxes, annotated, active_tiles = detect(
        image, model, conf_threshold, iou_threshold,
        tile_size, min_ship_px, show_labels, show_conf_box, pbar
    )
    pbar.progress(1.0, text="✅ Complete")
    elapsed = time.perf_counter() - t0
    pbar.empty()

    n     = len(boxes)
    confs = [b["conf"] for b in boxes]
    avg_c = np.mean(confs) if confs else 0.0
    top_c = max(confs)     if confs else 0.0

    # ── Metric strip ──────────────────────────────────────────────────────
    status_cls = "r" if n > 10 else "w" if n > 5 else "g"
    status_txt = "🔴 High density"   if n > 10 else ("🟡 Moderate" if n > 5 else "🟢 Normal")
    geo_cls    = "g" if geo else "w"
    geo_val    = f"EPSG:{geo['epsg']}" if geo and geo.get("epsg") else ("✓ GEO" if geo else "NO GEO")

    st.markdown(f"""
    <div class="mgrid">
      <div class="mc {status_cls}">
        <div class="lbl">Ships Detected</div>
        <div class="val">{n}</div>
        <div class="sub">{status_txt}</div>
      </div>
      <div class="mc g">
        <div class="lbl">Avg Confidence</div>
        <div class="val">{avg_c:.0%}</div>
        <div class="sub">model certainty</div>
      </div>
      <div class="mc w">
        <div class="lbl">Inference Time</div>
        <div class="val">{elapsed:.1f}<span style="font-size:.85rem;font-weight:400">s</span></div>
        <div class="sub">{DEVICE.upper()} · {tile_size}px tiles · {W}×{H}</div>
      </div>
      <div class="mc {geo_cls}">
        <div class="lbl">Geospatial CRS</div>
        <div class="val" style="font-size:1.1rem">{geo_val}</div>
        <div class="sub">{'Real GPS active' if geo else 'No CRS — upload GeoTIFF'}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Geo status ────────────────────────────────────────────────────────
    if geo:
        epsg_str = f"EPSG:{geo['epsg']}" if geo.get("epsg") else "Custom CRS"
        st.markdown(f"""<div class="geo-badge">🛰️ &nbsp;Native GeoTIFF Lock —
          <b>{epsg_str}</b> — Real GPS coordinates active</div>""",
          unsafe_allow_html=True)
    else:
        is_tiff = os.path.splitext(uf.name)[1].lower() in (".tif", ".tiff")
        if is_tiff:
            st.warning(f"⚠ **{uf.name}** is a TIFF but has no embedded CRS/projection. Map pinpointing unavailable.")
        else:
            st.info("ℹ Upload a GeoTIFF (.tif/.tiff) with embedded projection data to enable real-world GPS mapping.")

    with st.expander("🔬 GeoTIFF Metadata Diagnostics"):
        st.code(geo_debug, language="text")

    if n > 8:
        st.warning("⚠️ **High ship density detected** — possible cluster or fleet gathering.")

    # ══════════════════════════════════════════════════════════════════════
    # TABS — Results · Tile Explorer · GPS Map · Export
    # ══════════════════════════════════════════════════════════════════════
    tab_labels = ["📷 Results", f"🗂 Tile Explorer ({len(active_tiles)} tiles)", "🗺 GPS Map", "📋 Ship Log", "⬇ Export"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_labels)

    # ────────────────────────────────────────────────────────────────────
    # TAB 1 · Results
    # ────────────────────────────────────────────────────────────────────
    with tab1:
        if n == 0:
            st.info("No ships detected. Try lowering the **Confidence Threshold** in the sidebar.")
        else:
            show_cols = []
            if show_original:  show_cols.append("original")
            if show_detected:  show_cols.append("detected")

            if len(show_cols) == 0:
                st.warning("Both **Show Original Image** and **Show Detected Image** are turned OFF in the sidebar.")
            elif len(show_cols) == 2:
                c_l, c_r = st.columns(2)
                with c_l:
                    st.markdown('<div class="ipanel"><div class="ipanel-hdr">◈ Original</div>', unsafe_allow_html=True)
                    st.image(image, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with c_r:
                    st.markdown('<div class="ipanel"><div class="ipanel-hdr">◈ Detections</div>', unsafe_allow_html=True)
                    st.image(annotated, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ipanel"><div class="ipanel-hdr">◈ {"Original" if show_cols[0]=="original" else "Detections"}</div>', unsafe_allow_html=True)
                st.image(image if show_cols[0] == "original" else annotated, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────
    # TAB 2 · Tile Explorer  (sorted strongest → weakest)
    # ────────────────────────────────────────────────────────────────────
    with tab2:
        if not active_tiles:
            st.info("No tiles with detections found. Lower the confidence threshold and re-run.")
        else:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;flex-wrap:wrap">
              <span class="badge badge-blue">🗂 {len(active_tiles)} tiles with ships</span>
              <span class="badge badge-green">↓ Sorted by confidence</span>
              <span class="badge badge-purple">Tile {1} = Strongest</span>
            </div>""", unsafe_allow_html=True)

            cols3 = st.columns(3)
            for ti, td in enumerate(active_tiles):
                ship_count_tile = len(td["ships"])
                max_conf_tile   = td["max_conf"]
                conf_pct        = int(max_conf_tile * 100)

                if max_conf_tile >= 0.75:
                    badge_cls, grade = "badge-green",  "HIGH"
                elif max_conf_tile >= 0.50:
                    badge_cls, grade = "badge-yellow", "MED"
                else:
                    badge_cls, grade = "badge-red",    "LOW"

                with cols3[ti % 3]:
                    st.markdown(f"""
                    <div class="tile-card">
                      <div class="tile-hdr">
                        <div>
                          <span class="tile-num">#{ti+1:02d}</span>
                          <span style="color:var(--dim);margin:0 6px">·</span>
                          <span class="tile-ships">🚢 {ship_count_tile} ship{'s' if ship_count_tile!=1 else ''}</span>
                        </div>
                        <span class="badge {badge_cls}">{grade} {max_conf_tile:.0%}</span>
                      </div>
                      <div style="padding:4px 12px 2px">
                        <div class="conf-bar-wrap">
                          <div class="conf-bar" style="width:{conf_pct}%"></div>
                        </div>
                        <div class="tile-coord">Origin: X={td['x1']} Y={td['y1']} px</div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                    st.image(td["image"], use_container_width=True)

                    tile_pts_geo = []
                    if geo:
                        for ship in td["ships"]:
                            gx1, gy1, gx2, gy2 = ship["global_box"]
                            cx, cy = (gx1+gx2)/2, (gy1+gy2)/2
                            lat_s, lon_s = pixels_to_latlon(cx, cy, geo)
                            if lat_s is not None:
                                tile_pts_geo.append({"lat": lat_s, "lon": lon_s, "conf": ship["conf"]})

                    if tile_pts_geo:
                        btn_col1, btn_col2 = st.columns(2)
                        geojson_url  = make_geojson_url(tile_pts_geo, label=f"Tile#{ti+1} Ship")
                        gmaps_url    = make_google_maps_url(tile_pts_geo)
                        with btn_col1:
                            st.markdown(
                                f'<a href="{geojson_url}" target="_blank" style="'
                                'display:block;text-align:center;padding:7px 0;'
                                'background:rgba(0,255,153,.1);border:1px solid rgba(0,255,153,.4);'
                                'border-radius:6px;color:#00ff99;font-family:monospace;font-size:.68rem;'
                                'text-decoration:none;transition:all .2s;'
                                '">📍 View Tile on Map</a>',
                                unsafe_allow_html=True)
                        with btn_col2:
                            st.markdown(
                                f'<a href="{gmaps_url}" target="_blank" style="'
                                'display:block;text-align:center;padding:7px 0;'
                                'background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.4);'
                                'border-radius:6px;color:#00d4ff;font-family:monospace;font-size:.68rem;'
                                'text-decoration:none;transition:all .2s;'
                                '">🗺 Google Maps</a>',
                                unsafe_allow_html=True)
                    elif not geo:
                        st.caption("📍 Map button available for GeoTIFF files only")

                    with st.expander(f"▼ {ship_count_tile} ship detail(s) — Tile #{ti+1}"):
                        td_ships_sorted = sorted(td["ships"], key=lambda s: s["conf"], reverse=True)
                        for si, ship in enumerate(td_ships_sorted):
                            gx1, gy1, gx2, gy2 = ship["global_box"]
                            cx, cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
                            if geo:
                                lat, lon = pixels_to_latlon(cx, cy, geo)
                                gps = f"{lat:.6f}°, {lon:.6f}°" if lat else "CRS transform error"
                            else:
                                lat, lon = None, None
                                gps = f"Pixel ({int(cx)}, {int(cy)}) — no GeoTIFF"
                            st.markdown(f"""
                            <div class="ship-card">
                              <h4>🚢 Ship #{si+1} &nbsp;<span style="font-size:.7rem;color:var(--dim);font-weight:400">Tile #{ti+1}</span></h4>
                              <div class="coords">{gps}</div>
                              <div class="meta">Conf: {ship['conf']:.1%} &nbsp;·&nbsp;
                                {'Lat/Lon GPS' if geo and lat else 'Pixel coords — no GeoTIFF'}</div>
                            </div>""", unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────
    # TAB 3 · GPS Map
    # ────────────────────────────────────────────────────────────────────
    with tab3:
        if not geo:
            st.warning("🗺 GPS map requires a GeoTIFF with embedded CRS. Upload a georeferenced .tif/.tiff file.")
            st.info("Your file is not a GeoTIFF (or has no projection). For GPS mapping, use a georeferenced .tif/.tiff satellite image.")
        elif n == 0:
            st.info("No ships detected — nothing to plot on the map.")
        else:
            pts = []
            for b in boxes:
                x1, y1, x2, y2 = b["xyxy"]
                lat, lon = pixels_to_latlon((x1+x2)/2, (y1+y2)/2, geo)
                if lat is not None:
                    pts.append({"lat": lat, "lon": lon, "conf": round(b["conf"], 4)})

            if not pts:
                st.error("GPS projection failed for all ships. Check GeoTIFF Diagnostics expander above.")
            else:
                df_map = pd.DataFrame(pts)

                gsd = geo_pixel_size_m(geo)
                img_w_km = (geo["w"] * gsd / 1000) if gsd else None
                img_h_km = (geo["h"] * gsd / 1000) if gsd else None
                coverage_km2 = (img_w_km * img_h_km) if img_w_km else None

                lat_spread = df_map["lat"].max() - df_map["lat"].min()
                lon_spread = df_map["lon"].max() - df_map["lon"].min()
                spread_km  = haversine_km(
                    df_map["lat"].min(), df_map["lon"].min(),
                    df_map["lat"].max(), df_map["lon"].max()
                ) if len(pts) > 1 else 0.0

                min_dist_km = None
                if len(pts) > 1:
                    dists = []
                    for i in range(len(pts)):
                        for j in range(i+1, len(pts)):
                            d = haversine_km(pts[i]["lat"], pts[i]["lon"],
                                             pts[j]["lat"], pts[j]["lon"])
                            dists.append(d)
                    min_dist_km = min(dists)

                gsd_str = f"{gsd:.1f} m/px" if gsd else "N/A"
                cov_str = f"{coverage_km2:.2f} km²" if coverage_km2 else "N/A"

                st.markdown(f"""
                <div class="mgrid" style="grid-template-columns:repeat(4,1fr);margin:0 0 16px">
                  <div class="mc p">
                    <div class="lbl">Image Centre</div>
                    <div class="val" style="font-size:0.8rem;word-break:break-all">
                      {df_map['lat'].mean():.4f}°<br>{df_map['lon'].mean():.4f}°</div>
                    <div class="sub">Lat / Lon</div>
                  </div>
                  <div class="mc g">
                    <div class="lbl">GSD (Ground Resolution)</div>
                    <div class="val" style="font-size:1.1rem">{gsd_str}</div>
                    <div class="sub">metres per pixel</div>
                  </div>
                  <div class="mc w">
                    <div class="lbl">Detection Spread</div>
                    <div class="val" style="font-size:1.1rem">{spread_km:.3f} km</div>
                    <div class="sub">max ship-to-ship dist</div>
                  </div>
                  <div class="mc {'g' if min_dist_km is not None else 'w'}">
                    <div class="lbl">Nearest Ships</div>
                    <div class="val" style="font-size:1.1rem">{f'{min_dist_km:.3f} km' if min_dist_km is not None else 'N/A'}</div>
                    <div class="sub">closest pair distance</div>
                  </div>
                </div>""", unsafe_allow_html=True)

                all_pts_geo = [{"lat": r["lat"], "lon": r["lon"], "conf": r["conf"]} for _, r in df_map.iterrows()]
                geojson_all_url = make_geojson_url(all_pts_geo, label="Ship")
                gmaps_centre    = make_google_maps_url(all_pts_geo)
                ob1, ob2, ob3 = st.columns([1, 1, 4])
                with ob1:
                    st.markdown(
                        f'<a href="{geojson_all_url}" target="_blank" style="'
                        'display:block;text-align:center;padding:9px 0;'
                        'background:rgba(0,255,153,.12);border:1px solid rgba(0,255,153,.4);'
                        'border-radius:8px;color:#00ff99;font-family:monospace;font-size:.72rem;'
                        'font-weight:600;text-decoration:none;'
                        '">📍 Open All Ships — geojson.io</a>',
                        unsafe_allow_html=True)
                with ob2:
                    st.markdown(
                        f'<a href="{gmaps_centre}" target="_blank" style="'
                        'display:block;text-align:center;padding:9px 0;'
                        'background:rgba(0,212,255,.12);border:1px solid rgba(0,212,255,.4);'
                        'border-radius:8px;color:#00d4ff;font-family:monospace;font-size:.72rem;'
                        'font-weight:600;text-decoration:none;'
                        '">🗺 Open in Google Maps</a>',
                        unsafe_allow_html=True)
                st.caption("geojson.io shows all ships colour-coded by confidence · Google Maps centres on the detection area — both free, no sign-in needed")

                st.markdown('<div class="slbl">🗺 Interactive Map — Switch Layers via top-right control</div>', unsafe_allow_html=True)
                st.caption("Layers: 🛰 Satellite · 🗺 Street · 🌑 Dark · 🌊 Ocean · 🔥 Heatmap · 📍 Markers · 📷 Footprint | 📐 Ruler tool (bottom-left) | ⛶ Fullscreen (top-left)")
                render_map_folium(df_map, geo)

                st.markdown('<div class="slbl">📋 All GPS Coordinates</div>', unsafe_allow_html=True)
                display_df = df_map.copy()
                display_df.index = display_df.index + 1
                display_df.index.name = "Rank"
                display_df.columns = ["Latitude", "Longitude", "Confidence"]
                display_df["Confidence"] = display_df["Confidence"].map(lambda x: f"{x:.1%}")
                st.dataframe(display_df, use_container_width=True, height=240)

    # ────────────────────────────────────────────────────────────────────
    # TAB 4 · Ship Log
    # ────────────────────────────────────────────────────────────────────
    with tab4:
        if n == 0:
            st.info("No detections to log.")
        else:
            boxes_sorted = sorted(boxes, key=lambda b: b["conf"], reverse=True)
            rows = ""
            for i, b in enumerate(boxes_sorted):
                x1, y1, x2, y2 = b["xyxy"]
                c  = b["conf"]
                cx, cy = (x1+x2)/2, (y1+y2)/2
                if geo:
                    lat, lon = pixels_to_latlon(cx, cy, geo)
                    gps = f"{lat:.6f}, {lon:.6f}" if lat else "CRS transform error"
                else:
                    lat, lon = None, None
                    gps = f"px ({int(cx)}, {int(cy)}) — no GeoTIFF"
                bw, bh = int(x2-x1), int(y2-y1)
                bar = int(c * 70)
                grade_color = "00ff99" if c > 0.75 else "ffb400" if c > 0.5 else "ff3d3d"
                grade_txt   = "High"   if c > 0.75 else "Med"    if c > 0.5 else "Low"
                rows += f"""<tr>
                  <td style="color:#00d4ff;font-weight:700">#{i+1:02d}</td>
                  <td><span style="display:inline-block;width:{bar}px;height:5px;
                      background:linear-gradient(90deg,#00ff99,#00d4ff);border-radius:3px;
                      vertical-align:middle;margin-right:8px"></span>{c:.1%}</td>
                  <td>{bw}×{bh} px</td>
                  <td style="font-family:'JetBrains Mono',monospace;color:#00ff99;font-size:.67rem">{gps}</td>
                  <td style="color:#{grade_color}">{grade_txt}</td>
                </tr>"""
            st.markdown(f"""<table class="det-table">
              <thead><tr><th>#</th><th>Confidence</th><th>Size</th><th>GPS / Pixel</th><th>Grade</th></tr></thead>
              <tbody>{rows}</tbody></table>""", unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────
    # TAB 5 · Export
    # ────────────────────────────────────────────────────────────────────
    with tab5:
        rows_export = []
        for i, b in enumerate(sorted(boxes, key=lambda b: b["conf"], reverse=True)):
            x1, y1, x2, y2 = b["xyxy"]
            lat, lon = pixels_to_latlon((x1+x2)/2, (y1+y2)/2, geo) if geo else (None, None)
            rows_export.append({
                "Rank": i+1,
                "Confidence": round(b["conf"], 4),
                "Latitude":   lat,
                "Longitude":  lon,
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "Width_px": int(x2-x1), "Height_px": int(y2-y1),
            })
        df_ex = pd.DataFrame(rows_export)

        st.markdown('<div class="slbl">Download</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 4])
        with c1:
            st.download_button(
                "⬇ CSV — Ship Data",
                df_ex.to_csv(index=False).encode(),
                f"ships_{uf.name}.csv", "text/csv",
                use_container_width=True)
        with c2:
            buf = io.BytesIO()
            Image.fromarray(annotated).save(buf, format="PNG")
            st.download_button(
                "⬇ PNG — Annotated",
                buf.getvalue(),
                f"annotated_{uf.name}.png", "image/png",
                use_container_width=True)

        if not df_ex.empty:
            st.markdown('<div class="slbl">Preview</div>', unsafe_allow_html=True)
            st.dataframe(df_ex, use_container_width=True, height=300)

    if idx < len(uploaded_files) - 1:
        st.divider()