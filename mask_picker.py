#!/usr/bin/env python3
"""
Mask Picker Web — Outil web pour définir les zones à masquer par poste.

Ouvre un serveur web local et affiche la première frame d'une vidéo du poste.
Permet de dessiner des rectangles à masquer directement dans le navigateur.
Sauvegarde le résultat dans mask.json dans le dossier du poste.

Usage:
    python3 mask_picker.py data/videos/CA1
    python3 mask_picker.py data/videos/CA1 --port 8080
    python3 mask_picker.py data/videos/CA1 --video 2_20260228_100800_0025d5.avi
"""

import os
import sys
import json
import argparse
import base64
import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Globals set by main()
POSTE_DIR = ""
POSTE_NAME = ""
FRAME_B64 = ""
FRAME_W = 0
FRAME_H = 0
EXISTING_ZONES = []


HTML_PAGE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mask Picker — {poste_name}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0f0f23;
    color: #e0e0e0;
    min-height: 100vh;
  }}
  .header {{
    background: linear-gradient(135deg, #1a1a3e, #2d1b69);
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 2px solid #6c3ce0;
    flex-wrap: wrap;
    gap: 10px;
  }}
  .header h1 {{
    font-size: 20px;
    font-weight: 600;
    color: #fff;
  }}
  .header h1 span {{ color: #a78bfa; }}
  .controls {{
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
  }}
  .btn {{
    padding: 8px 18px;
    border: none;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }}
  .btn-save {{ background: #22c55e; color: #fff; }}
  .btn-save:hover {{ background: #16a34a; transform: scale(1.03); }}
  .btn-undo {{ background: #f59e0b; color: #000; }}
  .btn-undo:hover {{ background: #d97706; }}
  .btn-reset {{ background: #ef4444; color: #fff; }}
  .btn-reset:hover {{ background: #dc2626; }}
  .badge {{
    background: #6c3ce0;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
  }}
  .canvas-container {{
    display: flex;
    justify-content: center;
    padding: 16px;
    position: relative;
  }}
  canvas {{
    cursor: crosshair;
    border: 2px solid #333;
    border-radius: 4px;
    max-width: 100%;
    height: auto;
  }}
  .status {{
    text-align: center;
    padding: 10px;
    font-size: 14px;
    color: #9ca3af;
  }}
  .status.success {{ color: #22c55e; font-weight: 600; }}
  .status.error {{ color: #ef4444; }}
  .help {{
    text-align: center;
    padding: 8px;
    color: #6b7280;
    font-size: 13px;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>🎭 Mask Picker — <span>{poste_name}</span></h1>
  <div class="controls">
    <span class="badge" id="zoneCount">0 zone(s)</span>
    <button class="btn btn-undo" onclick="undoZone()">↩ Annuler</button>
    <button class="btn btn-reset" onclick="resetZones()">🔄 Reset</button>
    <button class="btn btn-save" onclick="saveZones()">💾 Sauvegarder</button>
  </div>
</div>

<div class="help">
  Cliquez et glissez pour dessiner des rectangles rouges sur les zones à masquer (objets fixes, plastiques, etc.)
</div>

<div class="canvas-container">
  <canvas id="canvas"></canvas>
</div>

<div class="status" id="status">Prêt — dessinez des zones à masquer</div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const zoneCountEl = document.getElementById('zoneCount');

const img = new Image();
const frameW = {frame_w};
const frameH = {frame_h};

// Scale to fit screen
const maxW = window.innerWidth - 40;
const maxH = window.innerHeight - 160;
const scaleX = maxW / frameW;
const scaleY = maxH / frameH;
const scale = Math.min(scaleX, scaleY, 1); // Never upscale

const displayW = Math.floor(frameW * scale);
const displayH = Math.floor(frameH * scale);

canvas.width = displayW;
canvas.height = displayH;

let zones = {existing_zones};
let drawing = false;
let startX, startY, curX, curY;

img.onload = () => {{ redraw(); }};
img.src = 'data:image/jpeg;base64,{frame_b64}';

function toReal(cx, cy) {{
  return [Math.round(cx / scale), Math.round(cy / scale)];
}}

function toCanvas(rx, ry) {{
  return [rx * scale, ry * scale];
}}

function redraw() {{
  ctx.drawImage(img, 0, 0, displayW, displayH);

  // Draw saved zones
  for (const z of zones) {{
    const [cx1, cy1] = toCanvas(z.x1, z.y1);
    const [cx2, cy2] = toCanvas(z.x2, z.y2);
    ctx.fillStyle = 'rgba(255, 0, 0, 0.35)';
    ctx.fillRect(cx1, cy1, cx2 - cx1, cy2 - cy1);
    ctx.strokeStyle = '#ff0000';
    ctx.lineWidth = 2;
    ctx.strokeRect(cx1, cy1, cx2 - cx1, cy2 - cy1);
  }}

  // Draw current rectangle
  if (drawing) {{
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(startX, startY, curX - startX, curY - startY);
    ctx.setLineDash([]);
  }}

  zoneCountEl.textContent = zones.length + ' zone(s)';
}}

canvas.addEventListener('mousedown', (e) => {{
  const rect = canvas.getBoundingClientRect();
  startX = e.clientX - rect.left;
  startY = e.clientY - rect.top;
  drawing = true;
}});

canvas.addEventListener('mousemove', (e) => {{
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  curX = e.clientX - rect.left;
  curY = e.clientY - rect.top;
  redraw();
}});

canvas.addEventListener('mouseup', (e) => {{
  if (!drawing) return;
  drawing = false;
  const rect = canvas.getBoundingClientRect();
  const endX = e.clientX - rect.left;
  const endY = e.clientY - rect.top;

  const [rx1, ry1] = toReal(Math.min(startX, endX), Math.min(startY, endY));
  const [rx2, ry2] = toReal(Math.max(startX, endX), Math.max(startY, endY));

  if (Math.abs(rx2 - rx1) > 5 && Math.abs(ry2 - ry1) > 5) {{
    zones.push({{ x1: rx1, y1: ry1, x2: rx2, y2: ry2 }});
    statusEl.textContent = `Zone ajoutée: (${{rx1}},${{ry1}}) → (${{rx2}},${{ry2}})`;
    statusEl.className = 'status';
  }}
  redraw();
}});

function undoZone() {{
  if (zones.length > 0) {{
    zones.pop();
    statusEl.textContent = `Zone supprimée (reste: ${{zones.length}})`;
    statusEl.className = 'status';
    redraw();
  }}
}}

function resetZones() {{
  zones = [];
  statusEl.textContent = 'Toutes les zones supprimées';
  statusEl.className = 'status';
  redraw();
}}

function saveZones() {{
  fetch('/save', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ mask_zones: zones }})
  }})
  .then(r => r.json())
  .then(data => {{
    if (data.ok) {{
      statusEl.textContent = `✅ Sauvegardé ! ${{zones.length}} zone(s) dans mask.json`;
      statusEl.className = 'status success';
    }} else {{
      statusEl.textContent = '❌ Erreur: ' + data.error;
      statusEl.className = 'status error';
    }}
  }})
  .catch(err => {{
    statusEl.textContent = '❌ Erreur réseau: ' + err;
    statusEl.className = 'status error';
  }});
}}
</script>
</body>
</html>"""


class MaskHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path.startswith('/?'):
            html = HTML_PAGE.format(
                poste_name=POSTE_NAME,
                frame_w=FRAME_W,
                frame_h=FRAME_H,
                frame_b64=FRAME_B64,
                existing_zones=json.dumps(EXISTING_ZONES)
            )
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/save':
            try:
                length = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(length))
                zones = body.get('mask_zones', [])

                mask_file = os.path.join(POSTE_DIR, "mask.json")
                data = {
                    "poste": POSTE_NAME,
                    "mask_zones": zones,
                    "frame_width": FRAME_W,
                    "frame_height": FRAME_H
                }
                with open(mask_file, 'w') as f:
                    json.dump(data, f, indent=2)

                print(f"✅ Mask sauvegardé: {mask_file} ({len(zones)} zones)")

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "zones": len(zones)}).encode())

            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def main():
    global POSTE_DIR, POSTE_NAME, FRAME_B64, FRAME_W, FRAME_H, EXISTING_ZONES

    parser = argparse.ArgumentParser(
        description="Mask Picker Web — Définir les zones à masquer pour un poste"
    )
    parser.add_argument(
        "poste_dir",
        help="Chemin vers le dossier du poste (ex: data/videos/CA1)"
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Nom de la vidéo à utiliser (par défaut: la plus grosse)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="Port du serveur web (défaut: 8888)"
    )
    args = parser.parse_args()

    POSTE_DIR = os.path.abspath(args.poste_dir)
    POSTE_NAME = os.path.basename(POSTE_DIR)

    if not os.path.isdir(POSTE_DIR):
        print(f"❌ Dossier introuvable: {POSTE_DIR}")
        sys.exit(1)

    # Trouver une vidéo
    if args.video:
        video_path = os.path.join(POSTE_DIR, args.video)
    else:
        videos = [f for f in os.listdir(POSTE_DIR)
                  if f.lower().endswith(('.avi', '.mp4', '.mkv', '.mov', '.264', '.265', '.h264', '.h265'))]
        if not videos:
            print(f"❌ Aucune vidéo trouvée dans {POSTE_DIR}")
            sys.exit(1)
        videos.sort(key=lambda f: os.path.getsize(os.path.join(POSTE_DIR, f)), reverse=True)
        video_path = os.path.join(POSTE_DIR, videos[0])

    print(f"📹 Vidéo: {os.path.basename(video_path)}")

    # Extraire une frame (on avance un peu pour avoir une image plus intéressante)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Aller à 10% de la vidéo pour avoir une frame avec du contenu
    target_frame = min(total_frames // 10, 500)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"❌ Impossible de lire la vidéo: {video_path}")
        sys.exit(1)

    FRAME_H, FRAME_W = frame.shape[:2]

    # Encoder en base64
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    FRAME_B64 = base64.b64encode(buffer).decode('utf-8')

    # Charger les zones existantes
    mask_file = os.path.join(POSTE_DIR, "mask.json")
    if os.path.exists(mask_file):
        try:
            with open(mask_file, 'r') as f:
                data = json.load(f)
            EXISTING_ZONES = data.get("mask_zones", [])
            print(f"📂 Mask existant: {len(EXISTING_ZONES)} zone(s)")
        except Exception:
            EXISTING_ZONES = []

    # Lancer le serveur
    server = HTTPServer(('0.0.0.0', args.port), MaskHandler)
    print(f"\n{'=' * 50}")
    print(f"🎭 MASK PICKER — {POSTE_NAME}")
    print(f"{'=' * 50}")
    print(f"🌐 Ouvrez dans votre navigateur:")
    print(f"   http://localhost:{args.port}")
    print(f"   Frame: {FRAME_W}x{FRAME_H}")
    print(f"\n   Ctrl+C pour arrêter le serveur")
    print(f"{'=' * 50}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Serveur arrêté")
        server.server_close()


if __name__ == "__main__":
    main()
