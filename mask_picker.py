#!/usr/bin/env python3
"""
Mask Picker Web — Outil web pour définir les zones à masquer par poste.

Ouvre un serveur web local et affiche la première frame d'une vidéo du poste.
Permet de dessiner des polygones à masquer directement dans le navigateur.
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
  .btn-cancel {{ background: #8b5cf6; color: #fff; }}
  .btn-cancel:hover {{ background: #7c3aed; }}
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
    <button class="btn btn-cancel" id="btnCancel" onclick="cancelPoly()" style="display:none;">✖ Annuler poly</button>
    <button class="btn btn-undo" onclick="undoZone()">↩ Annuler</button>
    <button class="btn btn-reset" onclick="resetZones()">🔄 Reset</button>
    <button class="btn btn-save" onclick="saveZones()">💾 Sauvegarder</button>
  </div>
</div>

<div class="help">
  🖱️ <b>Clic gauche</b> = ajouter un sommet &nbsp;|&nbsp;
  <b>Double-clic</b> ou clic proche du 1er point = fermer le polygone &nbsp;|&nbsp;
  <b>Échap</b> = annuler le polygone en cours
</div>

<div class="canvas-container">
  <canvas id="canvas"></canvas>
</div>

<div class="status" id="status">Prêt — cliquez pour dessiner des polygones sur les zones à masquer</div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const zoneCountEl = document.getElementById('zoneCount');
const btnCancel = document.getElementById('btnCancel');

const img = new Image();
const frameW = {frame_w};
const frameH = {frame_h};

// Scale to fit screen
const maxW = window.innerWidth - 40;
const maxH = window.innerHeight - 160;
const scaleX = maxW / frameW;
const scaleY = maxH / frameH;
const scale = Math.min(scaleX, scaleY, 1);

const displayW = Math.floor(frameW * scale);
const displayH = Math.floor(frameH * scale);

canvas.width = displayW;
canvas.height = displayH;

// zones = array of polygons. Each polygon = {{ points: [{{x, y}}, ...] }}
// Auto-convert old rect format (x1,y1,x2,y2) to polygon format on load
let zonesRaw = {existing_zones};
let zones = zonesRaw.map(z => {{
  if (z.points) return z;
  // Convert rectangle to polygon
  return {{ points: [
    {{x: z.x1, y: z.y1}},
    {{x: z.x2, y: z.y1}},
    {{x: z.x2, y: z.y2}},
    {{x: z.x1, y: z.y2}}
  ]}};
}});

let currentPoly = [];   // points being drawn (canvas coords)
let mousePos = null;     // current mouse position for live preview
const CLOSE_DIST = 12;   // px distance to snap-close to first point

img.onload = () => {{ redraw(); }};
img.src = 'data:image/jpeg;base64,{frame_b64}';

function toReal(cx, cy) {{
  return {{ x: Math.round(cx / scale), y: Math.round(cy / scale) }};
}}

function toCanvas(rx, ry) {{
  return [rx * scale, ry * scale];
}}

function dist(ax, ay, bx, by) {{
  return Math.sqrt((ax - bx) ** 2 + (ay - by) ** 2);
}}

function redraw() {{
  ctx.drawImage(img, 0, 0, displayW, displayH);

  // Draw saved polygons
  for (const z of zones) {{
    ctx.beginPath();
    const pts = z.points;
    for (let i = 0; i < pts.length; i++) {{
      const [cx, cy] = toCanvas(pts[i].x, pts[i].y);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }}
    ctx.closePath();
    ctx.fillStyle = 'rgba(255, 0, 0, 0.30)';
    ctx.fill();
    ctx.strokeStyle = '#ff0000';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw vertices
    for (const pt of pts) {{
      const [cx, cy] = toCanvas(pt.x, pt.y);
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#ff4444';
      ctx.fill();
    }}
  }}

  // Draw current polygon being built
  if (currentPoly.length > 0) {{
    ctx.beginPath();
    ctx.moveTo(currentPoly[0].cx, currentPoly[0].cy);
    for (let i = 1; i < currentPoly.length; i++) {{
      ctx.lineTo(currentPoly[i].cx, currentPoly[i].cy);
    }}
    // Line to mouse cursor
    if (mousePos) {{
      ctx.lineTo(mousePos.cx, mousePos.cy);
    }}
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Fill preview
    if (currentPoly.length >= 3) {{
      ctx.beginPath();
      ctx.moveTo(currentPoly[0].cx, currentPoly[0].cy);
      for (let i = 1; i < currentPoly.length; i++) {{
        ctx.lineTo(currentPoly[i].cx, currentPoly[i].cy);
      }}
      if (mousePos) ctx.lineTo(mousePos.cx, mousePos.cy);
      ctx.closePath();
      ctx.fillStyle = 'rgba(0, 255, 0, 0.12)';
      ctx.fill();
    }}

    // Draw vertices
    for (let i = 0; i < currentPoly.length; i++) {{
      ctx.beginPath();
      ctx.arc(currentPoly[i].cx, currentPoly[i].cy, i === 0 ? 6 : 4, 0, Math.PI * 2);
      ctx.fillStyle = i === 0 ? '#00ffaa' : '#00ff00';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1;
      ctx.stroke();
    }}

    // Highlight first point if mouse is close (snap zone)
    if (mousePos && currentPoly.length >= 3) {{
      const d = dist(mousePos.cx, mousePos.cy, currentPoly[0].cx, currentPoly[0].cy);
      if (d < CLOSE_DIST) {{
        ctx.beginPath();
        ctx.arc(currentPoly[0].cx, currentPoly[0].cy, 10, 0, Math.PI * 2);
        ctx.strokeStyle = '#00ffaa';
        ctx.lineWidth = 2;
        ctx.stroke();
      }}
    }}
  }}

  zoneCountEl.textContent = zones.length + ' zone(s)';
  btnCancel.style.display = currentPoly.length > 0 ? 'inline-block' : 'none';
}}

function closePoly() {{
  if (currentPoly.length < 3) {{
    statusEl.textContent = 'Il faut au moins 3 points pour fermer un polygone';
    statusEl.className = 'status error';
    return;
  }}
  const points = currentPoly.map(p => toReal(p.cx, p.cy));
  zones.push({{ points }});
  const n = points.length;
  statusEl.textContent = `Polygone ajouté (${{n}} sommets)`;
  statusEl.className = 'status';
  currentPoly = [];
  mousePos = null;
  redraw();
}}

function cancelPoly() {{
  currentPoly = [];
  mousePos = null;
  statusEl.textContent = 'Polygone en cours annulé';
  statusEl.className = 'status';
  redraw();
}}

canvas.addEventListener('click', (e) => {{
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;

  // If close to first point and enough vertices → close
  if (currentPoly.length >= 3) {{
    const d = dist(cx, cy, currentPoly[0].cx, currentPoly[0].cy);
    if (d < CLOSE_DIST) {{
      closePoly();
      return;
    }}
  }}

  currentPoly.push({{ cx, cy }});
  statusEl.textContent = `Sommet ${{currentPoly.length}} ajouté — ${{currentPoly.length < 3 ? 'ajoutez encore ' + (3 - currentPoly.length) + ' point(s) minimum' : 'double-clic ou clic sur le 1er point pour fermer'}}`;
  statusEl.className = 'status';
  redraw();
}});

canvas.addEventListener('dblclick', (e) => {{
  e.preventDefault();
  // Remove the last single-click point that was just added by the click event
  if (currentPoly.length > 3) {{
    currentPoly.pop();
  }}
  closePoly();
}});

canvas.addEventListener('mousemove', (e) => {{
  if (currentPoly.length === 0) return;
  const rect = canvas.getBoundingClientRect();
  mousePos = {{ cx: e.clientX - rect.left, cy: e.clientY - rect.top }};
  redraw();
}});

document.addEventListener('keydown', (e) => {{
  if (e.key === 'Escape') {{
    cancelPoly();
  }}
  // Ctrl+Z to undo last vertex or last zone
  if (e.key === 'z' && (e.ctrlKey || e.metaKey)) {{
    e.preventDefault();
    if (currentPoly.length > 0) {{
      currentPoly.pop();
      if (currentPoly.length === 0) mousePos = null;
      statusEl.textContent = currentPoly.length > 0 ? `Sommet retiré (reste: ${{currentPoly.length}})` : 'Polygone en cours annulé';
      statusEl.className = 'status';
      redraw();
    }} else {{
      undoZone();
    }}
  }}
}});

function undoZone() {{
  if (currentPoly.length > 0) {{
    cancelPoly();
    return;
  }}
  if (zones.length > 0) {{
    zones.pop();
    statusEl.textContent = `Zone supprimée (reste: ${{zones.length}})`;
    statusEl.className = 'status';
    redraw();
  }}
}}

function resetZones() {{
  zones = [];
  currentPoly = [];
  mousePos = null;
  statusEl.textContent = 'Toutes les zones supprimées';
  statusEl.className = 'status';
  redraw();
}}

function saveZones() {{
  if (currentPoly.length > 0) {{
    statusEl.textContent = '⚠️ Fermez le polygone en cours avant de sauvegarder';
    statusEl.className = 'status error';
    return;
  }}
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
