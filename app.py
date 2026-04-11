"""
TRIYAMBAKA — The Three-Eyed AI Surveillance System System
Built with: Python Flask, OpenCV, Google Gemini Vision API
College final year project
"""

import os, base64, json, time, threading, smtplib, logging, re
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import requests
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

SITE_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

state = {
    "camera_active": False,
    "auto_mode": False,
    "frames_analyzed": 0,
    "alerts_triggered": 0,
    "last_alert_time": 0,
    "latest_result": None,
    "session_start": time.time(),
    "log_entries": [],
}

camera_obj   = None
latest_frame = None
frame_lock   = threading.Lock()
camera_lock  = threading.Lock()

# ── Helpers ────────────────────────────────────────────────────────────────────

def add_log(message, level="info"):
    entry = {"time": datetime.now().strftime("%H:%M:%S"), "msg": message, "level": level}
    state["log_entries"].append(entry)
    if len(state["log_entries"]) > 300:
        state["log_entries"].pop(0)
    log.info(f"[{level.upper()}] {message}")


def resize_image(img_bytes, max_size=512):
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def safe_parse_json(raw):
    """Bulletproof JSON extraction from any Gemini response."""
    # Clean markdown fences
    raw = re.sub(r'```json|```', '', raw).strip()

    # Try direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Find outermost { }
    try:
        start = raw.index('{')
        end   = raw.rindex('}') + 1
        return json.loads(raw[start:end])
    except Exception:
        pass

    # Extended regex fallback handles any format including plain text
    violence = bool(re.search(r'violence_detected[":\s]+true', raw, re.I))
    conf_m   = re.search(r'confidence[":\s]+(\d+)', raw, re.I)
    level_m  = re.search(r'threat_level[":\s]+"?(\w+)"?', raw, re.I)
    desc_m   = re.search(r'description[":\s]+"([^"]{3,})"', raw, re.I)

    return {
        "violence_detected": violence,
        "confidence": int(conf_m.group(1)) if conf_m else (70 if violence else 5),
        "threat_level": level_m.group(1) if level_m else ("high" if violence else "none"),
        "categories": ["physical_assault"] if violence else [],
        "description": desc_m.group(1) if desc_m else ("Violence detected" if violence else "No threat detected"),
        "action_required": violence,
        "details": "AI analysis completed."
    }


# ── Gemini Vision API ──────────────────────────────────────────────────────────

DETECTION_PROMPT = (
    'Analyze ONLY what you can SEE RIGHT NOW in this single image. '
    'Do NOT assume anything from previous frames. Each image is completely independent.\n'
    'Reply with ONLY a JSON object, no other text, no markdown:\n'
    '{"violence_detected":false,"confidence":0,"threat_level":"none",'
    '"categories":[],"description":"scene description",'
    '"action_required":false,"details":"assessment"}\n\n'
    'Fields: violence_detected=true/false, confidence=0-100, '
    'threat_level=none/low/medium/high/critical, '
    'categories=subset of [physical_assault,weapon_visible,crowd_aggression,'
    'property_destruction,threatening_behavior,self_harm,armed_robbery,domestic_violence], '
    'description=one sentence describing what you SEE NOW, '
    'action_required=true/false, details=two sentences.\n'
    'STRICT RULES:\n'
    '- Only flag violence you can CLEARLY SEE in this exact image\n'
    '- If the scene looks calm/normal NOW, violence_detected=false even if it was violent before\n'
    '- A knife held calmly = low threat. A knife being used to attack = high threat\n'
    '- Normal activity, empty rooms, people sitting/walking = false\n'
    '- Be honest about what you see right now, nothing more'
)

GEMINI_MODEL = "gemini-2.5-flash"
RESEND_API_KEY  = os.getenv("RESEND_API_KEY", "re_QWW73BDf_KKF2NzajRJ5wknTMAW1MmoTR")


def call_gemini_vision(api_key, image_bytes):
    image_bytes = resize_image(image_bytes)
    img_b64 = base64.b64encode(image_bytes).decode()

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent?key={api_key.strip()}")

    payload = {
        "contents": [{
            "parts": [
                {"text": DETECTION_PROMPT},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
            ]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 300,
            "topP": 0.1,
        }
    }

    resp = requests.post(url, json=payload, timeout=40)

    if not resp.ok:
        raise Exception(f"Gemini error {resp.status_code}: {resp.json().get('error',{}).get('message', resp.text)}")

    data  = resp.json()
    raw   = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    log.info(f"Gemini raw: {raw[:200]}")
    return safe_parse_json(raw)


def resolve_api_key(user_key):
    key = (user_key or "").strip()
    if key:
        return key
    if SITE_GEMINI_KEY:
        return SITE_GEMINI_KEY
    raise Exception("No API key. Enter your Gemini key in Settings.")


# ── Email ──────────────────────────────────────────────────────────────────────
def send_email_alert(to_email, result, location="Camera 01"):
    def _send():
        try:
            timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            threat      = result.get('threat_level', '').upper()
            confidence  = result.get('confidence', 0)
            description = result.get('description', '')
            details     = result.get('details', '')
            html = f"""<html><body style="font-family:Arial;background:#0d1117;color:#c8d8e8;padding:24px">
<div style="max-width:600px;margin:auto;border:1px solid #1c2a38;border-radius:8px;overflow:hidden">
<div style="background:#ff2d2d;padding:20px;text-align:center">
<h1 style="color:#fff;margin:0">🔱 TRIYAMBAKA ALERT</h1></div>
<div style="padding:24px;background:#111820">
<p>Time: {timestamp}</p><p>Location: {location}</p>
<p>Threat: {threat}</p><p>Confidence: {confidence}%</p>
<p>Description: {description}</p><p>Details: {details}</p>
</div></div></body></html>"""
            requests.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
                json={"from": "TRIYAMBAKA <onboarding@resend.dev>",
                      "to": [to_email],
                      "subject": f"🔱 TRIYAMBAKA ALERT — {threat} Detected",
                      "html": html},
                timeout=15
            )
            add_log(f"Email sent to {to_email}", "safe")
        except Exception as e:
            add_log(f"Email error: {e}", "danger")
    threading.Thread(target=_send, daemon=True).start()

# ── Camera ────────────────────────────────────────────────────────────────────

def camera_thread(source):
    global latest_frame, camera_obj
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        add_log(f"Cannot open camera: {source}", "danger")
        state["camera_active"] = False
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    with camera_lock: camera_obj = cap
    add_log(f"Camera started: {source}", "safe")
    while state["camera_active"]:
        ret, frame = cap.read()
        if not ret: time.sleep(0.05); continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock: latest_frame = buf.tobytes()
    cap.release()
    with camera_lock: camera_obj = None
    add_log("Camera stopped.", "info")


def gen_frames():
    while state["camera_active"]:
        with frame_lock: frame = latest_frame
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(1/30)


def auto_thread(api_key, interval, threshold, settings):
    add_log(f"Auto ON — every {interval}s, threshold {threshold}%", "info")
    while state["auto_mode"] and state["camera_active"]:
        time.sleep(interval)
        if not state["auto_mode"]: break
        with frame_lock: frame = latest_frame
        if not frame: continue
        try:
            result   = call_gemini_vision(api_key, frame)
            state["frames_analyzed"] += 1
            state["latest_result"]   = result
            conf     = result.get("confidence", 0)
            detected = result.get("violence_detected", False)
            add_log(f"Auto: {'⚠ VIOLENCE' if detected else '✓ Clear'} ({conf}%) — {result.get('description','')}",
                    "danger" if detected else "safe")
            if detected and conf >= threshold:
                _fire_alert(result, frame, settings)
        except Exception as e:
            add_log(f"Auto error: {e}", "danger")
    add_log("Auto OFF.", "info")


def _fire_alert(result, snapshot, settings):
    now = time.time()
    if now - state["last_alert_time"] < int(settings.get("cooldown", 30)): return
    state["last_alert_time"]   = now
    state["alerts_triggered"] += 1
    add_log(f"🚨 ALERT #{state['alerts_triggered']} — "
            f"{result.get('threat_level','').upper()} — "
            f"{result.get('confidence',0)}% — {result.get('description','')}", "danger")
    alert_email = settings.get("alert_email", "")
    if settings.get("email_enabled") and alert_email:
        send_email_alert(alert_email, result, settings.get("location", "Camera 01"))

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", has_site_key=bool(SITE_GEMINI_KEY))


@app.route("/video_feed")
def video_feed():
    return Response(stream_with_context(gen_frames()),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/test_key", methods=["POST"])
def api_test_key():
    data = request.get_json() or {}
    key  = (SITE_GEMINI_KEY if data.get("use_site_key") else data.get("api_key", "")).strip()
    if not key:
        return jsonify({"ok": False, "msg": "No API key provided"})
    try:
        url  = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={key}"
        resp = requests.post(url, json={
            "contents": [{"parts": [{"text": "Say OK"}]}],
            "generationConfig": {"maxOutputTokens": 5}
        }, timeout=15)
        if resp.ok:
            return jsonify({"ok": True, "msg": "API key is valid ✓"})
        err = resp.json().get("error", {}).get("message", resp.text)
        return jsonify({"ok": False, "msg": f"Rejected: {err}"})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


@app.route("/api/camera/start", methods=["POST"])
def api_camera_start():
    data   = request.get_json() or {}
    source = data.get("source", 0)
    try: source = int(source)
    except: pass
    if state["camera_active"]:
        return jsonify({"ok": False, "msg": "Camera already running"})
    state["camera_active"] = True
    threading.Thread(target=camera_thread, args=(source,), daemon=True).start()
    time.sleep(1.5)
    return jsonify({"ok": True, "msg": f"Camera started (source={source})"})


@app.route("/api/camera/stop", methods=["POST"])
def api_camera_stop():
    state["camera_active"] = False
    state["auto_mode"]     = False
    return jsonify({"ok": True})


@app.route("/api/analyze/frame", methods=["POST"])
def api_analyze_frame():
    data = request.get_json() or {}
    try:
        key = resolve_api_key(data.get("api_key", ""))
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})

    # Accept frame from browser (base64) or server camera
    frame_b64 = data.get("frame")
    if frame_b64:
        # Browser sent a base64 frame
        try:
            frame = base64.b64decode(frame_b64)
        except Exception as e:
            return jsonify({"ok": False, "msg": f"Invalid frame: {e}"})
    else:
        # Fall back to server camera frame
        with frame_lock: frame = latest_frame
        if not frame:
            return jsonify({"ok": False, "msg": "No frame — start camera first"})

    try:
        result   = call_gemini_vision(key, frame)
        state["frames_analyzed"] += 1
        state["latest_result"]   = result
        conf     = result.get("confidence", 0)
        detected = result.get("violence_detected", False)
        add_log(f"{'⚠ VIOLENCE' if detected else '✓ Clear'} ({conf}%) — {result.get('description','')}",
                "danger" if detected else "safe")
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        add_log(f"Error: {e}", "danger")
        return jsonify({"ok": False, "msg": str(e)})


@app.route("/api/analyze/upload", methods=["POST"])
def api_analyze_upload():
    try:
        key = resolve_api_key(request.form.get("api_key", ""))
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})
    file = request.files.get("file")
    if not file:
        return jsonify({"ok": False, "msg": "No file provided"})
    fname = file.filename or "upload"
    ext   = os.path.splitext(fname)[1].lower()
    fpath = os.path.join(UPLOAD_FOLDER, f"up_{int(time.time())}{ext}")
    file.save(fpath)
    try:
        if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}:
            img_bytes = _extract_frame(fpath)
            add_log(f"Video frame extracted: {fname}", "info")
        else:
            with open(fpath, "rb") as f: raw = f.read()
            img_bytes = resize_image(raw)
            add_log(f"Image loaded: {fname}", "info")

        result   = call_gemini_vision(key, img_bytes)
        state["frames_analyzed"] += 1
        state["latest_result"]   = result
        conf     = result.get("confidence", 0)
        detected = result.get("violence_detected", False)
        add_log(f"Upload: {'⚠ VIOLENCE' if detected else '✓ Clear'} ({conf}%) — {result.get('description','')}",
                "danger" if detected else "safe")
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        add_log(f"Upload error: {e}", "danger")
        return jsonify({"ok": False, "msg": str(e)})
    finally:
        try: os.remove(fpath)
        except: pass


@app.route("/api/auto/start", methods=["POST"])
def api_auto_start():
    data = request.get_json() or {}
    try:
        key = resolve_api_key(data.get("api_key", ""))
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})
    if not state["camera_active"]:
        return jsonify({"ok": False, "msg": "Start camera first"})
    interval  = int(data.get("interval", 5))
    threshold = int(data.get("threshold", 60))
    settings  = data.get("settings", {})
    state["auto_mode"] = True
    threading.Thread(target=auto_thread, args=(key, interval, threshold, settings), daemon=True).start()
    return jsonify({"ok": True, "msg": f"Auto every {interval}s"})


@app.route("/api/auto/stop", methods=["POST"])
def api_auto_stop():
    state["auto_mode"] = False
    return jsonify({"ok": True})


@app.route("/api/alert/email", methods=["POST"])
def api_alert_email():
    data     = request.get_json() or {}
    to_email = data.get("to_email") or data.get("settings", {}).get("alert_email", "")
    location = data.get("location") or data.get("settings", {}).get("location", "Camera 01")
    if not to_email:
        return jsonify({"ok": False, "msg": "No email address provided"})
    result = data.get("result", {
        "violence_detected": True, "confidence": 99, "threat_level": "test",
        "categories": ["test_alert"], "description": "Test alert from TRIYAMBAKA",
        "details": "This is a test email. System working correctly!"
    })
    send_email_alert(to_email, result, location)
    return jsonify({"ok": True, "msg": f"Email sent to {to_email} ✓"})
    # Validate settings


@app.route("/api/status")
def api_status():
    up = int(time.time() - state["session_start"])
    m, s = divmod(up, 60)
    return jsonify({
        "camera_active":    state["camera_active"],
        "auto_mode":        state["auto_mode"],
        "frames_analyzed":  state["frames_analyzed"],
        "alerts_triggered": state["alerts_triggered"],
        "uptime":           f"{m:02d}:{s:02d}",
        "log_entries":      state["log_entries"][-60:],
        "latest_result":    state["latest_result"],
        "has_site_key":     bool(SITE_GEMINI_KEY),
    })


def _extract_frame(video_path):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(fps * 5), int(total * 0.3)))
    ret, frame = cap.read()
    cap.release()
    if not ret: raise ValueError("Could not read frame")
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return resize_image(buf.tobytes())


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    add_log("TRIYAMBAKA started.", "safe")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
