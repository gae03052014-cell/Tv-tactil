#!/usr/bin/env python3
"""
Télé tactile : programme du PC.

- La webcam du PC (face à la télé) donne la POSITION du doigt.
- La caméra du téléphone (au ras du bas de la télé) donne le CONTACT.
- Un détecteur d'écriture force l'appui quand le contact est raté, et rejoue
  ce qui a été tracé juste avant sa détection.

Lancement :  python server.py
"""
import argparse
import asyncio
import collections
import ctypes
import datetime
import ipaddress
import json
import math
import os
import socket
import ssl
import sys
import threading
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from aiohttp import web

try:
    from scipy.interpolate import RBFInterpolator
except Exception:
    RBFInterpolator = None

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()

HERE = Path(__file__).resolve().parent
CALIB_FILE = HERE / "calibration.json"
OVERLAY = "TV"
PREVIEW = "Apercu"
DEBUG = "Debug tactile"
DWELL = 0.7            # secondes à garder le doigt immobile sur un point de calibration
MAGNET_RADIUS = 48     # pixels écran : rayon de recherche d'un bouton proche
SPLINE_SMOOTH = 1e-3


# ----------------------------------------------------------------------------
# État partagé entre les threads
# ----------------------------------------------------------------------------
class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop = False
        self.status = "En attente"
        # téléphone
        self.phone_frame = None
        self.phone_ts = 0.0
        self.phone_event = threading.Event()
        self.phone_clients = 0
        self.phone_rot = 0
        self.phone_mirror = False
        self.phone_tip = None
        self.phone_tip_ts = 0.0
        self.phone_vis = None
        self.height = None
        self.surface = None        # {"p0": (x, y), "v": (vx, vy), "sign": ±1}
        self.thr_on = None
        self.thr_off = None
        self.contact = False
        self.contact_ts = 0.0
        # webcam PC
        self.cam_tip = None
        self.cam_ts = 0.0
        self.cam_vis = None
        self.cam_size = (1280, 720)


S = Shared()


# ----------------------------------------------------------------------------
# Utilitaires d'affichage (OpenCV ne gère pas les accents)
# ----------------------------------------------------------------------------
def _ascii(t):
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()


def put(img, text, org, scale=0.7, color=(255, 255, 255), th=2):
    cv2.putText(img, _ascii(text), org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def put_center(img, text, y, scale=1.0, color=(255, 255, 255), th=2):
    text = _ascii(text)
    (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    cv2.putText(img, text, ((img.shape[1] - w) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# Détection de la main (MediaPipe)
# ----------------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def make_hands():
    return mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1,
                          min_detection_confidence=0.5, min_tracking_confidence=0.5)


def find_tip(hands, bgr):
    res = hands.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return None, None
    lm = res.multi_hand_landmarks[0]
    h, w = bgr.shape[:2]
    t = lm.landmark[8]            # bout de l'index
    return (t.x * w, t.y * h), lm


def orient(img):
    r = S.phone_rot
    if r == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif r == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif r == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if S.phone_mirror:
        img = cv2.flip(img, 1)
    return img


def height_of(tip):
    """Distance signée (px) du bout du doigt à la ligne de la surface de l'écran."""
    sf = S.surface
    if tip is None or sf is None:
        return None
    x0, y0 = sf["p0"]
    vx, vy = sf["v"]
    return sf["sign"] * ((tip[0] - x0) * vy - (tip[1] - y0) * vx)


def webcam_loop(index, width):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, width * 9 // 16)
    hands = make_hands()
    while not S.stop:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        tip, lm = find_tip(hands, frame)
        vis = frame.copy()
        if lm is not None:
            mp_draw.draw_landmarks(vis, lm, mp_hands.HAND_CONNECTIONS)
            cv2.circle(vis, (int(tip[0]), int(tip[1])), 9, (0, 255, 255), -1)
        S.cam_size = (frame.shape[1], frame.shape[0])
        S.cam_tip, S.cam_ts, S.cam_vis = tip, time.time(), vis
    cap.release()


def phone_loop():
    hands = make_hands()
    miss = 0
    on_cnt = 0
    while not S.stop:
        if not S.phone_event.wait(0.2):
            continue
        S.phone_event.clear()
        with S.lock:
            fr = S.phone_frame
        if fr is None:
            continue
        img = orient(fr)
        tip, lm = find_tip(hands, img)
        S.phone_tip, S.phone_tip_ts = tip, time.time()
        h = height_of(tip)
        S.height = h

        if S.thr_on is not None:
            if h is None:
                miss += 1
                on_cnt = 0
                if miss >= 3:
                    S.contact = False
            else:
                miss = 0
                if not S.contact:
                    on_cnt = on_cnt + 1 if h <= S.thr_on else 0
                    if on_cnt >= 2:
                        S.contact = True
                elif h >= S.thr_off:
                    S.contact = False
                    on_cnt = 0
        S.contact_ts = time.time()

        vis = img.copy()
        if lm is not None:
            mp_draw.draw_landmarks(vis, lm, mp_hands.HAND_CONNECTIONS)
        sf = S.surface
        if sf is not None:
            (x0, y0), (vx, vy) = sf["p0"], sf["v"]
            cv2.line(vis, (int(x0 - vx * 3000), int(y0 - vy * 3000)),
                     (int(x0 + vx * 3000), int(y0 + vy * 3000)), (0, 200, 255), 2)
        if tip is not None:
            col = (0, 255, 0) if S.contact else (0, 255, 255)
            cv2.circle(vis, (int(tip[0]), int(tip[1])), 8, col, -1)
            if h is not None:
                put(vis, f"h={h:.0f}px {'CONTACT' if S.contact else ''}", (10, 30), 0.8, col)
        S.phone_vis = vis


# ----------------------------------------------------------------------------
# Serveur HTTPS / WebSocket (reçoit les images du téléphone)
# ----------------------------------------------------------------------------
def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_cert(ip):
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    d = HERE / "certs"
    d.mkdir(exist_ok=True)
    crt, key = d / f"cert_{ip}.pem", d / f"key_{ip}.pem"
    if crt.exists() and key.exists():
        return crt, key
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, ip)])
    san = x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(ip)), x509.DNSName("localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(k.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=825))
            .add_extension(san, critical=False).sign(k, hashes.SHA256()))
    crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key.write_bytes(k.private_bytes(serialization.Encoding.PEM,
                                    serialization.PrivateFormat.TraditionalOpenSSL,
                                    serialization.NoEncryption()))
    return crt, key


async def index_handler(request):
    return web.FileResponse(HERE / "phone" / "index.html")


async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024, heartbeat=15)
    await ws.prepare(request)
    S.phone_clients += 1

    async def pusher():
        try:
            while not ws.closed:
                await ws.send_json({"type": "status", "text": S.status})
                await asyncio.sleep(0.5)
        except Exception:
            pass

    task = asyncio.ensure_future(pusher())
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.BINARY:
                img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    with S.lock:
                        S.phone_frame, S.phone_ts = img, time.time()
                    S.phone_event.set()
    finally:
        task.cancel()
        S.phone_clients -= 1
    return ws


def server_thread(port, ctx):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", ws_handler)

    async def start():
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", port, ssl_context=ctx).start()

    loop.run_until_complete(start())
    loop.run_forever()


# ----------------------------------------------------------------------------
# Filtre One Euro, correspondance webcam -> écran
# ----------------------------------------------------------------------------
class OneEuro:
    def __init__(self, mincut=1.5, beta=0.02, dcut=1.0):
        self.mincut, self.beta, self.dcut = mincut, beta, dcut
        self.t = None

    @staticmethod
    def _alpha(cut, dt):
        tau = 1.0 / (2 * math.pi * cut)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.t = None

    def __call__(self, x, t):
        if self.t is None:
            self.t, self.x, self.dx = t, x, 0.0
            return x
        dt = max(t - self.t, 1e-3)
        ad = self._alpha(self.dcut, dt)
        self.dx = ad * ((x - self.x) / dt) + (1 - ad) * self.dx
        a = self._alpha(self.mincut + self.beta * abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


class Mapper:
    """Homographie webcam -> écran, plus correction des bords par spline (écran incurvé)."""

    def __init__(self, cam_pts, scr_pts, cam_size):
        self.cam = np.float32(cam_pts)
        self.scr = np.float32(scr_pts)
        self.sc = np.float32(cam_size)
        self.H, _ = cv2.findHomography(self.cam, self.scr, cv2.RANSAC, 0.03 * SCREEN_W)
        base = cv2.perspectiveTransform(self.cam.reshape(-1, 1, 2), self.H).reshape(-1, 2)
        res = self.scr - base
        self.rms = float(np.sqrt((res ** 2).sum(1).mean()))
        self.rbf = None
        if RBFInterpolator is not None and len(self.cam) >= 6:
            try:
                self.rbf = RBFInterpolator(self.cam / self.sc, res, kernel="thin_plate_spline",
                                           smoothing=SPLINE_SMOOTH)
            except Exception:
                self.rbf = None

    def __call__(self, p):
        q = cv2.perspectiveTransform(np.float32([[p]]), self.H)[0, 0]
        if self.rbf is not None:
            corr = self.rbf(np.float32([p]) / self.sc)[0]
            n = float(np.hypot(*corr))
            lim = 0.05 * SCREEN_W
            if n > lim:
                corr = corr * lim / n
            q = q + corr
        return float(np.clip(q[0], 0, SCREEN_W - 1)), float(np.clip(q[1], 0, SCREEN_H - 1))


# ----------------------------------------------------------------------------
# « IA » d'écriture : repère les traits courts et répétés d'une main qui écrit
# ----------------------------------------------------------------------------
class WritingDetector:
    WIN = 1.4

    def __init__(self):
        self.pts = collections.deque()
        self.writing = False
        self.last_active = 0.0

    def _looks_like_writing(self):
        a = np.array([[x, y] for _, x, y in self.pts])
        seg = np.diff(a, axis=0)
        ln = np.hypot(seg[:, 0], seg[:, 1])
        path = ln.sum()
        diag = math.hypot(*(a.max(0) - a.min(0)))
        if diag < 0.01 * SCREEN_W or diag > 0.30 * SCREEN_W:
            return False
        if path / max(diag, 1.0) < 1.8:
            return False
        v = seg[ln > 0.004 * SCREEN_W]
        if len(v) < 6:
            return False
        ang = np.arctan2(v[:, 1], v[:, 0])
        d = np.abs(np.angle(np.exp(1j * np.diff(ang))))
        return int((d > math.radians(55)).sum()) >= 4

    def update(self, t, pos, near):
        """Retourne (écrit, trace_à_rejouer ou None)."""
        if pos is None or not near:
            if self.pts and t - self.pts[-1][0] > 0.5:
                self.pts.clear()
            if self.writing and t - self.last_active > 0.6:
                self.writing = False
            return self.writing, None
        self.pts.append((t, pos[0], pos[1]))
        while self.pts and t - self.pts[0][0] > self.WIN:
            self.pts.popleft()
        replay = None
        if len(self.pts) >= 12 and self.pts[-1][0] - self.pts[0][0] >= 0.6 and self._looks_like_writing():
            self.last_active = t
            if not self.writing:
                self.writing = True
                replay = [(x, y) for _, x, y in self.pts]
        if self.writing and t - self.last_active > 0.6:
            self.writing = False
        return self.writing, replay


# ----------------------------------------------------------------------------
# Souris et magnétisme
# ----------------------------------------------------------------------------
CLICKABLE = {"ButtonControl", "HyperlinkControl", "MenuItemControl", "ListItemControl", "TabItemControl",
             "CheckBoxControl", "RadioButtonControl", "EditControl", "TreeItemControl", "ComboBoxControl",
             "SplitButtonControl", "DataItemControl"}


def snap(x, y, radius):
    """Si aucun bouton sous le doigt, cherche le plus proche (Windows, module uiautomation)."""
    try:
        import uiautomation as auto
    except Exception:
        return x, y
    t0 = time.time()

    def ok(px, py):
        try:
            c = auto.ControlFromPoint(int(px), int(py))
            return c is not None and c.ControlTypeName in CLICKABLE
        except Exception:
            return False

    if ok(x, y):
        return x, y
    for r in (radius / 3, 2 * radius / 3, radius):
        for k in range(8):
            if time.time() - t0 > 0.25:
                return x, y
            a = k * math.pi / 4
            px, py = x + r * math.cos(a), y + r * math.sin(a)
            if ok(px, py):
                return px, py
    return x, y


class Mouse:
    def __init__(self):
        self.down = False

    def move(self, x, y):
        pyautogui.moveTo(int(x), int(y), _pause=False)

    def press(self):
        if not self.down:
            pyautogui.mouseDown(_pause=False)
            self.down = True

    def release(self):
        if self.down:
            pyautogui.mouseUp(_pause=False)
            self.down = False

    def replay(self, pts):
        self.move(*pts[0])
        self.press()
        for p in pts[1:]:
            self.move(*p)
            time.sleep(0.003)


def key_down(vk):
    if os.name != "nt":
        return False
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


# ----------------------------------------------------------------------------
# Fenêtres : aperçu, calibration
# ----------------------------------------------------------------------------
def _fit(img, h=360):
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))


def _blank(text, w=640, h=360):
    img = np.zeros((h, w, 3), np.uint8)
    put_center(img, text, h // 2, 0.8)
    return img


def compose_preview(extra=""):
    now = time.time()
    ph = S.phone_vis if (S.phone_vis is not None and now - S.phone_ts < 1.5) else None
    left = _fit(ph) if ph is not None else _blank("Telephone : pas de signal")
    right = _fit(S.cam_vis) if S.cam_vis is not None else _blank("Webcam PC...")
    row = np.hstack([left, right])
    bar = np.zeros((64, row.shape[1], 3), np.uint8)
    put(bar, "Entree: calibrer | L: charger la calibration | R: pivoter le telephone | M: miroir | P: sans telephone | Q: quitter",
        (10, 24), 0.5)
    put(bar, extra, (10, 50), 0.6, (0, 255, 255))
    return np.vstack([row, bar])


def preview_phase():
    cv2.namedWindow(PREVIEW)
    warn = ""
    while True:
        extra = warn or (f"Telephone connecte ({S.phone_clients}) - place-le pour que le doigt touchant l'ecran soit visible"
                 if S.phone_clients else "En attente du telephone...")
        S.status = "Placement du telephone"
        cv2.imshow(PREVIEW, compose_preview(extra))
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10):
            if S.phone_clients:
                cv2.destroyWindow(PREVIEW)
                return "calibrate"
            warn = "Connecte d'abord le telephone (ou appuie sur P pour continuer sans)"
        elif k in (ord("p"), ord("P")):
            cv2.destroyWindow(PREVIEW)
            return "calibrate"
        elif k in (ord("l"), ord("L")) and CALIB_FILE.exists():
            cv2.destroyWindow(PREVIEW)
            return "load"
        elif k in (ord("r"), ord("R")):
            S.phone_rot = (S.phone_rot + 90) % 360
        elif k in (ord("m"), ord("M")):
            S.phone_mirror = not S.phone_mirror
        elif k in (ord("q"), ord("Q"), 27):
            return None


def setup_overlay():
    cv2.namedWindow(OVERLAY, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(OVERLAY, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def canvas():
    return np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)


def calibrate_points(use_phone):
    xs, ys = [0.08, 0.36, 0.64, 0.92], [0.12, 0.5, 0.88]
    targets = []
    for j, fy in enumerate(ys):
        row = [(fx, fy) for fx in xs]
        targets += row[::-1] if j % 2 else row
    cam_pts, scr_pts, ph_pts = [], [], []
    last_cap = None
    for i, (fx, fy) in enumerate(targets):
        tx, ty = int(fx * SCREEN_W), int(fy * SCREEN_H)
        hist, since = collections.deque(), None
        S.status = f"Calibration des points {i + 1}/{len(targets)}"
        while True:
            now = time.time()
            tip = S.cam_tip if now - S.cam_ts < 0.3 else None
            ptip = S.phone_tip if (use_phone and now - S.phone_tip_ts < 0.3) else None
            hist.append((now, tip, ptip))
            while hist and now - hist[0][0] > 0.5:
                hist.popleft()
            stable = False
            if hist and now - hist[0][0] >= 0.4 and all(h[1] is not None for h in hist):
                a = np.array([h[1] for h in hist])
                med = np.median(a, 0)
                camw = S.cam_size[0]
                far = last_cap is None or math.hypot(*(med - last_cap)) > 0.04 * camw
                stable = bool(np.max(np.hypot(*(a - med).T)) < 0.008 * camw and far)
            since = (since or now) if stable else None
            prog = 0.0 if since is None else min(1.0, (now - since) / DWELL)

            c = canvas()
            for (ox, oy) in targets[:i]:
                cv2.circle(c, (int(ox * SCREEN_W), int(oy * SCREEN_H)), 8, (70, 110, 70), -1)
            cv2.circle(c, (tx, ty), 26, (255, 255, 255), 3)
            cv2.circle(c, (tx, ty), 6, (255, 255, 255), -1)
            if prog > 0:
                cv2.ellipse(c, (tx, ty), (40, 40), -90, 0, 360 * prog, (90, 200, 255), 6)
            put_center(c, f"Point {i + 1} / {len(targets)}", SCREEN_H // 2 - 20, 1.2)
            put_center(c, "Touche le cercle avec l'index et garde-le immobile", SCREEN_H // 2 + 30, 0.9, (190, 190, 190))
            put_center(c, "Echap pour annuler", SCREEN_H - 30, 0.6, (120, 120, 120))
            cv2.imshow(OVERLAY, c)
            if cv2.waitKey(10) & 0xFF == 27:
                raise KeyboardInterrupt

            if prog >= 1.0:
                med = np.median(np.array([h[1] for h in hist]), 0)
                cam_pts.append(med.tolist())
                scr_pts.append([tx, ty])
                pt = [h[2] for h in hist if h[2] is not None]
                ph_pts.append(np.median(np.array(pt), 0).tolist() if len(pt) >= 3 else None)
                last_cap = med
                cv2.circle(c, (tx, ty), 40, (80, 220, 120), -1)
                cv2.imshow(OVERLAY, c)
                cv2.waitKey(250)
                break

    valid = [p for p in ph_pts if p is not None]
    if len(valid) >= 4:
        vx, vy, x0, y0 = cv2.fitLine(np.float32(valid), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        S.surface = {"p0": (float(x0), float(y0)), "v": (float(vx), float(vy)), "sign": 1}
    else:
        S.surface = None
    mapper = Mapper(cam_pts, scr_pts, S.cam_size)
    print(f"Calibration des points terminée : erreur moyenne {mapper.rms:.1f} px"
          f"{' (correction spline active)' if mapper.rbf is not None else ''}")
    return mapper, cam_pts, scr_pts


def collect_heights(title, sub, secs=6.0):
    t_prep = time.time() + 3.0
    hs = []
    while True:
        now = time.time()
        c = canvas()
        put_center(c, title, SCREEN_H // 2 - 60, 1.6, (90, 200, 255), 3)
        put_center(c, sub, SCREEN_H // 2, 0.95)
        if now < t_prep:
            put_center(c, str(int(math.ceil(t_prep - now))), SCREEN_H // 2 + 120, 2.5)
        else:
            rest = secs - (now - t_prep)
            if rest <= 0:
                break
            h = height_of(S.phone_tip) if now - S.phone_tip_ts < 0.3 else None
            if h is not None:
                hs.append(h)
            w = int(SCREEN_W * 0.5 * (1 - rest / secs))
            cv2.rectangle(c, (SCREEN_W // 4, SCREEN_H // 2 + 100), (SCREEN_W // 4 + w, SCREEN_H // 2 + 116), (90, 200, 255), -1)
            put_center(c, f"Enregistrement... {len(hs)} mesures", SCREEN_H // 2 + 160, 0.7, (190, 190, 190))
        cv2.imshow(OVERLAY, c)
        if cv2.waitKey(15) & 0xFF == 27:
            raise KeyboardInterrupt
    return np.array(hs)


def calibrate_contact():
    S.status = "Calibration du contact"
    while True:
        touch = collect_heights("DOIGT SUR L'ECRAN", "Pose l'index sur la télé et fais-le glisser partout")
        hover = collect_heights("DOIGT EN L'AIR", "Éloigne l'index de 3 à 5 cm de la télé et bouge-le")
        if len(touch) >= 15 and len(hover) >= 15:
            break
        c = canvas()
        put_center(c, "Le telephone ne voit pas assez le doigt. On recommence.", SCREEN_H // 2, 1.0, (120, 120, 255))
        cv2.imshow(OVERLAY, c)
        cv2.waitKey(2000)
    if hover.mean() < touch.mean():
        S.surface["sign"] = -1
        touch, hover = -touch, -hover
    lo, hi = float(np.percentile(touch, 85)), float(np.percentile(hover, 10))
    gap = hi - lo
    if gap < 4:
        print("Attention : contact et non-contact se distinguent mal sur la caméra du téléphone. "
              "Rapproche le téléphone de l'écran ou améliore l'éclairage.")
        S.thr_on = float(np.median(touch)) + 3
        S.thr_off = S.thr_on + 3
    else:
        S.thr_on, S.thr_off = lo + 0.35 * gap, lo + 0.6 * gap
    print(f"Seuils de contact : appui sous {S.thr_on:.1f} px, relâchement au-dessus de {S.thr_off:.1f} px")


def save_calibration(cam_pts, scr_pts):
    data = {"screen": [SCREEN_W, SCREEN_H], "cam_size": list(S.cam_size), "cam": cam_pts, "scr": scr_pts,
            "surface": S.surface, "thr_on": S.thr_on, "thr_off": S.thr_off}
    CALIB_FILE.write_text(json.dumps(data), encoding="utf-8")


def load_calibration():
    d = json.loads(CALIB_FILE.read_text(encoding="utf-8"))
    if d["screen"] != [SCREEN_W, SCREEN_H]:
        print("Attention : la résolution d'écran a changé depuis la calibration.")
    sf = d.get("surface")
    if sf:
        sf = {"p0": tuple(sf["p0"]), "v": tuple(sf["v"]), "sign": sf["sign"]}
    S.surface, S.thr_on, S.thr_off = sf, d.get("thr_on"), d.get("thr_off")
    return Mapper(d["cam"], d["scr"], d["cam_size"])


# ----------------------------------------------------------------------------
# Mode tactile
# ----------------------------------------------------------------------------
def run_loop(mapper, args):
    try:
        cv2.destroyWindow(OVERLAY)
    except Exception:
        pass
    fx, fy = OneEuro(), OneEuro()
    mouse, wd = Mouse(), WritingDetector()
    paused = False
    f8_prev = False
    last_dbg = 0.0
    S.status = "Mode tactile actif"
    print("Mode tactile actif. F8 : pause / reprise, F9 ou Q (fenêtre debug) : quitter.")
    if not args.no_debug:
        cv2.namedWindow(DEBUG)
    while not S.stop:
        now = time.time()
        f8 = key_down(0x77)
        if f8 and not f8_prev:
            paused = not paused
            S.status = "Pause" if paused else "Mode tactile actif"
            mouse.release()
        f8_prev = f8
        if key_down(0x78):
            break

        tip = S.cam_tip if now - S.cam_ts < 0.25 else None
        pos = None
        if tip is not None:
            x, y = mapper(tip)
            pos = (fx(x, now), fy(y, now))
        else:
            fx.reset()
            fy.reset()

        phone_ok = S.phone_clients > 0 and now - S.phone_tip_ts < 0.3
        contact = bool(S.contact and now - S.contact_ts < 0.3 and phone_ok)
        h = S.height if phone_ok else None
        thr = max(S.thr_off or 5.0, 5.0)
        near = h is None or h <= 3 * thr          # la main est clairement loin de l'écran : on ignore l'écriture
        writing, replay = wd.update(now, pos, near)

        want = (contact or writing) and pos is not None and not paused
        if want and not mouse.down:
            if writing and not contact and replay:
                mouse.replay(replay)
            else:
                sx, sy = pos
                if not args.no_magnet:
                    sx, sy = snap(sx, sy, MAGNET_RADIUS)
                mouse.move(sx, sy)
                mouse.press()
        elif want and mouse.down:
            mouse.move(*pos)
        elif not want and mouse.down:
            mouse.release()
        elif pos is not None and not paused and not args.no_hover:
            mouse.move(*pos)

        if not args.no_debug and now - last_dbg > 0.04:
            last_dbg = now
            if S.cam_vis is not None:
                v = _fit(S.cam_vis, 270)
                label = "PAUSE" if paused else ("APPUI" if mouse.down else "survol")
                put(v, f"{label}  contact={int(contact)} ecriture={int(writing)}", (8, 24), 0.55, (0, 255, 255))
                cv2.imshow(DEBUG, v)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
        else:
            time.sleep(0.005)
    mouse.release()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Télé tactile : serveur PC")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--cam", type=int, default=0, help="index de la webcam du PC")
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--no-magnet", action="store_true", help="désactive le magnétisme vers les boutons")
    ap.add_argument("--no-hover", action="store_true", help="le curseur ne suit pas le doigt hors contact")
    ap.add_argument("--no-debug", action="store_true", help="pas de fenêtre de debug en mode tactile")
    args = ap.parse_args()

    ip = lan_ip()
    crt, key = ensure_cert(ip)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(crt, key)
    threading.Thread(target=server_thread, args=(args.port, ctx), daemon=True).start()
    threading.Thread(target=webcam_loop, args=(args.cam, args.cam_width), daemon=True).start()
    threading.Thread(target=phone_loop, daemon=True).start()

    print("\n  Serveur prêt.")
    print(f"  Adresse IP à saisir sur le téléphone : {ip}   (port {args.port})")
    print(f"  Page de secours / certificat : https://{ip}:{args.port}/\n")

    try:
        choice = preview_phase()
        if choice is None:
            return
        setup_overlay()
        if choice == "load":
            mapper = load_calibration()
        else:
            mapper, cam_pts, scr_pts = calibrate_points(S.phone_clients > 0)
            if S.surface is not None:
                calibrate_contact()
            save_calibration(cam_pts, scr_pts)
        run_loop(mapper, args)
    except KeyboardInterrupt:
        pass
    finally:
        S.stop = True
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
