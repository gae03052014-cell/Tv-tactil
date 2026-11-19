"""
Logique de la télé tactile, sans dépendance à la caméra ni à la souris
(numpy uniquement) : réglages, lissage, gestes, détection de frappe et
d'écriture, modèles appris. Testable seul : python tests/test_logic.py
"""
import collections
import json
import math
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# Réglages (modifiables en direct depuis la page /settings)
# ----------------------------------------------------------------------------
def _n(key, label, default, lo, hi, step):
    return {"key": key, "label": label, "type": "num", "default": default, "min": lo, "max": hi, "step": step}


def _b(key, label, default):
    return {"key": key, "label": label, "type": "bool", "default": default}


def _c(key, label, default, choices):
    return {"key": key, "label": label, "type": "choice", "default": default, "choices": choices}


SCHEMA = [
    _n("smooth_mincut", "Lissage (plus bas = plus lisse)", 1.5, 0.5, 6, 0.1),
    _n("smooth_beta", "Réactivité du lissage", 0.02, 0, 0.1, 0.005),
    _n("contact_sens", "Sensibilité du contact", 1.0, 0.5, 1.6, 0.05),
    _n("contact_lead", "Anticipation du contact (s)", 0.06, 0, 0.15, 0.01),
    _n("drag_px", "Distance avant glisser (px)", 14, 6, 40, 1),
    _n("long_press", "Appui long = clic droit (s)", 0.8, 0.4, 1.5, 0.05),
    _n("scroll_gain", "Vitesse de défilement", 1.0, 0.2, 4, 0.1),
    _n("zoom_gain", "Vitesse de zoom", 2.0, 0.5, 6, 0.1),
    _n("magnet_radius", "Rayon du magnétisme (px)", 48, 0, 120, 4),
    _n("arm_ratio", "Seuil bras / paume (largeur de l'image)", 0.18, 0.08, 0.5, 0.01),
    _n("dark_thr", "Seuil d'image sombre", 90, 40, 160, 5),
    _n("type_dwell", "Pause minimale d'une frappe (s)", 0.07, 0.03, 0.2, 0.01),
    _b("magnet", "Magnétisme vers les boutons", True),
    _b("hover", "Le curseur suit le doigt", True),
    _b("palm_reject", "Rejeter la paume", True),
    _b("arm_reject", "Rejeter le bras", True),
    _b("lowlight", "Amélioration en faible lumière", True),
    _b("fusion", "Fusion position téléphone + webcam", True),
    _b("assist_stroke", "Aide à l'écriture au doigt (tracé)", True),
    _b("assist_typing", "Aide à la frappe (clavier à l'écran)", True),
    _b("learn", "Apprentissage sur mes gestes (données locales)", False),
    _c("hand", "Main utilisée", "any", ["any", "right", "left"]),
    _b("swap_hand", "Inverser gauche / droite", False),
]
CFG = {s["key"]: s["default"] for s in SCHEMA}
_BY_KEY = {s["key"]: s for s in SCHEMA}


def coerce(key, value):
    s = _BY_KEY[key]
    if s["type"] == "bool":
        return bool(value)
    if s["type"] == "choice":
        return value if value in s["choices"] else s["default"]
    return float(min(max(float(value), s["min"]), s["max"]))


def load_cfg(path):
    try:
        for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items():
            if k in _BY_KEY:
                CFG[k] = coerce(k, v)
    except Exception:
        pass


def save_cfg(path):
    try:
        Path(path).write_text(json.dumps(CFG, indent=1), encoding="utf-8")
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Lissage One Euro
# ----------------------------------------------------------------------------
class OneEuro:
    def __init__(self, mincut=1.5, beta=0.02, dcut=1.0):
        self.mincut, self.beta, self.dcut = mincut, beta, dcut
        self.t = None

    @staticmethod
    def _alpha(cut, dt):
        return 1.0 / (1.0 + (1.0 / (2 * math.pi * cut)) / dt)

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


# ----------------------------------------------------------------------------
# Régression logistique minuscule (contact vu par la webcam, apprentissage)
# ----------------------------------------------------------------------------
class Logistic:
    def __init__(self):
        self.w = self.mu = self.sd = None

    def fit(self, X, y, iters=500, lr=0.3, l2=1e-2):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        n1 = max(y.sum(), 1.0)
        n0 = max(len(y) - y.sum(), 1.0)
        sw = np.where(y > 0.5, len(y) / (2 * n1), len(y) / (2 * n0))
        w = np.zeros(Z.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
            g = Z.T @ ((p - y) * sw) / len(Z) + l2 * np.r_[w[:-1], 0.0]
            w -= lr * g
        self.w = w
        return self

    def predict(self, x):
        z = (np.asarray(x, float) - self.mu) / self.sd
        return float(1 / (1 + math.exp(-float(np.clip(np.r_[z, 1.0] @ self.w, -30, 30)))))

    def accuracy(self, X, y):
        p = np.array([self.predict(x) for x in X])
        return float(((p > 0.5) == (np.asarray(y) > 0.5)).mean())

    def to_dict(self):
        return {"w": self.w.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @staticmethod
    def from_dict(d):
        m = Logistic()
        m.w, m.mu, m.sd = np.array(d["w"]), np.array(d["mu"]), np.array(d["sd"])
        return m


# ----------------------------------------------------------------------------
# Caractéristiques d'une trajectoire (pour l'apprentissage)
# ----------------------------------------------------------------------------
def _turns(seg, ln, W, deg=55):
    v = seg[ln > 0.003 * W]
    if len(v) < 3:
        return 0
    ang = np.arctan2(v[:, 1], v[:, 0])
    d = np.abs(np.angle(np.exp(1j * np.diff(ang))))
    return int((d > math.radians(deg)).sum())


def traj_features(pts, W):
    """pts : liste de (t, x, y) sur environ 1 seconde. Retourne un vecteur ou None."""
    if len(pts) < 8:
        return None
    t = np.array([p[0] for p in pts])
    a = np.array([[p[1], p[2]] for p in pts])
    seg = np.diff(a, axis=0)
    dt = np.maximum(np.diff(t), 1e-3)
    ln = np.hypot(seg[:, 0], seg[:, 1])
    sp = ln / dt
    path = ln.sum()
    diag = math.hypot(*(a.max(0) - a.min(0)))
    return np.array([path / W, diag / W, min(path / max(diag, 1.0), 20.0), sp.mean() / W, sp.std() / W,
                     min(_turns(seg, ln, W), 20), float((sp < 0.05 * W).mean()),
                     np.abs(seg[:, 1]).sum() / max(path, 1.0)])


class Learner:
    """Enregistre localement des trajectoires étiquetées et entraîne un petit modèle."""

    def __init__(self, dirpath):
        self.dir = Path(dirpath)
        self.X, self.y, self.model, self.last_t, self.new = [], [], None, 0.0, 0
        try:
            if (self.dir / "samples.npz").exists():
                d = np.load(self.dir / "samples.npz")
                self.X, self.y = list(d["X"]), list(d["y"])
            if (self.dir / "model.json").exists():
                self.model = Logistic.from_dict(json.loads((self.dir / "model.json").read_text()))
        except Exception:
            self.X, self.y, self.model = [], [], None

    def add(self, now, feat, label):
        if feat is None or label is None or now - self.last_t < 0.16:
            return
        self.last_t = now
        self.X.append(np.asarray(feat, float))
        self.y.append(float(label))
        self.new += 1
        if len(self.X) > 20000:
            self.X, self.y = self.X[-20000:], self.y[-20000:]

    def predict(self, feat):
        if self.model is None or feat is None:
            return None
        return self.model.predict(feat)

    def train(self):
        """Retourne la précision sur 20 % de données mises de côté, ou None."""
        y = np.array(self.y)
        if len(y) < 300 or (y > 0.5).sum() < 120 or (y < 0.5).sum() < 120:
            return None
        X = np.array(self.X)
        idx = np.random.default_rng(0).permutation(len(y))
        cut = int(0.8 * len(y))
        tr, te = idx[:cut], idx[cut:]
        m = Logistic().fit(X[tr], y[tr])
        acc = m.accuracy(X[te], y[te])
        self.new = 0
        if acc >= 0.8:
            self.model = Logistic().fit(X, y)
            self.save()
        return acc

    def save(self):
        try:
            self.dir.mkdir(exist_ok=True)
            if self.X:
                np.savez(self.dir / "samples.npz", X=np.array(self.X), y=np.array(self.y))
            if self.model is not None:
                (self.dir / "model.json").write_text(json.dumps(self.model.to_dict()))
        except Exception:
            pass

    def forget(self):
        self.X, self.y, self.model, self.new = [], [], None, 0
        for n in ("samples.npz", "model.json"):
            try:
                (self.dir / n).unlink()
            except Exception:
                pass


# ----------------------------------------------------------------------------
# Écriture au doigt (tracé) : petits traits répétés dans une zone réduite
# ----------------------------------------------------------------------------
class WritingDetector:
    WIN = 1.4

    def __init__(self, W):
        self.W = W
        self.pts = collections.deque()
        self.writing = False
        self.last_active = 0.0

    def reset(self):
        self.pts.clear()
        self.writing = False

    def _rule(self):
        a = np.array([[x, y] for _, x, y in self.pts])
        seg = np.diff(a, axis=0)
        ln = np.hypot(seg[:, 0], seg[:, 1])
        path = ln.sum()
        diag = math.hypot(*(a.max(0) - a.min(0)))
        if diag < 0.01 * self.W or diag > 0.30 * self.W or path / max(diag, 1.0) < 1.8:
            return 0.0
        return min(1.0, _turns(seg, ln, self.W) / 4.0)

    def update(self, t, pos, near, p_learn=None):
        """Retourne (écrit, trace à rejouer ou None)."""
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
        if len(self.pts) >= 12 and self.pts[-1][0] - self.pts[0][0] >= 0.6:
            rule = self._rule()
            if p_learn is None:
                ok = rule >= 1.0
            else:
                ok = 0.4 * rule + 0.6 * p_learn >= 0.7
            if ok:
                self.last_active = t
                if not self.writing:
                    self.writing = True
                    replay = [(x, y) for _, x, y in self.pts]
        if self.writing and t - self.last_active > 0.6:
            self.writing = False
        return self.writing, replay


# ----------------------------------------------------------------------------
# Frappe sur un clavier à l'écran : pauses brèves entre déplacements rapides
# ----------------------------------------------------------------------------
class TypingDetector:
    V_HI, V_LO = 0.12, 0.05          # fractions de la largeur d'écran par seconde

    def __init__(self, W):
        self.W = W
        self.speed = 0.0
        self.last = None
        self.armed = False
        self.dw_start = None
        self.dw_pts = []
        self.events = collections.deque()
        self.active = False
        self.last_event = -9.0

    def reset(self):
        self.last = None
        self.armed = False
        self.dw_start = None
        self.events.clear()
        self.active = False

    def _pattern(self, now):
        rec = [e for e in self.events if now - e["t"] < 2.0]
        if len(rec) < 3:
            return False
        good = 0
        for a, b in zip(rec, rec[1:]):
            dt = b["t"] - a["t"]
            d = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
            if 0.08 <= dt <= 0.9 and 0.012 * self.W <= d <= 0.4 * self.W:
                good += 1
        xs = [e["x"] for e in rec]
        return good >= 2 and (max(xs) - min(xs)) > 0.03 * self.W

    def update(self, now, pos, near, clicks):
        """Retourne la liste des appuis (x, y) à forcer, en rattrapant les touches ratées."""
        if pos is None or not near:
            self.last, self.dw_start, self.armed = None, None, False
        else:
            if self.last is not None:
                dt = max(now - self.last[0], 1e-3)
                v = math.hypot(pos[0] - self.last[1], pos[1] - self.last[2]) / dt
                self.speed = 0.5 * self.speed + 0.5 * v
            self.last = (now, pos[0], pos[1])
            if self.speed > self.V_HI * self.W:
                self.armed, self.dw_start, self.dw_pts = True, None, []
            elif self.speed < self.V_LO * self.W and self.armed:
                if self.dw_start is None:
                    self.dw_start, self.dw_pts = now, []
                self.dw_pts.append(pos)
                if now - self.dw_start >= CFG["type_dwell"]:
                    m = np.array(self.dw_pts).mean(0)
                    self.events.append({"t": self.dw_start, "x": float(m[0]), "y": float(m[1]), "done": False})
                    self.last_event = now
                    self.armed, self.dw_start, self.dw_pts = False, None, []
        while self.events and now - self.events[0]["t"] > 2.5:
            self.events.popleft()
        if self._pattern(now):
            self.active = True
        elif now - self.last_event > 1.2:
            self.active = False
        taps = []
        if self.active:
            for e in self.events:
                if e["done"] or now - e["t"] < 0.45:
                    continue
                e["done"] = True
                if not any(e["t"] - 0.3 <= c <= e["t"] + 0.6 for c in clicks):
                    taps.append((e["x"], e["y"]))
        return taps


# ----------------------------------------------------------------------------
# Gestes : clic, double-clic, clic droit, glisser, défilement, zoom, retour
# ----------------------------------------------------------------------------
class Engine:
    """
    L'appui n'est envoyé qu'une fois le geste compris :
    relâché sans bouger = clic, immobile longtemps = clic droit, déplacé = glisser,
    deux doigts verticaux = défilement, pincement ou deux mains = zoom,
    glissé depuis le bord gauche / droit = retour / avance.
    """

    def __init__(self, W, H, mouse, snap=None):
        self.W, self.H, self.mouse, self.snap = W, H, mouse, snap
        self.mode = "idle"
        self.start = None
        self.t0 = 0.0
        self.edge = None
        self.last_touch = -9.0
        self.last_click = None
        self.click_times = collections.deque(maxlen=60)
        self.base = None
        self.zoom_ref = None
        self.acc = 0.0
        self.anchor = None

    # -- actions ---------------------------------------------------------
    def _click(self, now, p):
        x, y = p
        lc = self.last_click
        if lc and now - lc[0] < 0.45 and math.hypot(x - lc[1], y - lc[2]) < 0.03 * self.W:
            x, y = lc[1], lc[2]            # même point : le système y verra un double-clic
        elif CFG["magnet"] and self.snap:
            x, y = self.snap(x, y)
        self.mouse.click(x, y)
        self.last_click = (now, x, y)
        self.click_times.append(now)

    def tap(self, now, x, y):
        if CFG["magnet"] and self.snap:
            x, y = self.snap(x, y)
        self.mouse.click(x, y)
        self.click_times.append(now)

    def release_all(self):
        self.mouse.release()
        self.mouse.ctrl(False)
        self.mode, self.base = "idle", None

    def _finish(self, now, pos):
        m = self.mode
        if m == "pending":
            self._click(now, self.start)
        elif m == "edge" and pos is not None:
            dx = pos[0] - self.start[0]
            if self.edge == "L" and dx > 0.18 * self.W:
                self.mouse.hotkey("alt", "left")
            elif self.edge == "R" and dx < -0.18 * self.W:
                self.mouse.hotkey("alt", "right")
        elif m == "drag":
            if pos is not None:
                self.mouse.move(*pos)
            self.mouse.release()
        elif m == "zoom":
            self.mouse.ctrl(False)
        self.mode, self.base = "idle", None

    def _zoom(self, d, thr):
        if self.mode == "zoom":
            steps = math.log(d / self.zoom_ref) * CFG["zoom_gain"] * 600
            if abs(steps) >= 30:
                self.mouse.scroll(int(steps))
                self.zoom_ref = d
            return True
        if self.mode not in ("idle", "pending"):
            return False
        if self.base is None:
            self.base = d
            return False
        if abs(math.log(d / self.base)) > math.log(1 + thr):
            self.mode, self.zoom_ref = "zoom", d
            self.mouse.ctrl(True)
            return True
        return False

    def _scroll(self, pos):
        if self.mode != "scroll":
            self.mode, self.anchor, self.acc = "scroll", pos[1], 0.0
            self.mouse.move(*pos)
            return
        self.acc += (pos[1] - self.anchor) * CFG["scroll_gain"] * 2.0
        self.anchor = pos[1]
        u = int(self.acc)
        if u:
            self.mouse.scroll(u)
            self.acc -= u

    # -- une itération ---------------------------------------------------
    def step(self, now, pos, touching, pose, pinch_d, two_d, writing, replay):
        m = self.mouse
        touch = bool(touching and pos is not None)
        if touch:
            self.last_touch = now
        eff = touch or (self.mode not in ("idle", "forced") and now - self.last_touch < 0.06)

        if not eff:
            if self.mode in ("pending", "edge", "drag", "scroll", "zoom", "rightdone"):
                self._finish(now, pos)
            self.base = None
            if self.mode == "forced":
                if writing and pos is not None:
                    m.move(*pos)
                elif not writing:
                    m.release()
                    self.mode = "idle"
            elif writing and pos is not None:
                if replay:
                    m.replay(replay)
                else:
                    m.move(*pos)
                    m.press()
                self.mode = "forced"
            elif pos is not None and CFG["hover"]:
                m.move(*pos)
            return

        if self.mode == "forced":                    # un tracé forcé continue avec le vrai contact
            m.move(*pos)
            return
        if two_d is not None:                        # deux mains : zoom
            self._zoom(two_d, 0.08)
            return
        if pose == "pinch" and pinch_d and self._zoom(pinch_d, 0.15):
            return
        if self.mode == "zoom":
            return
        if pose == "scroll" and self.mode in ("idle", "pending", "scroll"):
            self._scroll(pos)
            return
        if self.mode == "scroll":
            return
        if pose not in ("pointer", "pinch"):
            return

        if self.mode == "idle":
            self.mode, self.start, self.t0 = "pending", pos, now
            self.edge = "L" if pos[0] < 0.03 * self.W else ("R" if pos[0] > 0.97 * self.W else None)
        elif self.mode == "pending":
            if math.hypot(pos[0] - self.start[0], pos[1] - self.start[1]) > CFG["drag_px"]:
                if self.edge:
                    self.mode = "edge"
                else:
                    m.move(*self.start)
                    m.press()
                    m.move(*pos)
                    self.mode = "drag"
            elif now - self.t0 > CFG["long_press"]:
                m.right_click(*self.start)
                self.mode = "rightdone"
        elif self.mode == "drag":
            m.move(*pos)
