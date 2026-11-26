"""
lighting.py  -  FIX #3: poor / changing lighting
================================================

WHY LIGHTING HURTS rPPG
    rPPG reads a ~0.5% colour wobble.  Any light change of that size or bigger looks like a
    pulse: room flicker, a TV, clouds, the webcam's own auto-exposure / auto-white-balance
    stepping, glare on the forehead.  Dim light also means sensor noise swamps the signal.

WHAT THIS MODULE DOES
    * LightingCompensator.assess()          - measures brightness, clipping, shadow spread, colour
                                              cast -> quality score + plain-language advice.
    * LightingCompensator.keep_mask()       - removes glare/saturated and deep-shadow pixels.
    * LightingCompensator.background_mask() - picks a reference patch of background (not skin).
    * BackgroundIlluminationCanceller       - THE KEY FIX.  The background contains the same light
                                              changes as the face but no pulse.  Regressing it out
                                              (in the log domain, where light is multiplicative)
                                              removes flicker, drift and auto-exposure steps.
    * smoothness_priors_detrend()           - removes slow drift without touching the HR band.
    * enhance_for_detection()               - gamma + CLAHE ONLY for the face detector, so faces
                                              are found in dim rooms WITHOUT distorting the signal.
    * lock_auto_controls()                  - best-effort: freeze webcam auto-exposure/auto-WB.

IMPORTANT HONESTY NOTE
    Software cannot create photons.  In a very dark room the best fix is physical: a lamp or
    higher screen brightness in front of the face.  assess() tells the user when to do that.
    Never apply CLAHE / local contrast to the frames you measure the pulse from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


# --------------------------------------------------------------------------------------
# Assessment
# --------------------------------------------------------------------------------------
@dataclass
class LightingReport:
    mean_luma: float
    p5: float
    p95: float
    clipped_high: float
    clipped_low: float
    cast_rg: float
    cast_bg: float
    level: str                  # good | dim | dark | bright | overexposed | uneven | unknown
    quality: float              # 0..1
    ok: bool
    advice: List[str] = field(default_factory=list)


class LightingCompensator:
    def __init__(self, dim_luma: float = 75, dark_luma: float = 45, bright_luma: float = 200,
                 clip_hi: int = 250, spread_limit: float = 110):
        self.dim_luma, self.dark_luma, self.bright_luma = dim_luma, dark_luma, bright_luma
        self.clip_hi, self.spread_limit = clip_hi, spread_limit

    # ---- 1) how good is the light on the skin? ----------------------------------------
    def assess(self, frame_bgr: np.ndarray, mask: np.ndarray) -> LightingReport:
        m = np.asarray(mask) > 0
        if int(m.sum()) < 30:
            return LightingReport(0, 0, 0, 0, 0, 1, 1, "unknown", 0.0, False, ["No skin area to judge lighting."])
        px = frame_bgr[m].astype(np.float32)
        b, g, r = px[:, 0], px[:, 1], px[:, 2]
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        mean, p5, p95 = float(luma.mean()), float(np.percentile(luma, 5)), float(np.percentile(luma, 95))
        hi = float((px.max(axis=1) >= self.clip_hi).mean())
        lo = float((luma < 15).mean())
        cast_rg, cast_bg = float(r.mean() / (g.mean() + 1e-6)), float(b.mean() / (g.mean() + 1e-6))

        advice: List[str] = []
        if mean < self.dark_luma:
            level = "dark"
            advice.append("Too dark: face a window or put a lamp in FRONT of you (not behind).")
        elif mean < self.dim_luma:
            level = "dim"
            advice.append("Light is dim: add a lamp or raise screen brightness for a steadier reading.")
        elif hi > 0.05:
            level, _ = "overexposed", advice.append("Parts of the face are overexposed: avoid direct glare / strong spot lights.")
        elif mean > self.bright_luma:
            level, _ = "bright", advice.append("Very bright: reduce direct light on the face.")
        elif (p95 - p5) > self.spread_limit:
            level, _ = "uneven", advice.append("Uneven lighting (strong shadows): use soft light from the front.")
        else:
            level = "good"

        s_luma = float(np.clip((mean - 25) / (80 - 25), 0, 1) * np.clip((250 - mean) / (250 - 190), 0, 1))
        s_clip = 1.0 - min(1.0, hi * 8.0)
        s_even = 1.0 - 0.7 * float(np.clip(((p95 - p5) - 90) / 90, 0, 1))
        quality = float(np.clip(s_luma * s_clip * s_even, 0, 1))
        return LightingReport(mean, p5, p95, hi, lo, cast_rg, cast_bg, level, quality, quality >= 0.5, advice)

    # ---- 2) glare / deep-shadow pixel rejection ---------------------------------------
    def keep_mask(self, frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Boolean mask of pixels in `mask` that are neither saturated/glare nor in deep shadow."""
        m = np.asarray(mask) > 0
        out = np.zeros(m.shape, bool)
        if not m.any():
            return out
        px = frame_bgr[m].astype(np.float32)
        luma = 0.299 * px[:, 2] + 0.587 * px[:, 1] + 0.114 * px[:, 0]
        med = float(np.median(luma))
        mad = float(np.median(np.abs(luma - med))) * 1.4826
        spec = (px.max(axis=1) >= self.clip_hi) | (luma > med + max(3.0 * mad, 30.0))
        shadow = luma < 15
        out[m] = ~(spec | shadow)
        return out

    # ---- 3) reference background -------------------------------------------------------
    def background_mask(self, frame_bgr: np.ndarray, face_box: Tuple[int, int, int, int],
                        skin_like: Optional[np.ndarray] = None, gap: float = 0.15, width: float = 0.45,
                        min_pixels: int = 300) -> Optional[np.ndarray]:
        """
        Strips left / right / above the face that are NOT skin-coloured and not clipped.
        Returns bool mask or None if there isn't enough usable background (face fills the frame).
        skin_like: optional bool mask of skin-coloured pixels (neck, hands) to exclude.
        """
        H, W = frame_bgr.shape[:2]
        x, y, w, h = [int(round(v)) for v in face_box]
        g, bw = int(gap * w), int(width * w)
        m = np.zeros((H, W), bool)
        m[max(y, 0):min(y + h, H), max(x - g - bw, 0):max(x - g, 0)] = True                  # left strip
        m[max(y, 0):min(y + h, H), min(x + w + g, W):min(x + w + g + bw, W)] = True          # right strip
        m[max(y - g - bw, 0):max(y - g, 0), max(x, 0):min(x + w, W)] = True                  # above head
        luma = frame_bgr.mean(axis=2)
        m &= (luma > 8) & (luma < 247)
        if skin_like is not None:
            m &= ~(cv2.dilate(skin_like.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        return m if int(m.sum()) >= min_pixels else None

    # ---- 4) detector-only enhancement ---------------------------------------------------
    @staticmethod
    def enhance_for_detection(frame_bgr: np.ndarray) -> np.ndarray:
        """
        Gamma + CLAHE on the lightness channel so a face/landmark detector works in dim rooms.
        Feed THIS to the detector, then measure the pulse on the ORIGINAL frame.
        """
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l = lab[..., 0]
        mean = float(np.clip(l.mean() / 255.0, 0.02, 0.98))
        gamma = float(np.clip(np.log(0.5) / np.log(mean), 0.5, 2.5))     # map mean lightness -> 0.5
        lut = (255.0 * (np.arange(256) / 255.0) ** gamma).astype(np.uint8)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.LUT(l, lut))
        lab[..., 0] = l
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------------------
# THE key fix: cancel illumination changes using the background
# --------------------------------------------------------------------------------------
class BackgroundIlluminationCanceller:
    """
    Light is multiplicative:  pixel = skin_reflectance x light(t).  In the log domain that becomes
    additive, so   log(face) = log(skin) + pulse + log(light)   and   log(bg) = log(wall) + log(light).
    The background shares log(light) but has no pulse, so a small ridge regression of the face
    trace on the background trace removes the light and leaves the pulse.

    Use on a window of data (e.g. the last 10 s):    clean = canceller.cancel(face_rgb, bg_rgb)
    face_rgb, bg_rgb: (T,3) RGB traces on the SAME uniform time grid.
    """

    def __init__(self, ridge: float = 1e-2, min_explained: float = 0.0):
        self.ridge = ridge
        self.min_explained = min_explained
        self.last_explained: float = 0.0

    def cancel(self, face_rgb: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
        y = np.log(np.clip(np.asarray(face_rgb, float), 1e-3, None))
        x = np.log(np.clip(np.asarray(bg_rgb, float), 1e-3, None))
        ym, xm = y.mean(0), x.mean(0)
        yc, xc = y - ym, x - xm
        # ridge scaled by data energy so it behaves the same for any exposure
        lam = self.ridge * np.trace(xc.T @ xc) / max(xc.shape[1], 1)
        B = np.linalg.solve(xc.T @ xc + lam * np.eye(xc.shape[1]), xc.T @ yc)    # (3,3)
        resid = yc - xc @ B
        denom = float((yc ** 2).sum()) + 1e-12
        self.last_explained = float(1.0 - (resid ** 2).sum() / denom)
        if self.last_explained < self.min_explained:                              # reference is useless
            return np.asarray(face_rgb, float)
        return np.exp(resid + ym)


def smoothness_priors_detrend(x: np.ndarray, fs: float, cutoff_hz: float = 0.4) -> np.ndarray:
    """
    Tarvainen smoothness-priors detrending.  Removes the slow trend (lamp drift, slow head/face
    brightness change) but leaves the heart-rate band.  Works on (T,) or (T,C).

    How lambda is chosen (simple version): the filter's trend cut-off is about
        omega_c = 1/sqrt(lambda)   [radians per sample]   =>   lambda = 1 / omega_c^2
    and omega_c = 2*pi*cutoff_hz/fs.
    """
    x = np.asarray(x, float)
    one_d = x.ndim == 1
    if one_d:
        x = x[:, None]
    T = x.shape[0]
    if T < 8:
        return x[:, 0] - x.mean() if one_d else x - x.mean(0)
    lam = 1.0 / (2 * np.pi * cutoff_hz / fs) ** 2
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(T - 2, T), format="csc")
    A = (sparse.identity(T, format="csc") + (lam ** 2) * (D.T @ D)).tocsc()
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        trend = spsolve(A, x[:, c])
        out[:, c] = x[:, c] - trend
    return out[:, 0] if one_d else out


# --------------------------------------------------------------------------------------
# Best-effort camera control
# --------------------------------------------------------------------------------------
def lock_auto_controls(cap: "cv2.VideoCapture", settle_frames: int = 45) -> Dict[str, str]:
    """
    Freeze auto-exposure and auto-white-balance AFTER they have settled, so the camera stops
    changing brightness/colour in the middle of a measurement.

    Camera drivers differ a lot (V4L2 vs DirectShow vs MSMF vs AVFoundation), and many webcams
    ignore some of these.  This function tries, reads values back, and reports what happened.
    It was written without access to your webcam: test it, and if your driver ignores it, use
    the camera's own software or the OS camera settings to lock exposure/white-balance.
    """
    log: Dict[str, str] = {}
    for _ in range(settle_frames):                       # let auto-exposure converge
        cap.read()
    cur_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
    backend = ""
    try:
        backend = cap.getBackendName()
    except Exception:
        pass
    log["backend"] = backend or "unknown"

    def attempt(label: str, prop: int, value: float):
        try:
            ok = cap.set(prop, value)
            log[label] = f"set {value} -> {'accepted' if ok else 'rejected'}; reads back {cap.get(prop)}"
        except Exception as e:                           # pragma: no cover
            log[label] = f"error: {e}"

    attempt("auto_white_balance_off", cv2.CAP_PROP_AUTO_WB, 0)
    manual_values = [1] if backend.upper().startswith("V4L2") else [0.25, 1, 0]
    for v in manual_values:
        attempt(f"auto_exposure_manual({v})", cv2.CAP_PROP_AUTO_EXPOSURE, v)
    if cur_exp not in (None, 0.0) and np.isfinite(cur_exp):
        attempt("exposure_hold_current", cv2.CAP_PROP_EXPOSURE, cur_exp)
    return log


# --------------------------------------------------------------------------------------
# Optional: score illumination stability of a finished trace
# --------------------------------------------------------------------------------------
def illumination_jumps(trace_rgb: np.ndarray, fs: float, k: float = 6.0) -> np.ndarray:
    """Indices where the (log) brightness jumps far more than any pulse could (auto-exposure steps)."""
    lum = np.log(np.clip(np.asarray(trace_rgb, float).mean(axis=1), 1e-3, None))
    d = np.diff(lum)
    mad = np.median(np.abs(d - np.median(d))) * 1.4826 + 1e-9
    return np.where(np.abs(d - np.median(d)) > k * mad)[0] + 1
