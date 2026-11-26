"""
resolution.py  -  FIX #4: low camera resolution / low or uneven frame-rate
=========================================================================

WHY LOW RESOLUTION HURTS rPPG
    With a small face (e.g. 40x40 px) each ROI has only tens to hundreds of pixels, so:
      * the ROI edge jumps by whole pixels from frame to frame -> fake "pulses" in the average
      * face/landmark detectors jitter or fail
      * video compression (MJPEG/H.264 at low bitrate) smooths away the tiny colour change
      * webcams often deliver 12-20 fps with irregular timing, which breaks the FFT
    Upscaling does NOT add information, so it is used only to help DETECTION.  The pulse is
    always measured from the original pixels.

WHAT THIS MODULE DOES
    * FaceUpscaler        - crops + upscales (Lanczos or optional DNN super-resolution) the face for the
                            DETECTOR only, and maps points back to the original frame.
    * ROIStabilizer       - One-Euro filter on ROI polygons: removes detector jitter, keeps real motion.
    * soft_mask_from_polygon / soft_mask_from_mask / weighted_mean_rgb
                          - anti-aliased, sub-pixel ROI averaging (no more whole-pixel jumps).
    * TemporalResampler   - uses real timestamps to resample to an even frame-rate (fixes dropped /
                            irregular frames) and flags large gaps.
    * SpectralRefiner     - zero-padded FFT + parabolic peak interpolation: HR to a fraction of a BPM
                            plus an SNR-based confidence.
    * HRTracker           - confidence-gated, rate-limited smoothing of successive HR estimates.
    * configure_capture() - best-effort: ask the webcam for MJPG at the resolution/fps you want.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.interpolate import CubicSpline

from ._common import HR_BAND, power_spectrum, spectral_snr


# --------------------------------------------------------------------------------------
# Detection-side help (upscale a SMALL face so the detector can find it)
# --------------------------------------------------------------------------------------
class FaceUpscaler:
    """
    prepare(frame, box) -> (crop_up, transform).  Run your landmark/face detector on crop_up,
    then transform.to_original(points) maps results back to ORIGINAL frame pixels.

    dnn_model: optional (path, name, scale) for cv2.dnn_superres (needs opencv-contrib), e.g.
               ("ESPCN_x3.pb", "espcn", 3).  Falls back to Lanczos if unavailable.
    """

    @dataclass
    class Transform:
        scale: float
        x0: int
        y0: int

        def to_original(self, pts: np.ndarray) -> np.ndarray:
            return np.asarray(pts, float) / self.scale + np.array([self.x0, self.y0])

    def __init__(self, target_face_px: int = 192, max_scale: float = 4.0, margin: float = 0.35,
                 dnn_model: Optional[Tuple[str, str, int]] = None):
        self.target, self.max_scale, self.margin = target_face_px, max_scale, margin
        self.sr = None
        if dnn_model is not None and hasattr(cv2, "dnn_superres"):
            try:
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                sr.readModel(dnn_model[0]); sr.setModel(dnn_model[1], dnn_model[2])
                self.sr, self._sr_scale = sr, dnn_model[2]
            except Exception:
                self.sr = None

    def prepare(self, frame_bgr: np.ndarray, box: Sequence[float]):
        H, W = frame_bgr.shape[:2]
        x, y, w, h = box
        mx, my = self.margin * w, self.margin * h
        x0, y0 = int(max(x - mx, 0)), int(max(y - my, 0))
        x1, y1 = int(min(x + w + mx, W)), int(min(y + h + my, H))
        crop = frame_bgr[y0:y1, x0:x1]
        scale = float(np.clip(self.target / max(w, 1.0), 1.0, self.max_scale))
        if scale <= 1.01:
            return crop, self.Transform(1.0, x0, y0)
        if self.sr is not None:
            up = self.sr.upsample(crop)
            s = float(self._sr_scale)
            if abs(s - scale) > 0.1:
                up = cv2.resize(up, None, fx=scale / s, fy=scale / s, interpolation=cv2.INTER_AREA)
            return up, self.Transform(scale, x0, y0)
        up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        return up, self.Transform(scale, x0, y0)


# --------------------------------------------------------------------------------------
# ROI jitter removal
# --------------------------------------------------------------------------------------
class OneEuroFilter:
    """Casiez et al. 2012: heavy smoothing when still, light smoothing when moving fast."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.04, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, float)
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


class ROIStabilizer:
    """One filter per ROI polygon (N,2).  stabilizer(name, polygon, timestamp) -> smoothed polygon."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.04):
        self.kw = dict(min_cutoff=min_cutoff, beta=beta)
        self.f: Dict[str, OneEuroFilter] = {}

    def __call__(self, name: str, polygon: np.ndarray, t: float) -> np.ndarray:
        if name not in self.f:
            self.f[name] = OneEuroFilter(**self.kw)
        return self.f[name](np.asarray(polygon, float), t)


# --------------------------------------------------------------------------------------
# Sub-pixel (anti-aliased) ROI averaging
# --------------------------------------------------------------------------------------
def soft_mask_from_polygon(shape: Tuple[int, int], polygon: np.ndarray, feather_px: float = 0.8,
                           shift: int = 4) -> np.ndarray:
    """
    Anti-aliased float mask 0..1 from a FLOAT polygon.  Pixels on the edge get fractional weight, so
    when the ROI moves by 0.3 px the average changes smoothly instead of jumping.
    """
    m = np.zeros(shape[:2], np.uint8)
    pts = np.round(np.asarray(polygon, float) * (1 << shift)).astype(np.int32)
    cv2.fillPoly(m, [pts], 255, lineType=cv2.LINE_AA, shift=shift)
    soft = m.astype(np.float32) / 255.0
    if feather_px > 0:
        soft = cv2.GaussianBlur(soft, (0, 0), feather_px)
    return soft


def soft_mask_from_mask(mask: np.ndarray, feather_px: float = 1.0) -> np.ndarray:
    """Feather an existing binary ROI mask (use this if FaceVitals gives you masks, not polygons)."""
    soft = (np.asarray(mask) > 0).astype(np.float32)
    return cv2.GaussianBlur(soft, (0, 0), feather_px) if feather_px > 0 else soft


def weighted_mean_rgb(frame_bgr: np.ndarray, weights: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Weighted mean colour (RGB order) and the total weight (= effective pixel count)."""
    ys, xs = np.nonzero(weights > 1e-3)
    if ys.size == 0:
        return None, 0.0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    w = weights[y0:y1, x0:x1].astype(np.float64)
    s = float(w.sum())
    if s < 1e-6:
        return None, 0.0
    px = frame_bgr[y0:y1, x0:x1].astype(np.float64)
    return ((px * w[..., None]).sum(axis=(0, 1)) / s)[::-1], s


# --------------------------------------------------------------------------------------
# Time handling
# --------------------------------------------------------------------------------------
class TemporalResampler:
    """
    Webcams do not deliver frames at exactly 30.000 fps.  The FFT and the band-pass assume even
    spacing, so we resample onto a uniform grid using the real timestamps.
    resample(t, x, fs_out) -> (t_uniform, x_uniform, fs, gap_mask)
    gap_mask is True where the original frames were > gap_factor x the normal spacing apart.
    """

    def __init__(self, gap_factor: float = 2.5):
        self.gap_factor = gap_factor

    def resample(self, t: np.ndarray, x: np.ndarray, fs_out: Optional[float] = None):
        t = np.asarray(t, float)
        x = np.asarray(x, float)
        order = np.argsort(t)
        t, x = t[order], x[order]
        keep = np.concatenate(([True], np.diff(t) > 1e-6))
        t, x = t[keep], x[keep]
        if len(t) < 4:
            raise ValueError("need at least 4 samples to resample")
        dt_med = float(np.median(np.diff(t)))
        fs = float(fs_out) if fs_out else 1.0 / dt_med
        tu = np.arange(t[0], t[-1], 1.0 / fs)
        xu = CubicSpline(t, x, axis=0)(tu)
        idx = np.clip(np.searchsorted(t, tu), 1, len(t) - 1)
        gap = (t[idx] - t[idx - 1]) > self.gap_factor * dt_med
        if gap.any():                                       # cubic spline rings across big gaps -> linear there
            flat = x.reshape(len(t), -1)
            lin = np.stack([np.interp(tu, t, flat[:, c]) for c in range(flat.shape[1])], axis=1).reshape(xu.shape)
            xu[gap] = lin[gap]
        return tu, xu, fs, gap


# --------------------------------------------------------------------------------------
# HR estimation with sub-bin precision
# --------------------------------------------------------------------------------------
@dataclass
class HRResult:
    bpm: float
    snr_db: float
    confidence: float        # 0..1
    peak_hz: float


class SpectralRefiner:
    """Zero-padded FFT + 3-point parabolic interpolation of the log-spectrum peak."""

    def __init__(self, band=HR_BAND, zero_pad: int = 8):
        self.band, self.zero_pad = band, zero_pad

    def estimate(self, bvp: np.ndarray, fs: float) -> HRResult:
        f, p = power_spectrum(bvp, fs, self.zero_pad)
        sel = np.where((f >= self.band[0]) & (f <= self.band[1]))[0]
        if sel.size < 3 or p[sel].max() <= 0:
            return HRResult(float("nan"), -99.0, 0.0, float("nan"))
        k = sel[int(np.argmax(p[sel]))]
        f0 = f[k]
        if 0 < k < len(p) - 1:
            a, b, c = np.log(p[k - 1] + 1e-18), np.log(p[k] + 1e-18), np.log(p[k + 1] + 1e-18)
            den = a - 2 * b + c
            if abs(den) > 1e-12:
                f0 = f[k] + 0.5 * (a - c) / den * (f[1] - f[0])
        _, snr = spectral_snr(bvp, fs, self.band)
        snr_db = 10.0 * np.log10(max(snr, 1e-6))
        conf = float(1.0 / (1.0 + np.exp(-(snr_db - 1.0) / 2.0)))
        return HRResult(float(f0 * 60.0), float(snr_db), conf, float(f0))


class HRTracker:
    """
    Smooths successive HR estimates: ignores low-confidence ones, limits how fast HR may change
    (a real heart does not jump 30 BPM in a second), then takes a short median.
    """

    def __init__(self, conf_min: float = 0.35, max_step_bpm: float = 6.0, history: int = 5):
        self.conf_min, self.max_step, self.history = conf_min, max_step_bpm, history
        self.values: List[float] = []
        self.last: Optional[float] = None

    def update(self, bpm: float, confidence: float, dt: float = 1.0) -> Optional[float]:
        if np.isfinite(bpm) and confidence >= self.conf_min:
            if self.last is not None:
                lim = self.max_step * max(dt, 0.2)
                bpm = float(np.clip(bpm, self.last - lim, self.last + lim))
            self.values = (self.values + [bpm])[-self.history:]
            self.last = float(np.median(self.values))
        return self.last


# --------------------------------------------------------------------------------------
# Capture settings
# --------------------------------------------------------------------------------------
def configure_capture(cap: "cv2.VideoCapture", width: int = 1280, height: int = 720, fps: float = 30,
                      fourcc: str = "MJPG") -> Dict[str, object]:
    """
    Ask the webcam for MJPG at the given size/fps.  Many webcams drop to 5-15 fps for big
    uncompressed (YUY2) modes but run 30 fps in MJPG.  Order matters: FOURCC, then size, then fps.
    Returns what the driver actually granted.  Written without your webcam: check the result.
    """
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    return {"width": cap.get(cv2.CAP_PROP_FRAME_WIDTH), "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps_reported": cap.get(cv2.CAP_PROP_FPS),
            "fourcc": "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))}
