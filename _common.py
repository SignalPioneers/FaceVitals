"""
_common.py - tiny shared helpers used by the four FaceVitals enhancement modules.

Conventions used everywhere in this package
-------------------------------------------
* Video frames are OpenCV BGR uint8 arrays, shape (H, W, 3).
* Colour TRACES (the mean colour of a region over time) are RGB float arrays, shape (T, 3).
* HR_BAND is the plausible heart-rate band: 0.7-3.0 Hz = 42-180 BPM.

pos_bvp / chrom_bvp here are compact reference versions of the published algorithms.
FaceVitals already has its own POS and CHROM: keep using yours. These are only used
(a) as defaults when a module needs a quick pulse estimate to judge signal quality and
(b) by the self-test in examples/synthetic_benchmark.py.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal

HR_BAND: Tuple[float, float] = (0.7, 3.0)


def bandpass(x: np.ndarray, fs: float, band=HR_BAND, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass along axis 0. Falls back to mean removal if too short."""
    x = np.asarray(x, dtype=float)
    nyq = 0.5 * fs
    lo, hi = band[0] / nyq, min(band[1], 0.95 * nyq) / nyq
    if lo >= hi:
        return x - x.mean(axis=0)
    sos = signal.butter(order, [lo, hi], btype="band", output="sos")
    if x.shape[0] <= 3 * (2 * sos.shape[0] + 1):
        return x - x.mean(axis=0)
    return signal.sosfiltfilt(sos, x, axis=0)


def pos_bvp(rgb: np.ndarray, fs: float, win_sec: float = 1.6) -> np.ndarray:
    """
    POS (Plane-Orthogonal-to-Skin, Wang et al. 2017).
    rgb: (T,3) -> (T,)   or   (T,P,3) -> (T,P)  (P = many traces processed at once).
    """
    rgb = np.asarray(rgb, dtype=float)
    squeeze = rgb.ndim == 2
    if squeeze:
        rgb = rgb[:, None, :]
    T, P, _ = rgb.shape
    L = max(int(round(win_sec * fs)), 8)
    H = np.zeros((T, P))
    for t in range(0, T - L + 1):
        seg = rgb[t:t + L]                                   # (L,P,3)
        cn = seg / (seg.mean(axis=0, keepdims=True) + 1e-9)  # remove skin-tone / brightness (DC)
        s1 = cn[..., 1] - cn[..., 2]                         # G - B
        s2 = -2.0 * cn[..., 0] + cn[..., 1] + cn[..., 2]     # -2R + G + B
        h = s1 + (s1.std(axis=0) / (s2.std(axis=0) + 1e-9)) * s2
        H[t:t + L] += h - h.mean(axis=0)
    return H[:, 0] if squeeze else H


def chrom_bvp(rgb: np.ndarray, fs: float, win_sec: float = 1.6, band=HR_BAND) -> np.ndarray:
    """CHROM (de Haan & Jeanne 2013), windowed with 50% overlap. rgb (T,3) -> (T,)."""
    rgb = np.asarray(rgb, dtype=float)
    T = rgb.shape[0]
    L = max(int(round(win_sec * fs)), 16)
    L += L % 2
    step = L // 2
    out = np.zeros(T)
    hann = np.hanning(L)
    for s in range(0, T - L + 1, step):
        seg = rgb[s:s + L]
        cn = seg / (seg.mean(axis=0) + 1e-9)
        x = 3.0 * cn[:, 0] - 2.0 * cn[:, 1]
        y = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
        xf, yf = bandpass(x, fs, band), bandpass(y, fs, band)
        alpha = xf.std() / (yf.std() + 1e-9)
        h = xf - alpha * yf
        out[s:s + L] += hann * (h / (h.std() + 1e-9))
    return out


def power_spectrum(x: np.ndarray, fs: float, zero_pad: int = 8):
    """Hann-windowed, zero-padded power spectrum. Returns (freqs_hz, power)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    nfft = int(2 ** np.ceil(np.log2(max(n * zero_pad, 512))))
    spec = np.fft.rfft(x * np.hanning(n), nfft)
    return np.fft.rfftfreq(nfft, 1.0 / fs), np.abs(spec) ** 2


def spectral_snr(bvp: np.ndarray, fs: float, band=HR_BAND) -> Tuple[float, float]:
    """
    Pulse SNR (de Haan style): power around the strongest peak (and its 2nd harmonic)
    divided by all remaining power in the HR band.  Returns (peak_hz, snr_linear).
    Roughly: > 1 (0 dB) means a clear pulse, << 1 means noise.
    """
    f, p = power_spectrum(bvp, fs)
    sel = (f >= band[0]) & (f <= band[1])
    fb, pb = f[sel], p[sel]
    if pb.size == 0 or pb.sum() <= 0:
        return float("nan"), 0.0
    f0 = fb[int(np.argmax(pb))]
    hw = max(0.1, 2.0 / (len(bvp) / fs))                     # Hann main-lobe half width
    sig = (np.abs(fb - f0) <= hw) | (np.abs(fb - 2 * f0) <= hw)
    s, n = pb[sig].sum(), pb[~sig].sum()
    return float(f0), float(s / max(n, 1e-12))


def fill_nans(x: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaNs along axis 0 (edges held). Works for (T,) and (T,C)."""
    x = np.array(x, dtype=float, copy=True)
    one_d = x.ndim == 1
    if one_d:
        x = x[:, None]
    idx = np.arange(x.shape[0])
    for c in range(x.shape[1]):
        bad = ~np.isfinite(x[:, c])
        if bad.all():
            x[:, c] = 0.0
        elif bad.any():
            x[bad, c] = np.interp(idx[bad], idx[~bad], x[~bad, c])
    return x[:, 0] if one_d else x
