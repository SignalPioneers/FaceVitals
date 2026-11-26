"""
skin_tone.py  -  FIX #1: skin-tone bias
=========================================

WHY SKIN TONE HURTS rPPG
    Melanin absorbs light, so on darker skin the pulse-driven colour change is a much
    smaller part of what the camera sees.  The pulse is the same, but the SNR is lower,
    so noise, hair, shadows and specular glare win.  Fixed skin thresholds (the classic
    "Cr in 133-173" rule) also fail more often on very dark or very light skin.

WHAT THIS MODULE DOES
    1. SkinChromaModel   - learns THIS person's skin colour (no fixed thresholds) and keeps
                           only pixels that really are skin (drops hair, brows, shadows).
    2. SkinToneAdapter   - measures skin tone with ITA (Individual Typology Angle) and
                           returns tone-appropriate settings (window length, pooling power).
    3. PatchSNRPooler    - splits each ROI into small patches, measures how pulse-like each
                           patch is, and averages mostly the GOOD patches.  This is the part
                           that recovers the most signal on low-SNR (darker) skin.

QUICK USE
    tone = SkinToneAdapter()
    skin = tone.refine_mask(frame_bgr, roi_mask)          # bool mask of true skin pixels
    tone.update(frame_bgr, skin, lighting_ok=True)        # learns ITA over time
    cfg = tone.params()                                   # {'window_sec':..., 'patch_power':...}

ITA in simple words:   ITA = arctan( (L* - 50) / b* ) * 180/pi
    L* = how light the skin is, b* = how yellow/brown it is (both from the CIELAB colour space).
    Big positive ITA = light skin, negative ITA = dark skin.
    NOTE: ITA is only trustworthy in decent light (a dim room makes everyone look "darker"),
    so update() ignores frames the lighting module flags as poor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from ._common import HR_BAND, fill_nans, pos_bvp, spectral_snr

# Wide, tone-agnostic prior for 8-bit OpenCV YCrCb. Used ONLY to bootstrap the personal model.
_PRIOR_CR = (131, 185)      # lower edge kept above 128 (= neutral grey) so walls/masks are not 'skin'
_PRIOR_CB = (70, 135)


# --------------------------------------------------------------------------------------
# 1) Personal skin-colour model
# --------------------------------------------------------------------------------------
class SkinChromaModel:
    """
    Learns the person's skin chroma (Cr, Cb) from the ROIs and classifies pixels as skin.
    One instance should be SHARED by SkinToneAdapter and the occlusion module.
    """

    def __init__(self, k_sigma: float = 3.0, luma_range: Tuple[int, int] = (16, 245),
                 min_scale: float = 4.0, max_scale: float = 14.0, ema: float = 0.02,
                 min_pixels: int = 40):
        self.k = k_sigma
        self.luma_range = luma_range
        self.min_scale, self.max_scale = min_scale, max_scale
        self.ema = ema
        self.min_pixels = min_pixels
        self.ref: Optional[np.ndarray] = None        # [Cr, Cb]
        self.scale = np.array([6.0, 6.0])

    @property
    def calibrated(self) -> bool:
        return self.ref is not None

    def _prior(self, px: np.ndarray) -> np.ndarray:
        y, cr, cb = px[:, 0], px[:, 1], px[:, 2]
        return ((y >= self.luma_range[0]) & (y <= self.luma_range[1]) &
                (cr >= _PRIOR_CR[0]) & (cr <= _PRIOR_CR[1]) &
                (cb >= _PRIOR_CB[0]) & (cb <= _PRIOR_CB[1]))

    def calibrate(self, ycc: np.ndarray, masks: Dict[str, np.ndarray], min_fraction: float = 0.35) -> bool:
        """Bootstrap from ROIs that mostly look like skin under the wide prior."""
        chunks = []
        for m in masks.values():
            m = np.asarray(m) > 0
            if int(m.sum()) < self.min_pixels:
                continue
            px = ycc[m]
            ok = self._prior(px)
            if ok.mean() >= min_fraction:
                chunks.append(px[ok])
        if not chunks:
            return False
        px = np.concatenate(chunks).astype(np.float32)
        med = np.median(px[:, 1:], axis=0)
        mad = np.median(np.abs(px[:, 1:] - med), axis=0) * 1.4826
        self.ref = med
        self.scale = np.clip(mad, self.min_scale, self.max_scale)
        return True

    def classify(self, ycc: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Boolean full-frame mask: pixels inside `mask` that look like THIS person's skin."""
        m = np.asarray(mask) > 0
        out = np.zeros(m.shape, dtype=bool)
        if not m.any():
            return out
        px = ycc[m].astype(np.float32)
        y, cr, cb = px[:, 0], px[:, 1], px[:, 2]
        in_luma = (y >= self.luma_range[0]) & (y <= self.luma_range[1])
        if self.ref is None:
            good = in_luma & self._prior(px)
        else:
            d2 = ((cr - self.ref[0]) / self.scale[0]) ** 2 + ((cb - self.ref[1]) / self.scale[1]) ** 2
            good = in_luma & (d2 <= self.k ** 2)
        out[m] = good
        return out

    def update(self, ycc: np.ndarray, skin: np.ndarray) -> None:
        """Slowly follow colour drift (e.g. lamp warming up). Scale is deliberately NOT adapted
        from inliers (that would shrink it every frame and eventually reject everything)."""
        if self.ref is None or int(skin.sum()) < self.min_pixels:
            return
        cur = np.median(ycc[skin][:, 1:].astype(np.float32), axis=0)
        self.ref = (1 - self.ema) * self.ref + self.ema * cur


# --------------------------------------------------------------------------------------
# 2) Skin-tone profile (ITA) and tone-specific settings
# --------------------------------------------------------------------------------------
_ITA_BINS = [(55.0, "very_light"), (41.0, "light"), (28.0, "intermediate"),
             (10.0, "tan"), (-30.0, "brown"), (-1e9, "dark")]

# Heuristic starting points. Darker skin => weaker pulse => integrate longer and trust the
# best patches more.  TUNE these on your own recordings.
_PRESETS = {
    "very_light":   dict(window_sec=8.0,  patch_power=1.0),
    "light":        dict(window_sec=8.0,  patch_power=1.0),
    "intermediate": dict(window_sec=10.0, patch_power=1.25),
    "tan":          dict(window_sec=10.0, patch_power=1.5),
    "brown":        dict(window_sec=12.0, patch_power=2.0),
    "dark":         dict(window_sec=14.0, patch_power=2.0),
}


@dataclass
class SkinToneProfile:
    ita: float
    category: str
    window_sec: float
    patch_power: float
    n_updates: int
    confident: bool          # False until enough good-light frames were seen


class SkinToneAdapter:
    def __init__(self, chroma_model: Optional[SkinChromaModel] = None, ema: float = 0.05,
                 min_updates: int = 30, max_pixels: int = 3000):
        self.model = chroma_model or SkinChromaModel()
        self.ema = ema
        self.min_updates = min_updates
        self.max_pixels = max_pixels
        self._ita: Optional[float] = None
        self._n = 0
        self._override: Optional[str] = None

    def set_category(self, category: Optional[str]) -> None:
        """
        Force a tone category ('very_light','light','intermediate','tan','brown','dark') - e.g. from a
        Fitzpatrick picker in your UI - or None to go back to automatic ITA.  Useful in dim rooms where
        ITA cannot be trusted.
        """
        if category is not None and category not in _PRESETS:
            raise ValueError(f"category must be one of {list(_PRESETS)}")
        self._override = category

    # ---- skin pixel selection ----------------------------------------------------------
    def refine_mask(self, frame_bgr: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        ycc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        if not self.model.calibrated:
            self.model.calibrate(ycc, {"roi": roi_mask})
        return self.model.classify(ycc, roi_mask)

    # ---- tone estimation ---------------------------------------------------------------
    def update(self, frame_bgr: np.ndarray, skin_mask: np.ndarray, lighting_ok: bool = True) -> SkinToneProfile:
        if lighting_ok:
            px = frame_bgr[np.asarray(skin_mask) > 0]
            if len(px) >= 50:
                if len(px) > self.max_pixels:
                    px = px[np.random.default_rng(0).choice(len(px), self.max_pixels, replace=False)]
                lab = cv2.cvtColor((px.astype(np.float32) / 255.0).reshape(1, -1, 3), cv2.COLOR_BGR2Lab)[0]
                L, b = float(np.median(lab[:, 0])), float(np.median(lab[:, 2]))
                ita = float(np.degrees(np.arctan2(L - 50.0, max(b, 1e-3))))
                self._ita = ita if self._ita is None else (1 - self.ema) * self._ita + self.ema * ita
                self._n += 1
        return self.profile

    @property
    def profile(self) -> SkinToneProfile:
        if self._override is not None:
            p = _PRESETS[self._override]
            return SkinToneProfile(self._ita if self._ita is not None else float("nan"), self._override,
                                   p["window_sec"], p["patch_power"], self._n, True)
        if self._ita is None:                                   # unknown yet -> neutral middle setting
            # unknown (not enough good-light frames yet): cautious defaults - a bit longer, a bit pickier
            return SkinToneProfile(float("nan"), "unknown", 12.0, 1.5, 0, False)
        cat = next(name for thr, name in _ITA_BINS if self._ita > thr)
        p = _PRESETS[cat]
        return SkinToneProfile(self._ita, cat, p["window_sec"], p["patch_power"], self._n, self._n >= self.min_updates)

    def params(self) -> Dict[str, float]:
        pr = self.profile
        return {"window_sec": pr.window_sec, "patch_power": pr.patch_power}


# --------------------------------------------------------------------------------------
# 3) SNR-weighted patch pooling  (the big win for low-SNR skin)
# --------------------------------------------------------------------------------------
class PatchSNRPooler:
    """
    Instead of ONE average over the whole ROI, keep a grid x grid set of patch averages,
    score every patch by how pulse-like it is (spectral SNR), and average with those scores
    as weights.  Patches on hair, glare or shadow get ~0 weight automatically.

    bvp_fn(rgb_Tx3, fs) -> bvp_T is OPTIONAL.  Pass FaceVitals' own POS/CHROM function to
    score patches with the same algorithm you use for the final answer.  If omitted, a fast
    batched POS is used.
    """

    def __init__(self, grid: int = 4, power: float = 1.0, band=HR_BAND,
                 bvp_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
                 min_coverage: float = 0.25, keep_frac: float = 0.15):
        self.grid, self.power, self.band = grid, power, band
        self.bvp_fn = bvp_fn
        self.min_coverage, self.keep_frac = min_coverage, keep_frac

    # ---- per-frame: cheap grid of patch means ------------------------------------------
    def grid_means(self, frame_bgr: np.ndarray, weights: np.ndarray, bbox: Tuple[int, int, int, int]):
        """
        weights: float (H,W) in 0..1  (soft ROI mask x clean-skin mask)
        bbox   : (x, y, w, h) of the ROI - keep it from the STABLE ROI so patches don't shift
                 when part of the ROI becomes occluded.
        Returns patch_rgb (P,3) RGB, coverage (P,) in 0..1 where P = grid*grid.
        """
        x, y, w, h = bbox
        g = self.grid
        crop = frame_bgr[y:y + h, x:x + w].astype(np.float32)
        wt = weights[y:y + h, x:x + w].astype(np.float32)
        if crop.size == 0:
            return np.full((g * g, 3), np.nan), np.zeros(g * g)
        interp = cv2.INTER_AREA if (h >= g and w >= g) else cv2.INTER_LINEAR
        num = cv2.resize(crop * wt[..., None], (g, g), interpolation=interp)
        den = cv2.resize(wt, (g, g), interpolation=interp)
        rgb = (num / np.maximum(den[..., None], 1e-6))[..., ::-1].reshape(-1, 3)
        return rgb, den.reshape(-1)

    # ---- per-window: score patches, then pool ------------------------------------------
    def score(self, patch_traces: np.ndarray, coverage: np.ndarray, fs: float) -> np.ndarray:
        """patch_traces (T,P,3) uniformly sampled, coverage (T,P).  Returns weights (P,)."""
        T, P, _ = patch_traces.shape
        tr = patch_traces.astype(float).copy()
        tr[coverage < self.min_coverage] = np.nan
        valid_frac = np.isfinite(tr[..., 0]).mean(axis=0)
        for p in range(P):
            tr[:, p, :] = fill_nans(tr[:, p, :])
        if self.bvp_fn is None:
            bvp = pos_bvp(tr, fs)                                # (T,P) batched
        else:
            bvp = np.stack([self.bvp_fn(tr[:, p, :], fs) for p in range(P)], axis=1)
        snr = np.array([spectral_snr(bvp[:, p], fs, self.band)[1] for p in range(P)])
        snr[valid_frac < 0.5] = 0.0
        w = np.power(np.maximum(snr, 0.0), self.power)
        if w.max() > 0:
            w[w < self.keep_frac * w.max()] = 0.0
        return w

    @staticmethod
    def pool(patch_traces: np.ndarray, coverage: np.ndarray, weights: np.ndarray,
             min_coverage: float = 0.25) -> Optional[np.ndarray]:
        """Weighted average of patch traces -> (T,3).  None if no patch is usable."""
        if weights.sum() <= 0:
            return None
        T, P, _ = patch_traces.shape
        tr = patch_traces.astype(float).copy()
        tr[coverage < min_coverage] = np.nan
        for p in range(P):
            tr[:, p, :] = fill_nans(tr[:, p, :])
        dc = tr.mean(axis=0)                                     # (P,3)
        norm = tr / (dc[None] + 1e-9)                            # each patch ~1.0 +- pulse
        wn = weights / weights.sum()
        pooled = (norm * wn[None, :, None]).sum(axis=1)          # (T,3)
        return pooled * (dc * wn[:, None]).sum(axis=0)           # back to pixel units
