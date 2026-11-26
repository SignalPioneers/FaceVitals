"""
occlusion.py  -  FIX #2: masks, sunglasses, hair, hands
======================================================

WHY OCCLUSION HURTS rPPG
    The pulse is read from skin pixels.  A surgical mask turns the cheek ROIs into blue/white
    cloth, sunglasses and hair add dark pixels, and a hand passing the face adds a huge jump.
    Averaging those pixels in with real skin buries the pulse.

WHAT THIS MODULE DOES
    * OcclusionHandler.analyze()  - per ROI, per frame: what fraction is real skin?  Returns a
                                    CLEAN pixel mask, a 0..1 quality score, a stable state
                                    ('ok' / 'partial' / 'occluded') and warnings such as
                                    "mask detected".
    * pool_rgb_traces()           - merges the ROIs frame by frame using their quality, so a
                                    covered cheek simply stops contributing (forehead takes over).
    * fuse_spectra()              - alternative: merge at the pulse-spectrum level.
    * FaceRoiBuilder              - OPTIONAL helper: builds forehead / cheek / eye / lower-face
                                    regions from MediaPipe face-mesh landmarks (or from a plain
                                    face box if landmarks fail, which happens when a mask is worn).
                                    If FaceVitals already produces its 3 ROIs, ignore this class.

QUICK USE
    occ = OcclusionHandler(chroma_model=shared_model)
    rep = occ.analyze(frame_bgr, {"forehead": m1, "left_cheek": m2, "right_cheek": m3})
    for name, st in rep.rois.items():  st.clean_mask, st.quality, st.state
    print(rep.message)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ._common import HR_BAND, fill_nans, power_spectrum, spectral_snr
from .skin_tone import SkinChromaModel


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------
@dataclass
class ROIStatus:
    name: str
    skin_ratio: float            # fraction of the ROI that is real skin this frame
    dark_ratio: float            # fraction of very dark pixels (sunglasses / hair / shadow)
    n_skin: int
    state: str                   # 'ok' | 'partial' | 'occluded'  (stabilised over time)
    quality: float               # 0..1, use as a weight
    clean_mask: np.ndarray       # bool (H,W): skin pixels only
    rgb: Optional[np.ndarray]    # mean RGB of clean pixels (None if too few)
    jump: bool = False           # sudden colour jump (hand, auto-exposure, big motion)


@dataclass
class OcclusionReport:
    rois: Dict[str, ROIStatus]
    flags: Dict[str, bool]
    usable: List[str]
    overall_quality: float
    message: str = ""


# --------------------------------------------------------------------------------------
# Per-frame occlusion analysis
# --------------------------------------------------------------------------------------
class OcclusionHandler:
    def __init__(self, chroma_model: Optional[SkinChromaModel] = None,
                 ok_ratio: float = 0.70, occluded_ratio: float = 0.35,
                 min_pixels: int = 40, good_pixels: int = 60,
                 frames_to_drop: int = 3, frames_to_restore: int = 15,
                 dark_luma: int = 35, jump_thresh: float = 0.05, jump_cooldown: int = 3,
                 morph_min_pixels: int = 400):
        self.model = chroma_model or SkinChromaModel()
        self.ok_ratio, self.occluded_ratio = ok_ratio, occluded_ratio
        self.min_pixels, self.good_pixels = min_pixels, good_pixels
        self.frames_to_drop, self.frames_to_restore = frames_to_drop, frames_to_restore
        self.dark_luma = dark_luma
        self.jump_thresh, self.jump_cooldown = jump_thresh, jump_cooldown
        self.morph_min_pixels = morph_min_pixels
        self._state: Dict[str, str] = {}
        self._bad: Dict[str, int] = {}
        self._good: Dict[str, int] = {}
        self._prev_rgb: Dict[str, np.ndarray] = {}
        self._cool: Dict[str, int] = {}

    # ---- helpers -----------------------------------------------------------------------
    @staticmethod
    def _label(ratio: float, ok: float, occ: float) -> str:
        return "ok" if ratio >= ok else ("partial" if ratio >= occ else "occluded")

    def _stabilise(self, name: str, inst: str) -> str:
        """Hysteresis: drop quickly, recover slowly, so a flickering ROI does not flap."""
        cur = self._state.get(name, inst)
        rank = {"ok": 0, "partial": 1, "occluded": 2}
        if rank[inst] > rank[cur]:                       # getting worse
            self._bad[name] = self._bad.get(name, 0) + 1
            self._good[name] = 0
            if self._bad[name] >= self.frames_to_drop:
                cur, self._bad[name] = inst, 0
        elif rank[inst] < rank[cur]:                     # getting better
            self._good[name] = self._good.get(name, 0) + 1
            self._bad[name] = 0
            if self._good[name] >= self.frames_to_restore:
                cur, self._good[name] = inst, 0
        else:
            self._bad[name] = self._good[name] = 0
        self._state[name] = cur
        return cur

    # ---- main entry --------------------------------------------------------------------
    def analyze(self, frame_bgr: np.ndarray, roi_masks: Dict[str, np.ndarray],
                eye_mask: Optional[np.ndarray] = None,
                lower_face_mask: Optional[np.ndarray] = None,
                global_light_change: bool = False) -> OcclusionReport:
        """
        global_light_change: True when the BACKGROUND jumped together with the face (auto-exposure step,
        light switched).  That is a lighting event, not an occlusion, so it is not treated as a hand/jump.
        """
        ycc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        if not self.model.calibrated:
            self.model.calibrate(ycc, roi_masks)
        eye = (np.asarray(eye_mask) > 0) if eye_mask is not None else None

        rois: Dict[str, ROIStatus] = {}
        for name, m in roi_masks.items():
            m = np.asarray(m) > 0
            if eye is not None and name != "forehead":
                m = m & ~eye                              # never average eyes / glasses frames
            n = int(m.sum())
            if n < self.min_pixels:
                rois[name] = ROIStatus(name, 0.0, 0.0, 0, self._stabilise(name, "occluded"), 0.0,
                                       np.zeros(m.shape, bool), None)
                continue
            skin = self.model.classify(ycc, m)
            if n >= self.morph_min_pixels:                # drop single-pixel speckle (hair strands)
                skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN,
                                        np.ones((3, 3), np.uint8)).astype(bool) & m
            n_skin = int(skin.sum())
            ratio = n_skin / n
            dark = float((ycc[..., 0][m] < self.dark_luma).mean())
            state = self._stabilise(name, self._label(ratio, self.ok_ratio, self.occluded_ratio))

            q_ratio = float(np.clip((ratio - self.occluded_ratio) / (self.ok_ratio - self.occluded_ratio), 0, 1))
            q_pix = float(np.clip(n_skin / self.good_pixels, 0, 1)) ** 0.5
            quality = 0.0 if state == "occluded" else q_ratio * q_pix

            rgb = None
            jump = False
            if n_skin >= self.min_pixels:
                rgb = frame_bgr[skin].astype(np.float64).mean(axis=0)[::-1]
                prev = self._prev_rgb.get(name)
                if (prev is not None and not global_light_change
                        and float(np.max(np.abs(rgb - prev) / (prev + 1e-6))) > self.jump_thresh):
                    self._cool[name] = self.jump_cooldown
                self._prev_rgb[name] = rgb
            if self._cool.get(name, 0) > 0:               # hand pass / exposure step: ignore briefly
                jump, quality = True, 0.0
                self._cool[name] -= 1

            if state == "ok" and ratio > 0.9:
                self.model.update(ycc, skin)              # only learn from clean frames
            rois[name] = ROIStatus(name, ratio, dark, n_skin, state, quality, skin, rgb, jump)

        # ---- whole-face reasoning ---------------------------------------------------------
        flags = {"mask_suspected": False, "sunglasses_suspected": False, "no_skin_visible": False}
        cheeks = [s for k, s in rois.items() if "cheek" in k]
        fore = rois.get("forehead")
        if lower_face_mask is not None and np.asarray(lower_face_mask).any():
            lf = np.asarray(lower_face_mask) > 0
            lf_ratio = float(self.model.classify(ycc, lf).sum() / max(int(lf.sum()), 1))
            flags["mask_suspected"] = lf_ratio < 0.35
        elif cheeks:
            flags["mask_suspected"] = all(c.state == "occluded" for c in cheeks) and (fore is not None and fore.state != "occluded")
        if eye is not None and eye.any():
            e_dark = float((ycc[..., 0][eye] < self.dark_luma + 15).mean())
            e_skin = float(self.model.classify(ycc, eye).sum() / int(eye.sum()))
            flags["sunglasses_suspected"] = e_dark > 0.5 and e_skin < 0.3

        usable = [k for k, s in rois.items() if s.quality > 0.05]
        flags["no_skin_visible"] = len(usable) == 0
        overall = float(max((s.quality for s in rois.values()), default=0.0))
        overall = float(np.clip(np.mean([s.quality for s in rois.values()]) * 0.5 + overall * 0.5, 0, 1)) if rois else 0.0

        msg = ""
        if flags["no_skin_visible"]:
            msg = "No clear skin visible - heart-rate estimate paused. Please face the camera uncovered."
        elif flags["mask_suspected"]:
            msg = "Face mask detected - using the forehead only; the reading will be less stable."
        elif flags["sunglasses_suspected"]:
            msg = "Sunglasses detected - eye area is ignored."
        elif any(s.state == "partial" for s in rois.values()):
            msg = "Part of the face is covered (hair/glasses/hand) - using the visible skin only."
        return OcclusionReport(rois, flags, usable, overall, msg)


# --------------------------------------------------------------------------------------
# Fusion of several ROIs
# --------------------------------------------------------------------------------------
def pool_rgb_traces(traces: Dict[str, np.ndarray], quality: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Quality-weighted merge of per-ROI RGB traces (each (T,3)) into one (T,3) trace.
    Each trace is first divided by its own mean so different ROI brightness cannot cause steps
    when the weights change (e.g. a cheek gets covered).  When every ROI is bad at a frame the
    gap is bridged by interpolation.
    """
    names = list(traces)
    T = traces[names[0]].shape[0]
    num, den, dc_w, dc = np.zeros((T, 3)), np.zeros(T), 0.0, np.zeros(3)
    for n in names:
        tr = np.asarray(traces[n], float)
        q = np.clip(np.asarray(quality[n], float), 0, 1)
        if q.sum() <= 1e-6:
            continue
        mean = (tr * q[:, None]).sum(0) / q.sum()
        num += (tr / (mean + 1e-9)) * q[:, None]
        den += q
        dc += mean * q.mean()
        dc_w += q.mean()
    out = np.full((T, 3), np.nan)
    ok = den > 1e-6
    out[ok] = num[ok] / den[ok, None]
    out = fill_nans(out)
    return out * (dc / max(dc_w, 1e-9) if dc_w > 0 else 1.0)


def fuse_spectra(bvps: Dict[str, np.ndarray], fs: float, weights: Optional[Dict[str, float]] = None,
                 band=HR_BAND):
    """
    Phase-safe alternative to adding waveforms: average the normalised POWER SPECTRA of each
    ROI's pulse signal (weighted by quality x spectral SNR).  Returns (freqs_hz, fused_power).
    """
    acc, f_ref, wsum = None, None, 0.0
    for name, b in bvps.items():
        f, p = power_spectrum(b, fs)
        sel = (f >= band[0]) & (f <= band[1])
        p = p / max(p[sel].sum(), 1e-12)
        _, snr = spectral_snr(b, fs, band)
        w = (weights.get(name, 1.0) if weights else 1.0) * max(snr, 1e-3)
        acc = p * w if acc is None else acc + p * w
        f_ref, wsum = f, wsum + w
    return f_ref, acc / max(wsum, 1e-12)


# --------------------------------------------------------------------------------------
# OPTIONAL: build the ROIs yourself
# --------------------------------------------------------------------------------------
class FaceRoiBuilder:
    """
    Builds ROI polygons in a rotation-aware 'face frame' (u = across the face, v = down the face),
    so they follow head roll.  Names follow the SUBJECT's own left/right.

    from_landmarks : needs the 468-point MediaPipe FaceMesh landmarks in PIXELS, shape (468,2).
    from_face_box  : fallback when landmarks fail (they often do behind a mask): (x, y, w, h).

    ALWAYS eyeball the result once with draw_debug() on your own camera before trusting it.
    """
    _BROW = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]

    @staticmethod
    def _rect(frame, s0, s1, t0, t1):
        c, u, v = frame
        return np.array([c + s * u + t * v for s, t in ((s0, t0), (s1, t0), (s1, t1), (s0, t1))])

    def from_landmarks(self, lm: np.ndarray) -> Dict[str, np.ndarray]:
        lm = np.asarray(lm, float)
        top, chin, fr, fl = lm[10], lm[152], lm[234], lm[454]
        v = chin - top
        fh = np.linalg.norm(v); v = v / fh
        u = fl - fr
        u = u - (u @ v) * v
        fw = np.linalg.norm(u); u = u / fw
        c = 0.5 * (top + chin)
        fr_ = (c, u, v)
        s = lambda i: float((lm[i] - c) @ u)
        t = lambda i: float((lm[i] - c) @ v)

        t_brow = min(t(i) for i in self._BROW)
        H = t_brow - t(10)
        forehead = self._rect(fr_, s(105), s(334), t(10) + 0.30 * H, t_brow - 0.10 * H)

        t0 = max(t(145), t(374)) + 0.05 * fh
        t1 = min(t(61), t(291))
        right = self._rect(fr_, s(234) + 0.13 * fw, s(129) - 0.04 * fw, t0, t1)
        left = self._rect(fr_, s(358) + 0.04 * fw, s(454) - 0.13 * fw, t0, t1)
        eye = self._rect(fr_, s(33) - 0.04 * fw, s(263) + 0.04 * fw,
                         min(t(33), t(263)) - 0.06 * fh, max(t(145), t(374)) + 0.04 * fh)
        lower = self._rect(fr_, s(234) + 0.12 * fw, s(454) - 0.12 * fw, t(1) + 0.02 * fh, t(152) - 0.05 * fh)
        return {"forehead": forehead, "right_cheek": right, "left_cheek": left,
                "eye_band": eye, "lower_face": lower}

    def from_face_box(self, box: Sequence[float]) -> Dict[str, np.ndarray]:
        x, y, w, h = box
        r = lambda a, b, c, d: np.array([(x + a * w, y + c * h), (x + b * w, y + c * h),
                                         (x + b * w, y + d * h), (x + a * w, y + d * h)])
        return {"forehead": r(0.28, 0.72, 0.02, 0.17), "right_cheek": r(0.12, 0.34, 0.50, 0.72),
                "left_cheek": r(0.66, 0.88, 0.50, 0.72), "eye_band": r(0.15, 0.85, 0.22, 0.45),
                "lower_face": r(0.20, 0.80, 0.62, 0.98)}

    @staticmethod
    def polygons_to_masks(polys: Dict[str, np.ndarray], shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        out = {}
        for k, p in polys.items():
            m = np.zeros(shape[:2], np.uint8)
            cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
            out[k] = m
        return out

    @staticmethod
    def draw_debug(frame_bgr: np.ndarray, polys: Dict[str, np.ndarray]) -> np.ndarray:
        img = frame_bgr.copy()
        for k, p in polys.items():
            cv2.polylines(img, [np.round(p).astype(np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img, k, tuple(np.round(p[0]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        return img
