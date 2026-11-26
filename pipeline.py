"""
pipeline.py - glue that chains the four fixes in front of FaceVitals' existing POS / CHROM code.

                camera frame + ROI polygons/masks
                          |
     resolution : ROI jitter filter -> anti-aliased sub-pixel ROI masks
     occlusion  : keep only real skin, per-ROI quality, mask/sunglasses/hand handling
     lighting   : glare/shadow pixels dropped, scene-light quality score
     skin_tone  : ITA profile, per-patch means (SNR-weighted pooling later)
                          |   (stored per frame with its real timestamp)
                          v
     get_window(): uniform resample -> patch pooling -> background light cancellation
                   -> quality-weighted merge of the ROIs      ==>  clean (T,3) RGB trace
                          |
     YOUR POS / CHROM  ==> pulse signal ==> SpectralRefiner + HRTracker ==> BPM

The only thing FaceVitals must provide per frame is the ROI geometry (polygons or masks) that it
already computes.  Everything else, including POS/CHROM, stays yours.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from ._common import bandpass, chrom_bvp, pos_bvp
from .lighting import BackgroundIlluminationCanceller, LightingCompensator, LightingReport
from .occlusion import FaceRoiBuilder, OcclusionHandler, OcclusionReport, pool_rgb_traces
from .resolution import (HRResult, HRTracker, ROIStabilizer, SpectralRefiner, TemporalResampler,
                         soft_mask_from_mask, soft_mask_from_polygon, weighted_mean_rgb)
from .skin_tone import PatchSNRPooler, SkinChromaModel, SkinToneAdapter, SkinToneProfile

BvpFn = Callable[[np.ndarray, float], np.ndarray]


@dataclass
class FrameInfo:
    occlusion: OcclusionReport
    lighting: LightingReport
    tone: SkinToneProfile
    messages: List[str]


@dataclass
class WindowData:
    t: np.ndarray
    fs: float
    pooled_rgb: np.ndarray                    # (T,3)  <- feed this to POS / CHROM
    roi_rgb: Dict[str, np.ndarray]
    roi_quality: Dict[str, np.ndarray]
    window_sec: float
    background_explained: float               # 0..1, how much light variation was removed
    n_gap_samples: int


@dataclass
class HRSummary:
    bpm: Optional[float]                      # smoothed value to show the user
    raw_bpm: float
    method: str
    snr_db: float
    confidence: float
    per_method: Dict[str, HRResult]
    messages: List[str] = field(default_factory=list)


class EnhancedFrontEnd:
    def __init__(self, roi_names=("forehead", "left_cheek", "right_cheek"), fs_target: float = 30.0,
                 patch_grid: int = 4, buffer_sec: float = 40.0, min_window_sec: float = 6.0,
                 use_background: bool = True, use_patches: bool = True,
                 patch_bvp_fn: Optional[BvpFn] = None, min_eff_pixels: float = 25.0):
        self.roi_names = tuple(roi_names)
        self.fs_target, self.min_window_sec = fs_target, min_window_sec
        self.buffer_sec = buffer_sec
        self.use_background, self.use_patches = use_background, use_patches
        self.min_eff_pixels = min_eff_pixels

        self.chroma = SkinChromaModel()                       # ONE shared skin model
        self.tone = SkinToneAdapter(self.chroma)
        self.occ = OcclusionHandler(self.chroma)
        self.light = LightingCompensator()
        self.canceller = BackgroundIlluminationCanceller()
        self.stab = ROIStabilizer()
        self.pooler = PatchSNRPooler(grid=patch_grid, bvp_fn=patch_bvp_fn)
        self.resampler = TemporalResampler()
        self.refiner = SpectralRefiner()
        self.tracker = HRTracker()
        self.buf: deque = deque()
        self.last_info: Optional[FrameInfo] = None
        self._last_est_t: Optional[float] = None
        self._prev_bg: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ per frame --------
    def process(self, frame_bgr: np.ndarray, t: float,
                roi_polygons: Optional[Dict[str, np.ndarray]] = None,
                roi_masks: Optional[Dict[str, np.ndarray]] = None,
                face_box: Optional[tuple] = None) -> FrameInfo:
        """
        t: capture timestamp in SECONDS (use time.perf_counter() or the camera timestamp, NOT frame_index/30).
        roi_polygons / roi_masks: your ROIs by name. They may also contain 'eye_band' and 'lower_face'
                      (FaceRoiBuilder produces them) - these improve mask / sunglasses detection.
        face_box: (x, y, w, h) of the face - enables the background light reference.

        Speed note: all per-pixel work is done on a crop around the ROIs, and the background reference on a
        down-scaled copy of the frame, so cost does not grow with camera resolution.
        """
        H, W = frame_bgr.shape[:2]
        # ---- 1) geometry: stabilise polygons, find the crop that contains every ROI -------------
        polys: Dict[str, np.ndarray] = {}
        masks: Dict[str, np.ndarray] = {}
        for name, geom in (roi_polygons or {}).items():
            g = np.asarray(geom, float)
            polys[name] = self.stab(name, g, t) if name in self.roi_names else g
        for name, m in (roi_masks or {}).items():
            if name not in polys:
                masks[name] = (np.asarray(m) > 0).astype(np.uint8)
        xs, ys = [], []
        for g in polys.values():
            xs += [g[:, 0].min(), g[:, 0].max()]; ys += [g[:, 1].min(), g[:, 1].max()]
        for m in masks.values():
            bx, by, bw, bh = cv2.boundingRect(m)
            if bw > 0:
                xs += [bx, bx + bw]; ys += [by, by + bh]
        if not xs:
            empty = FrameInfo(self.occ.analyze(frame_bgr[:1, :1], {}), self.light.assess(frame_bgr[:1, :1], np.zeros((1, 1), bool)),
                              self.tone.profile, ["No ROI supplied for this frame."])
            self.last_info = empty
            return empty
        pad = 6
        x0, y0 = max(int(np.floor(min(xs))) - pad, 0), max(int(np.floor(min(ys))) - pad, 0)
        x1, y1 = min(int(np.ceil(max(xs))) + pad, W), min(int(np.ceil(max(ys))) + pad, H)
        fc = frame_bgr[y0:y1, x0:x1]                                   # crop (a view, no copy)
        Hc, Wc = fc.shape[:2]
        off = np.array([x0, y0], float)

        soft: Dict[str, np.ndarray] = {}
        extra: Dict[str, np.ndarray] = {}
        for name, g in polys.items():
            if name in ("eye_band", "lower_face"):
                extra[name] = soft_mask_from_polygon((Hc, Wc), g - off, 0.0) > 0.5
            elif name in self.roi_names:
                soft[name] = soft_mask_from_polygon((Hc, Wc), g - off)
        for name, m in masks.items():
            mc = m[y0:y1, x0:x1]
            if name in ("eye_band", "lower_face"):
                extra[name] = mc > 0
            elif name in self.roi_names:
                soft[name] = soft_mask_from_mask(mc)

        bbox: Dict[str, tuple] = {}
        for n, sm in soft.items():
            yy, xx = np.nonzero(sm > 0.01)
            if yy.size:
                bbox[n] = (int(xx.min()), int(yy.min()), int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1))

        # ---- 2) background light reference (also tells us if a sudden jump is a LIGHTING event) ----
        bg_rgb, illum_step = None, False
        if self.use_background and face_box is not None and self.chroma.calibrated:
            ds = max(1, int(round(min(H, W) / 160.0)))
            small = cv2.resize(frame_bgr, (W // ds, H // ds), interpolation=cv2.INTER_AREA) if ds > 1 else frame_bgr
            strip = self.light.background_mask(small, tuple(v / ds for v in face_box), None, min_pixels=1)
            if strip is not None:
                skin_like = self.chroma.classify(cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb), strip)
                bg = strip & ~(cv2.dilate(skin_like.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
                if int(bg.sum()) >= 80:
                    bg_rgb = small[bg].astype(np.float64).mean(axis=0)[::-1]
        if bg_rgb is not None and self._prev_bg is not None:
            illum_step = float(np.max(np.abs(bg_rgb - self._prev_bg) / (self._prev_bg + 1e-6))) > 0.03
        self._prev_bg = bg_rgb
        if illum_step:
            # A colour-changing light step moves skin chroma too. Forget the old skin colour so the model
            # re-learns it from the very next classification (instead of rejecting every pixel).
            self.chroma.ref = None

        # ---- 3) occlusion: classify skin inside the (slightly generous) ROI, keep soft edge weights ----
        rep = self.occ.analyze(fc, {n: sm > 0.02 for n, sm in soft.items()},
                               extra.get("eye_band"), extra.get("lower_face"), global_light_change=illum_step)
        weights, union = {}, np.zeros((Hc, Wc), bool)
        for n, st in rep.rois.items():
            keep = st.clean_mask & self.light.keep_mask(fc, st.clean_mask)
            weights[n] = (soft[n] * keep).astype(np.float32)
            union |= keep

        lrep = self.light.assess(fc, union)
        prof = self.tone.update(fc, union, lighting_ok=lrep.ok)

        # ---- 4) per-frame measurements -----------------------------------------------------------
        rec = {"t": float(t), "rgb": {}, "q": {}, "patch": {}, "cov": {}, "bg": bg_rgb}
        for n in soft:
            rgb, wsum = weighted_mean_rgb(fc, weights[n])
            q = rep.rois[n].quality
            if rgb is None or wsum < self.min_eff_pixels:
                rgb, q = None, 0.0
            rec["rgb"][n], rec["q"][n] = rgb, q
            if self.use_patches and n in bbox:
                rec["patch"][n], rec["cov"][n] = self.pooler.grid_means(fc, weights[n], bbox[n])

        self.buf.append(rec)
        while self.buf and rec["t"] - self.buf[0]["t"] > self.buffer_sec:
            self.buf.popleft()

        msgs = [m for m in [rep.message] + lrep.advice if m]
        self.last_info = FrameInfo(rep, lrep, prof, msgs)
        return self.last_info

    # ------------------------------------------------------------------ per window -------
    @staticmethod
    def _interp_time(t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Fill NaN rows of x (N,...) by interpolation over time t."""
        flat = x.reshape(len(t), -1).copy()
        for c in range(flat.shape[1]):
            bad = ~np.isfinite(flat[:, c])
            if bad.all():
                flat[:, c] = 0.0
            elif bad.any():
                flat[bad, c] = np.interp(t[bad], t[~bad], flat[~bad, c])
        return flat.reshape(x.shape)

    def get_window(self, window_sec: Optional[float] = None) -> Optional[WindowData]:
        win = float(window_sec or self.tone.profile.window_sec)
        if len(self.buf) < 20:
            return None
        t_all = np.array([r["t"] for r in self.buf])
        sel = t_all >= t_all[-1] - win
        recs = [r for r, s in zip(self.buf, sel) if s]
        t = t_all[sel]
        if t[-1] - t[0] < self.min_window_sec:
            return None
        fs = self.fs_target
        N = len(recs)
        self.pooler.power = self.tone.profile.patch_power

        def rs(x):                                           # cubic resample onto uniform grid
            return self.resampler.resample(t, x, fs)

        bg = np.array([r["bg"] if r["bg"] is not None else [np.nan] * 3 for r in recs], float)
        have_bg = self.use_background and np.isfinite(bg[:, 0]).mean() > 0.5
        bg_u = None
        if have_bg:
            tu, bg_u, _, gap = rs(self._interp_time(t, bg))
        traces, quals, explained = {}, {}, []
        n_gap = 0
        for n in self.roi_names:
            rgb = np.array([r["rgb"].get(n) if r["rgb"].get(n) is not None else [np.nan] * 3 for r in recs], float)
            if not np.isfinite(rgb[:, 0]).any():
                continue
            q = np.array([r["q"].get(n, 0.0) for r in recs], float)
            rgb = self._interp_time(t, rgb)
            tu, rgb_u, _, gap = rs(rgb)
            n_gap = max(n_gap, int(gap.sum()))
            q_u = np.clip(np.interp(tu, t, q), 0, 1)
            q_u[gap] = 0.0

            tr = rgb_u
            if self.use_patches and all(n in r["patch"] for r in recs):
                P = recs[0]["patch"][n].shape[0]
                patch = np.array([r["patch"][n] for r in recs], float)                    # (N,P,3)
                cov = np.array([r["cov"][n] for r in recs], float)                        # (N,P)
                patch = self._interp_time(t, patch)
                _, patch_u, _, _ = rs(patch.reshape(N, -1))
                patch_u = patch_u.reshape(len(tu), P, 3)
                cov_u = np.clip(np.stack([np.interp(tu, t, cov[:, p]) for p in range(P)], axis=1), 0, 1)
                w = self.pooler.score(patch_u, cov_u, fs)
                pooled = self.pooler.pool(patch_u, cov_u, w)
                if pooled is not None:
                    tr = pooled
            if bg_u is not None:
                bad = q_u < 0.05                                       # do not let garbage frames steer the fit
                tr_fit = tr.copy()
                tr_fit[bad] = np.nan
                tr_fit = self._interp_time(tu, tr_fit)
                tr = self.canceller.cancel(tr_fit, bg_u)
                explained.append(self.canceller.last_explained)
            traces[n], quals[n] = tr, q_u
        if not traces:
            return None
        pooled = pool_rgb_traces(traces, quals)
        return WindowData(tu, fs, pooled, traces, quals, win,
                          float(np.mean(explained)) if explained else 0.0, n_gap)

    # ------------------------------------------------------------------ heart rate -------
    def estimate_hr(self, methods: Optional[Dict[str, BvpFn]] = None,
                    window_sec: Optional[float] = None) -> Optional[HRSummary]:
        """
        methods: {"POS": your_pos, "CHROM": your_chrom}; each is fn(rgb_Tx3, fs) -> pulse_T.
        The method with the better pulse SNR on THIS window wins (usually POS/CHROM differ by lighting).
        Returns None until enough data has been collected.
        """
        w = self.get_window(window_sec)
        if w is None:
            return None
        methods = methods or {"POS": pos_bvp, "CHROM": chrom_bvp}
        results: Dict[str, HRResult] = {}
        for name, fn in methods.items():
            bvp = bandpass(np.asarray(fn(w.pooled_rgb, w.fs), float), w.fs)
            results[name] = self.refiner.estimate(bvp, w.fs)
        best_name = max(results, key=lambda k: results[k].snr_db)
        best = results[best_name]
        dt = 1.0 if self._last_est_t is None else max(w.t[-1] - self._last_est_t, 0.2)
        self._last_est_t = float(w.t[-1])
        smooth = self.tracker.update(best.bpm, best.confidence, dt)
        msgs = list(self.last_info.messages) if self.last_info else []
        if best.confidence < self.tracker.conf_min:
            msgs.append("Signal is weak - hold still and improve the light.")
        return HRSummary(smooth, best.bpm, best_name, best.snr_db, best.confidence, results, msgs)
