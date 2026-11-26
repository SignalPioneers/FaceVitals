# FaceVitals enhancements – integration guide

Four fixes for the four things that skew rPPG heart rate, as drop-in Python modules.
Your existing POS / CHROM code stays exactly as it is.

| Problem | Module | What it does (plain words) |
|---|---|---|
| Skin tone | `skin_tone.py` | Learns *this person's* skin colour instead of fixed thresholds, measures tone (ITA), picks tone-appropriate window length, and averages mostly the best (highest-SNR) patches of each ROI |
| Occlusion (mask, sunglasses, hair, hand) | `occlusion.py` | Keeps only real skin pixels, scores each ROI 0..1, drops covered ROIs (cheeks behind a mask) and lets the visible ones take over, ignores hand-pass jumps |
| Poor / changing lighting | `lighting.py` | Scores the light and tells the user how to fix it, removes glare pixels, and cancels light changes (flicker, drift, auto-exposure steps) using the background as a reference |
| Low resolution / bad frame timing | `resolution.py` | Removes ROI jitter, averages with sub-pixel (anti-aliased) ROI masks, resamples with real timestamps, finds the HR peak to a fraction of a BPM, smooths readings |

`_common.py` = shared helpers. `pipeline.py` = glue that chains the four.

## 1. Install
Copy the `facevitals_enhancements/` folder into the root of your facevitals repo.
Needs `numpy`, `scipy`, `opencv-python`. Optional: `mediapipe` (landmarks), `opencv-contrib-python` (DNN super-resolution).

## 2. Hook it in (replaces "ROI → mean RGB → POS/CHROM")
```python
import time, cv2
from facevitals_enhancements import EnhancedFrontEnd, configure_capture, lock_auto_controls

cap = cv2.VideoCapture(0)
print(configure_capture(cap, 1280, 720, 30, "MJPG"))   # best effort - read what it prints
print(lock_auto_controls(cap))                          # best effort - read what it prints

fe = EnhancedFrontEnd(roi_names=("forehead", "left_cheek", "right_cheek"))
methods = {"POS": my_pos, "CHROM": my_chrom}            # YOUR functions: fn(rgb_Tx3, fs) -> pulse_T

last = 0.0
while True:
    ok, frame = cap.read()
    t = time.perf_counter()                             # real timestamp in seconds (not index/30!)
    polys, face_box = my_roi_code(frame)                # {name: Nx2 float polygon}, (x, y, w, h)
    info = fe.process(frame, t, roi_polygons=polys, face_box=face_box)
    if t - last >= 1.0:                                 # update about once per second
        res = fe.estimate_hr(methods)
        if res: print(res.bpm, res.confidence, res.method, info.messages)
        last = t
```
* ROIs as masks instead of polygons: `fe.process(frame, t, roi_masks={"forehead": mask, ...})`.
* For better mask/sunglasses detection also pass polygons named `"eye_band"` and `"lower_face"`
  (`FaceRoiBuilder().from_landmarks(landmarks_px)` makes them; `.from_face_box(box)` is the fallback when
  landmarks fail behind a mask; `.draw_debug()` lets you SEE the ROIs – check them on your own camera first).
* `info.messages` are ready-made user hints ("Light is dim…", "Face mask detected…").
* Dim room? Feed your face detector `LightingCompensator.enhance_for_detection(frame)`, but always measure the
  pulse on the ORIGINAL frame. Small face? `FaceUpscaler.prepare(frame, box)` helps the detector only.

## 3. Using the modules one at a time
```python
# skin tone                                         # occlusion
tone = SkinToneAdapter()                            occ = OcclusionHandler(chroma_model=tone.model)
skin = tone.refine_mask(frame, roi_mask)            rep = occ.analyze(frame, {"forehead": m1, "left_cheek": m2})
tone.update(frame, skin); tone.params()             rep.rois["left_cheek"].quality; rep.message
tone.set_category("brown")  # force from a UI picker  pool_rgb_traces(traces, qualities)

# lighting                                          # resolution
lc = LightingCompensator()                          soft = soft_mask_from_polygon(frame.shape, poly)
lc.assess(frame, skin_mask).advice                  rgb, n = weighted_mean_rgb(frame, soft)
clean = BackgroundIlluminationCanceller().cancel(   tu, xu, fs, gap = TemporalResampler().resample(t, x, 30)
          face_rgb_Tx3, background_rgb_Tx3)         SpectralRefiner().estimate(bvp, fs)  # .bpm .snr_db .confidence
```

## 4. Tuning knobs (start here if results disagree with your data)
* `skin_tone._PRESETS` – window length / patch-pooling power per tone category.
* `OcclusionHandler(ok_ratio, occluded_ratio, jump_thresh)` – how strict "real skin" and "hand jump" are.
* `LightingCompensator(dim_luma, …)`, `BackgroundIlluminationCanceller(ridge)` – light thresholds / regression strength.
* `ROIStabilizer(min_cutoff, beta)` – more smoothing vs faster tracking. `HRTracker(max_step_bpm)`.

## 5. What was tested (synthetic video, known HR = 72 BPM) – and what that means
`python examples/synthetic_benchmark.py` renders fake faces with each problem and compares a plain pipeline
(hard ROI mean → POS or CHROM → FFT peak) with `EnhancedFrontEnd`. Mean error in BPM (share of 10 s+ windows within ±5 BPM):

| Scenario | Plain POS | Plain CHROM | Enhanced |
|---|---|---|---|
| 1 clean, light skin | 0.4 (100%) | 0.3 (100%) | 0.2 (100%) |
| 2 dark skin + dim + glare | 12.0 (71%) | 14.9 (54%) | 0.2 (100%) |
| 3 mask + sunglasses + hair + hand | 17.3 (62%) | 9.7 (79%) | 0.3 (100%) |
| 4 colour flicker + exposure step | 23.0 (4%) | 24.0 (0%) | 0.6 (100%) |
| 5 low-res + dropped frames | 7.2 (12%) | 9.0 (12%) | 0.2 (100%) |
| 6 realistic combo (medium-dark skin, mask, dim, flicker) | 24.0 (0%) | 24.0 (0%) | 0.4 (100%) |
| 7 STRESS: everything at once, ~100 skin pixels | 30.0 (0%) | 30.0 (0%) | 26.9 (33%) |

`python examples/module_smoke_test.py` checks each module on its own (all pass).
Speed (this sandbox, 1 CPU): ~12 ms/frame at 320×240, ~17 ms/frame at 640×480, ~0.1–0.2 s per HR update.

## 6. Honest limits – please read
* **This is a simulation I generated.** It shows the code removes the failure mechanisms it targets; it does NOT
  prove accuracy on real people. Validate on real recordings (UBFC-rPPG, PURE, and MMPD which has skin-tone
  labels) and report error PER skin-tone group.
* Scenario 7 shows the floor: with a dark skin, dim light, a mask, glasses, hair and a tiny face together there
  are too few good pixels. No software fixes that; the cure is physical (a lamp in front, remove the mask, move closer).
* ITA (skin-tone measurement) is unreliable in dim light, so the tone adapter ignores dim frames and falls back to
  cautious defaults. If you know the person's tone, call `set_category(...)`.
* The background light canceller needs some visible background beside/above the face. If the face fills the frame it
  silently does nothing (`use_background=False` to disable).
* `lock_auto_controls` / `configure_capture` depend on the webcam driver and were written without your hardware:
  read the log they return, and lock exposure/white-balance in the OS camera settings if the driver ignores them.
* `FaceRoiBuilder` landmark proportions were not tested on real faces: use `draw_debug` before trusting them.
* A hand that is skin-coloured is only caught by the sudden-jump detector, not by colour.
