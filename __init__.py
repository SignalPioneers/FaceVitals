"""FaceVitals enhancements: skin tone, occlusion, lighting and resolution fixes for rPPG."""
from .skin_tone import PatchSNRPooler, SkinChromaModel, SkinToneAdapter, SkinToneProfile
from .occlusion import FaceRoiBuilder, OcclusionHandler, OcclusionReport, fuse_spectra, pool_rgb_traces
from .lighting import (BackgroundIlluminationCanceller, LightingCompensator, LightingReport,
                       lock_auto_controls, smoothness_priors_detrend)
from .resolution import (FaceUpscaler, HRResult, HRTracker, ROIStabilizer, SpectralRefiner, TemporalResampler,
                         configure_capture, soft_mask_from_mask, soft_mask_from_polygon, weighted_mean_rgb)
from .pipeline import EnhancedFrontEnd, FrameInfo, HRSummary, WindowData

__all__ = [n for n in dir() if not n.startswith("_")]
