"""Full Auto-AVSR lip-reading provider.

Uses mpc001/auto_avsr to perform visual speech recognition on video
segments, producing text transcripts from lip movements.

Setup:
    Run scripts/setup_auto_avsr.sh to download the model and dependencies.
    Or manually:
        git clone https://github.com/mpc001/auto_avsr.git models/auto_avsr
        # Download VSR model weights to models/auto_avsr/
        pip install pytorch-lightning sentencepiece av

The model produces text transcripts from lip movements alone (no audio).
WER: ~19-20% on LRS3 benchmark (visual-only).
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

from . import AVSRHint

logger = logging.getLogger(__name__)

# Singleton pipeline instance (expensive to load — ~1GB model)
_pipeline = None
_pipeline_lock = threading.Lock()


def _find_auto_avsr_dir() -> Optional[str]:
    """Search for the auto_avsr installation directory."""
    candidates = [
        os.environ.get("AUTO_AVSR_DIR"),
        os.path.join(Path(__file__).parent.parent.parent, "models", "auto_avsr"),
        os.path.join(Path.home(), ".auto_avsr"),
        os.path.join(Path.home(), "auto_avsr"),
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            # Check if it has the expected structure
            if os.path.exists(os.path.join(path, "pipelines")) or \
               os.path.exists(os.path.join(path, "configs")):
                return path
    return None


def _get_pipeline(model_dir: Optional[str] = None, device: str = "cpu"):
    """Lazy-initialize the Auto-AVSR inference pipeline (singleton)."""
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        avsr_dir = model_dir or _find_auto_avsr_dir()
        if avsr_dir is None:
            logger.warning(
                "Auto-AVSR directory not found. Run scripts/setup_auto_avsr.sh "
                "or set AUTO_AVSR_DIR environment variable."
            )
            return None

        logger.info("Loading Auto-AVSR from %s (device=%s)...", avsr_dir, device)

        # Add auto_avsr to Python path so its imports work
        if avsr_dir not in sys.path:
            sys.path.insert(0, avsr_dir)

        try:
            import argparse
            import torch
            import torchvision
            from lightning import ModelModule
            from datamodule.transforms import VideoTransform
            from preparation.detectors.mediapipe.detector import LandmarksDetector
            from preparation.detectors.mediapipe.video_process import VideoProcess

            # Find model weights
            model_path = None
            for name in [
                "benchmarks/LRS3/models/vsr_trlrs2lrs3vox2avsp_base.pth",
                "benchmarks/LRS3/models/vsr_trlrs3vox2_base.pth",
                "benchmarks/LRS3/models/vsr_trlrs3_base.pth",
            ]:
                candidate = os.path.join(avsr_dir, name)
                if os.path.exists(candidate):
                    model_path = candidate
                    break

            if model_path is None:
                logger.error("No VSR model weights found in %s/benchmarks/LRS3/models/", avsr_dir)
                return None

            logger.info("Loading VSR model: %s", model_path)

            # Build inline InferencePipeline (same as tutorial notebook)
            class _InferencePipeline(torch.nn.Module):
                def __init__(self, ckpt_path):
                    super().__init__()
                    self.landmarks_detector = LandmarksDetector()
                    self.video_process = VideoProcess(convert_gray=False)
                    self.video_transform = VideoTransform(subset="test")

                    parser = argparse.ArgumentParser()
                    args, _ = parser.parse_known_args(args=[])
                    setattr(args, "modality", "video")

                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    self.modelmodule = ModelModule(args)
                    self.modelmodule.model.load_state_dict(ckpt)
                    self.modelmodule.eval()

                def forward(self, data_filename):
                    data_filename = os.path.abspath(data_filename)
                    video = torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()
                    landmarks = self.landmarks_detector(video)
                    video = self.video_process(video, landmarks)
                    video = torch.tensor(video)
                    video = video.permute((0, 3, 1, 2))
                    video = self.video_transform(video)
                    with torch.no_grad():
                        transcript = self.modelmodule(video)
                    return transcript

            _pipeline = _InferencePipeline(model_path)
            logger.info("Auto-AVSR pipeline loaded successfully")
            return _pipeline

        except ImportError as e:
            logger.error(
                "Failed to import Auto-AVSR: %s. "
                "Install dependencies: pip install pytorch-lightning sentencepiece av torchvision",
                e,
            )
            return None
        except Exception as e:
            logger.error("Failed to initialize Auto-AVSR pipeline: %s", e, exc_info=True)
            return None


class AutoAVSRProvider:
    """Full lip-reading AVSR provider using Auto-AVSR.

    Produces actual text transcripts from lip movements in video segments.
    Requires the auto_avsr package and model weights to be installed.
    """

    def __init__(self, model_dir: Optional[str] = None, device: str = "cpu"):
        self._model_dir = model_dir
        self._device = device
        self._available = False

        # Try to find and verify installation
        avsr_dir = model_dir or _find_auto_avsr_dir()
        if avsr_dir and os.path.isdir(avsr_dir):
            self._available = True
            logger.info("Auto-AVSR provider initialized (dir=%s)", avsr_dir)
        else:
            logger.info(
                "Auto-AVSR provider initialized (NOT AVAILABLE — "
                "run scripts/setup_auto_avsr.sh to install)"
            )

    def _detect_active_speaker_bbox(
        self, video_url: str, start_s: float, end_s: float
    ) -> Optional[dict]:
        """Use MediaPipe to find the active speaker's face bounding box.

        For multi-speaker videos, identifies which face has the most
        mouth movement (MAR variance) and returns its bounding box
        so we can crop the video to just that face for lip reading.

        Returns:
            dict with {x, y, w, h, face_id, num_faces} in normalized coords,
            or None if no faces found.
        """
        try:
            from .mouth_extractor import extract_segment_frames
            import cv2
            import mediapipe as mp

            frames = extract_segment_frames(video_url, start_s, end_s, num_frames=8)
            if not frames:
                return None

            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=5,
                refine_landmarks=True, min_detection_confidence=0.5,
            )

            # Track faces: face_id → {mar_values, bbox_sum, count}
            face_data: dict = {}

            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if not results.multi_face_landmarks:
                    continue

                for fl in results.multi_face_landmarks:
                    lm = fl.landmark
                    nose_x = round(lm[1].x, 1)

                    # Compute MAR
                    upper, lower = lm[13], lm[14]
                    left, right = lm[78], lm[308]
                    vert = ((upper.x - lower.x)**2 + (upper.y - lower.y)**2)**0.5
                    horiz = ((left.x - right.x)**2 + (left.y - right.y)**2)**0.5
                    mar = vert / horiz if horiz > 1e-6 else 0.0

                    # Get face bounding box from landmarks
                    xs = [l.x for l in lm]
                    ys = [l.y for l in lm]
                    bbox = {
                        "x": min(xs), "y": min(ys),
                        "w": max(xs) - min(xs), "h": max(ys) - min(ys),
                    }

                    if nose_x not in face_data:
                        face_data[nose_x] = {"mars": [], "bboxes": [], "count": 0}
                    face_data[nose_x]["mars"].append(mar)
                    face_data[nose_x]["bboxes"].append(bbox)
                    face_data[nose_x]["count"] += 1

            face_mesh.close()

            if not face_data:
                return None

            import numpy as np
            # Active speaker = highest MAR variance
            active_id = max(
                face_data.keys(),
                key=lambda fid: np.std(face_data[fid]["mars"]) if len(face_data[fid]["mars"]) >= 2 else 0.0,
            )

            # Average bounding box across frames for stability
            bboxes = face_data[active_id]["bboxes"]
            avg_bbox = {
                "x": sum(b["x"] for b in bboxes) / len(bboxes),
                "y": sum(b["y"] for b in bboxes) / len(bboxes),
                "w": sum(b["w"] for b in bboxes) / len(bboxes),
                "h": sum(b["h"] for b in bboxes) / len(bboxes),
                "face_id": active_id,
                "num_faces": len(face_data),
            }

            logger.info(
                "Active speaker [%.1f-%.1fs]: face_id=%.1f, %d faces, bbox=(%.2f,%.2f,%.2f,%.2f)",
                start_s, end_s, active_id, len(face_data),
                avg_bbox["x"], avg_bbox["y"], avg_bbox["w"], avg_bbox["h"],
            )
            return avg_bbox

        except Exception as e:
            logger.warning("Active speaker detection failed: %s", e)
            return None

    def analyze_segment(
        self, video_url: str, start_s: float, end_s: float
    ) -> Optional[AVSRHint]:
        """Analyze a video segment using full lip-reading model.

        For multi-speaker videos:
        1. Detects all faces and identifies the active speaker (most mouth movement)
        2. Crops the video to just the active speaker's face
        3. Runs lip reading on the cropped face only

        Args:
            video_url: URL or local path to the video.
            start_s: Segment start time in seconds.
            end_s: Segment end time in seconds.

        Returns:
            AVSRHint with lip_transcript (text from lip reading),
            or None if not available or no face detected.
        """
        if not self._available:
            logger.debug("Auto-AVSR not installed, skipping")
            return None

        pipeline = _get_pipeline(self._model_dir, self._device)
        if pipeline is None:
            return None

        tmpdir = None
        try:
            from .mouth_extractor import extract_video_clip

            tmpdir = tempfile.mkdtemp(prefix="avsr_segment_")

            # Step 1: Detect active speaker
            speaker_bbox = self._detect_active_speaker_bbox(video_url, start_s, end_s)
            num_faces = speaker_bbox["num_faces"] if speaker_bbox else 1
            active_face_id = speaker_bbox["face_id"] if speaker_bbox else None

            # Step 2: Extract video clip, cropped to active speaker if multi-face
            crop_filter = None
            if speaker_bbox and speaker_bbox["num_faces"] > 1:
                # Add padding around the face (20% each side)
                pad = 0.2
                x = max(0, speaker_bbox["x"] - pad * speaker_bbox["w"])
                y = max(0, speaker_bbox["y"] - pad * speaker_bbox["h"])
                w = min(1.0 - x, speaker_bbox["w"] * (1 + 2 * pad))
                h = min(1.0 - y, speaker_bbox["h"] * (1 + 2 * pad))
                # FFmpeg crop filter uses pixel coords — pass as normalized for extract_video_clip
                crop_filter = {"x": x, "y": y, "w": w, "h": h}
                logger.info(
                    "Cropping to active speaker (face %.1f): x=%.2f y=%.2f w=%.2f h=%.2f",
                    speaker_bbox["face_id"], x, y, w, h,
                )

            clip_path = extract_video_clip(
                video_url, start_s, end_s, output_dir=tmpdir,
                crop_normalized=crop_filter,
            )

            logger.info(
                "Running Auto-AVSR lip reading on [%.1f-%.1fs] (%d faces, active=%.1f)...",
                start_s, end_s, num_faces, active_face_id or 0,
            )

            # Step 3: Run lip reading on (cropped) clip
            transcript = pipeline(clip_path)

            if not transcript or not str(transcript).strip():
                return AVSRHint(
                    face_detected=False, speaking_confidence=0.0,
                    lip_transcript=None, mode="auto_avsr",
                    num_faces=num_faces, active_speaker_id=active_face_id,
                )

            transcript = str(transcript).strip()
            logger.info(
                "Auto-AVSR transcript [%.1f-%.1fs]: '%s' (%d faces)",
                start_s, end_s, transcript[:100], num_faces,
            )

            return AVSRHint(
                face_detected=True, speaking_confidence=0.8,
                lip_transcript=transcript, mode="auto_avsr",
                active_speaker_id=active_face_id, num_faces=num_faces,
            )

        except Exception as e:
            logger.warning(
                "Auto-AVSR failed for segment [%.1f-%.1fs]: %s",
                start_s, end_s, e,
            )
            return None

        finally:
            if tmpdir and os.path.exists(tmpdir):
                try:
                    shutil.rmtree(tmpdir)
                except OSError:
                    pass
