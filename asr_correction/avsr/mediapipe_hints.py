"""MediaPipe-based lightweight AVSR hint provider.

Uses MediaPipe Face Mesh to detect:
- Whether a face is present in the video segment
- Mouth openness via Mouth Aspect Ratio (MAR)
- Speaking activity by measuring MAR variance across frames

This is a lightweight alternative to full lip-reading models, providing
useful hints about speaker presence and activity without requiring
GPU-heavy inference.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from . import AVSRHint
from .mouth_extractor import extract_segment_frames

logger = logging.getLogger(__name__)

# MediaPipe Face Mesh landmark indices for mouth
# Upper inner lip (top): landmark 13
# Lower inner lip (bottom): landmark 14
# Left mouth corner: landmark 78
# Right mouth corner: landmark 308
_UPPER_LIP_IDX = 13
_LOWER_LIP_IDX = 14
_LEFT_CORNER_IDX = 78
_RIGHT_CORNER_IDX = 308


def _compute_mouth_aspect_ratio(landmarks) -> float:
    """Compute Mouth Aspect Ratio (MAR) from face mesh landmarks.

    MAR = vertical_distance / horizontal_distance
    Higher MAR indicates a more open mouth.

    Args:
        landmarks: MediaPipe NormalizedLandmarkList.

    Returns:
        Mouth aspect ratio (float). Typically 0.0-0.8 range.
    """
    upper = landmarks[_UPPER_LIP_IDX]
    lower = landmarks[_LOWER_LIP_IDX]
    left = landmarks[_LEFT_CORNER_IDX]
    right = landmarks[_RIGHT_CORNER_IDX]

    # Vertical distance (lip opening)
    vertical = np.sqrt(
        (upper.x - lower.x) ** 2
        + (upper.y - lower.y) ** 2
    )

    # Horizontal distance (mouth width)
    horizontal = np.sqrt(
        (left.x - right.x) ** 2
        + (left.y - right.y) ** 2
    )

    if horizontal < 1e-6:
        return 0.0

    return vertical / horizontal


class MediaPipeHintProvider:
    """Lightweight AVSR provider using MediaPipe Face Mesh.

    Analyzes a few frames from a video segment to detect face presence
    and estimate speaking activity from mouth movement patterns.
    """

    def __init__(self):
        self._face_mesh = None

    def _get_face_mesh(self):
        """Lazy-load MediaPipe Face Mesh to avoid import cost at startup."""
        if self._face_mesh is None:
            import mediapipe as mp

            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=5,  # Detect up to 5 faces for multi-speaker
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        return self._face_mesh

    def detect_active_speaker(
        self, video_url: str, start_s: float, end_s: float, num_frames: int = 8
    ) -> Optional[int]:
        """Detect which face is the active speaker in a multi-person video.

        Computes MAR variance for each detected face across frames.
        The face with the highest mouth movement variance is the active speaker.

        Returns:
            Index of the active speaker face (0-based), or None if no faces found.
        """
        frames = extract_segment_frames(video_url, start_s, end_s, num_frames=num_frames)
        if not frames:
            return None

        import cv2
        face_mesh = self._get_face_mesh()

        # Track MAR values per face position (using face center x-coord as ID)
        face_tracks: dict = {}  # face_id → list of MAR values

        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                continue

            for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                landmarks = face_landmarks.landmark
                # Use nose tip x-coordinate to track face identity across frames
                nose_x = landmarks[1].x
                face_id = round(nose_x, 1)  # Bucket to nearest 0.1

                mar = _compute_mouth_aspect_ratio(landmarks)
                if face_id not in face_tracks:
                    face_tracks[face_id] = []
                face_tracks[face_id].append(mar)

        if not face_tracks:
            return None

        # The active speaker has the highest MAR variance (mouth opening/closing)
        best_face_id = max(
            face_tracks.keys(),
            key=lambda fid: np.std(face_tracks[fid]) if len(face_tracks[fid]) >= 2 else 0.0,
        )

        logger.debug(
            "Active speaker detection: %d faces tracked, active=%.1f (MAR std=%.4f)",
            len(face_tracks), best_face_id,
            np.std(face_tracks[best_face_id]) if len(face_tracks[best_face_id]) >= 2 else 0.0,
        )

        return best_face_id

    def analyze_segment(
        self, video_url: str, start_s: float, end_s: float
    ) -> Optional[AVSRHint]:
        """Analyze a video segment for face presence and speaking activity.

        For multi-speaker videos, detects the active speaker first,
        then measures speaking confidence for that specific face.

        Args:
            video_url: URL or local path to the video.
            start_s: Segment start time in seconds.
            end_s: Segment end time in seconds.

        Returns:
            AVSRHint with detection results, or None on failure.
        """
        frames = extract_segment_frames(video_url, start_s, end_s, num_frames=8)
        if not frames:
            logger.debug(
                "No frames extracted for segment [%.2f-%.2f]s", start_s, end_s
            )
            return AVSRHint(
                face_detected=False,
                speaking_confidence=0.0,
                mode="mediapipe",
            )

        import cv2
        face_mesh = self._get_face_mesh()

        # Track all faces and their MAR values
        face_tracks: dict = {}  # face_id → list of MAR values

        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                continue

            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                nose_x = landmarks[1].x
                face_id = round(nose_x, 1)
                mar = _compute_mouth_aspect_ratio(landmarks)
                if face_id not in face_tracks:
                    face_tracks[face_id] = []
                face_tracks[face_id].append(mar)

        if not face_tracks:
            return AVSRHint(
                face_detected=False,
                speaking_confidence=0.0,
                mode="mediapipe",
            )

        # Find the active speaker (highest MAR variance)
        active_face_id = max(
            face_tracks.keys(),
            key=lambda fid: np.std(face_tracks[fid]) if len(face_tracks[fid]) >= 2 else 0.0,
        )
        active_mar_values = face_tracks[active_face_id]

        speaking_confidence = _estimate_speaking_confidence(active_mar_values)
        face_ratio = len(active_mar_values) / len(frames)
        speaking_confidence *= face_ratio

        logger.debug(
            "Segment [%.2f-%.2f]s: %d faces, active_speaker=%.1f, MAR=%s, confidence=%.2f",
            start_s, end_s, len(face_tracks), active_face_id,
            [f"{v:.3f}" for v in active_mar_values], speaking_confidence,
        )

        return AVSRHint(
            face_detected=True,
            speaking_confidence=round(speaking_confidence, 3),
            lip_transcript=None,
            mode="mediapipe",
            active_speaker_id=active_face_id,
            num_faces=len(face_tracks),
        )


def _estimate_speaking_confidence(mar_values: List[float]) -> float:
    """Estimate speaking confidence from mouth aspect ratios across frames.

    Speaking is characterized by:
    - Variation in MAR (mouth opening and closing)
    - At least some frames with elevated MAR (mouth open)

    Args:
        mar_values: List of MAR measurements across frames.

    Returns:
        Confidence score 0.0-1.0.
    """
    if not mar_values:
        return 0.0

    mar_array = np.array(mar_values)
    mar_std = float(np.std(mar_array))
    mar_mean = float(np.mean(mar_array))

    # Speaking indicators:
    # 1. MAR variance — mouth opens and closes while speaking
    variance_score = min(mar_std / 0.05, 1.0)  # Normalize: std of 0.05+ is high activity

    # 2. Mean MAR — some baseline mouth opening
    openness_score = min(mar_mean / 0.15, 1.0)  # Normalize: MAR of 0.15+ is clearly open

    # Combine: variance is the stronger signal for actual speech
    confidence = 0.6 * variance_score + 0.4 * openness_score

    return min(max(confidence, 0.0), 1.0)
