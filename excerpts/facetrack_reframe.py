"""Face-tracked 9:16 reframing: YuNet face centres -> gap-filled, clamped, rate-limited crop path.

Excerpt from clip-engine modules `reframe.facetrack` and `reframe.base`.

YuNet (OpenCV Zoo, Apache-2.0) runs on CPU by design: the small GPU is kept for the local LLM that ranks
clips, and evicting it would break ranking -- a worse outcome than a few seconds of CPU detection. YOLO /
Ultralytics was rejected for this job because it is AGPL-3.0.

Removed for the excerpt: the OpenCV frame-sampling loop (`cv2.FaceDetectorYN` over frames taken at
SAMPLE_FPS), ffprobe dimension probing and the letterbox `cropdetect` pass. `face_crop_path` is the body
of `FaceTrack.crop_path` with those inputs passed in (`centres`, the content rectangle) so it is pure and
testable; the geometry is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

# Frames per second at which crop positions are sampled. Paths are smoothed, so full-rate detection would
# cost many times more for no visible gain.
SAMPLE_FPS = 4
MAX_SHIFT_PX = 18  # max crop movement between consecutive samples


@dataclass(frozen=True)
class CropBox:
    """A crop rectangle in source pixel coordinates."""

    x: int
    y: int
    w: int
    h: int


def crop_dimensions(width: int, height: int, enc: dict) -> tuple[int, int]:
    """Return the widest crop of `width`x`height` matching the output ratio, both sides even.

    Shared by every cropping strategy so a centre crop and a face-tracked crop give the same window size for
    the same source. Rounded down to even numbers because yuv420p chroma subsampling requires it.
    """
    ratio = enc["width"] / enc["height"]
    crop_w = min(width, int(height * ratio))
    crop_h = min(height, int(crop_w / ratio))
    return crop_w - crop_w % 2, crop_h - crop_h % 2


def centre_of_faces(faces, frame_width: int) -> float | None:
    """Area-weighted horizontal centre of detected faces (YuNet rows: x, y, w, h, ...).

    Weighting by area keeps the crop on the speaker in the foreground rather than drifting toward a distant
    background face. Returns None if nothing was detected.
    """
    if faces is None or len(faces) == 0:
        return None
    total_weight = 0.0
    weighted_sum = 0.0
    for row in faces:
        x, y, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        area = max(w * h, 1.0)
        weighted_sum += (x + w / 2.0) * area
        total_weight += area
    if total_weight <= 0:
        return None
    return max(0.0, min(float(frame_width), weighted_sum / total_weight))


def fill_detection_gaps(centres: list[float | None]) -> list[float]:
    """Turn a per-sample detection list into a gap-free per-sample list.

    The crop path is timed one command per SAMPLE, so it must carry one entry per sample. Dropping the
    samples where nothing was detected (profile turns, motion blur, a cutaway) shortened the list and
    shifted every later crop command earlier, so the pan drifted further out of sync with the picture the
    longer the clip ran. The last known centre is carried forward across a gap (the frame holds still
    rather than jumping), and leading gaps hold the first known centre backwards.
    """
    known = [c for c in centres if c is not None]
    if not known:
        return []
    out, last = [], known[0]
    for centre in centres:
        if centre is not None:
            last = centre
        out.append(last)
    return out


def smooth_path(boxes: list[CropBox], max_shift_px: int) -> list[CropBox]:
    """Rate-limit a crop path so the frame never snaps: same length, same dimensions."""
    if not boxes:
        return []
    out = [boxes[0]]
    for box in boxes[1:]:
        prev = out[-1]
        dx = max(-max_shift_px, min(max_shift_px, box.x - prev.x))
        dy = max(-max_shift_px, min(max_shift_px, box.y - prev.y))
        out.append(CropBox(prev.x + dx, prev.y + dy, box.w, box.h))
    return out


def face_crop_path(centres: list[float], content: tuple[int, int, int, int], enc: dict) -> list[CropBox]:
    """Build a smoothed 9:16 crop path following detected faces (body of `FaceTrack.crop_path`).

    Args:
        centres: Gap-filled face centre-x per sample (`fill_detection_gaps`); empty if no face anywhere.
        content: (x, y, w, h) of the real picture inside the frame. Every box stays inside this CONTENT
            rectangle, not merely the frame, so a letterboxed recording's black bars are never cropped in
            and scaled up.
        enc: The preset's encode block (output width / height).

    Returns:
        The crop path; a single centred box when no face was detected in the window.
    """
    base_x, base_y, content_w, content_h = content
    crop_w, crop_h = crop_dimensions(content_w, content_h, enc)
    top = base_y + (content_h - crop_h) // 2
    max_x = base_x + content_w - crop_w
    if not centres:
        return [CropBox(base_x + (content_w - crop_w) // 2, top, crop_w, crop_h)]
    raw = [CropBox(int(max(base_x, min(max_x, c - crop_w / 2))), top, crop_w, crop_h) for c in centres]
    return smooth_path(raw, MAX_SHIFT_PX)
