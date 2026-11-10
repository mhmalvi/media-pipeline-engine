"""Tests for the face-tracked reframing geometry."""
from facetrack_reframe import (MAX_SHIFT_PX, CropBox, centre_of_faces, crop_dimensions, face_crop_path,
                               fill_detection_gaps, smooth_path)

ENC_9_16 = {"width": 1080, "height": 1920}


def test_crop_dimensions_are_9_16_and_even():
    # Arrange / Act
    w, h = crop_dimensions(1920, 1080, ENC_9_16)
    # Assert
    assert h <= 1080 and w % 2 == 0 and h % 2 == 0 and abs(w / h - 9 / 16) < 0.01


def test_larger_face_dominates_the_centre():
    # Arrange: a big face at x=100..300 and a small background face at x=1600..1620
    faces = [[100, 0, 200, 200], [1600, 0, 20, 20]]
    # Act
    centre = centre_of_faces(faces, 1920)
    # Assert
    assert 200 <= centre < 300


def test_no_faces_gives_none():
    # Arrange / Act / Assert
    assert centre_of_faces([], 1920) is None and centre_of_faces(None, 1920) is None


def test_gaps_keep_one_entry_per_sample():
    # Arrange
    centres = [None, 500.0, None, None, 700.0, None]
    # Act
    filled = fill_detection_gaps(centres)
    # Assert
    assert filled == [500.0, 500.0, 500.0, 500.0, 700.0, 700.0]


def test_nothing_detected_anywhere_gives_empty():
    # Arrange / Act / Assert
    assert fill_detection_gaps([None, None]) == []


def test_smoothing_caps_movement_per_sample():
    # Arrange
    raw = [CropBox(0, 0, 10, 10), CropBox(500, 0, 10, 10)]
    # Act
    out = smooth_path(raw, MAX_SHIFT_PX)
    # Assert
    assert out[1].x == MAX_SHIFT_PX and len(out) == len(raw)


def test_crop_stays_inside_letterboxed_content():
    # Arrange: picture occupies x 100..1820 inside the frame; face hard against the right edge
    content = (100, 84, 1720, 912)
    # Act
    path = face_crop_path([1919.0] * 400, content, ENC_9_16)
    # Assert
    right_edge = path[-1].x + path[-1].w
    assert right_edge == 100 + 1720
    assert all(b.x >= 100 and b.y >= 84 for b in path)


def test_no_face_falls_back_to_a_centred_box():
    # Arrange
    content = (0, 0, 1920, 1080)
    # Act
    [box] = face_crop_path([], content, ENC_9_16)
    # Assert
    assert box.x == (1920 - box.w) // 2
