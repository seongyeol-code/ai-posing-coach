"""Unit tests for src/metrics.py.

MediaPipe landmarks are mocked with a minimal stub so no GPU or camera is needed.
All expected values are derived analytically from the landmark coordinates.
"""

import sys
import os
import math
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics import (
    vtaper_ratio,
    symmetry_score,
    joint_angles,
    compute_all_metrics,
    _angle_at,
    _angle_from_vertical,
)

# ---------------------------------------------------------------------------
# MediaPipe landmark stub
# ---------------------------------------------------------------------------

class _FakeLandmark:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class _FakeLandmarks:
    """Minimal mock of a MediaPipe NormalizedLandmarkList."""
    def __init__(self, coord_map: dict):
        self._lm = {idx: _FakeLandmark(x, y) for idx, (x, y) in coord_map.items()}

    @property
    def landmark(self):
        return self._lm


# Matches _IDX in metrics.py
_IDX = {
    "left_shoulder": 11,  "right_shoulder": 12,
    "left_elbow":    13,  "right_elbow":    14,
    "left_wrist":    15,  "right_wrist":    16,
    "left_hip":      23,  "right_hip":      24,
    "left_knee":     25,  "right_knee":     26,
    "left_ankle":    27,  "right_ankle":    28,
}


def _make_landmarks(**kwargs) -> _FakeLandmarks:
    return _FakeLandmarks({_IDX[k]: v for k, v in kwargs.items()})


def _symmetric_pose() -> _FakeLandmarks:
    """Perfectly symmetric front-facing pose (x=0.5 mirror axis)."""
    return _make_landmarks(
        left_shoulder=(0.3, 0.30),  right_shoulder=(0.7, 0.30),
        left_elbow   =(0.1, 0.45),  right_elbow   =(0.9, 0.45),
        left_wrist   =(0.0, 0.55),  right_wrist   =(1.0, 0.55),
        left_hip     =(0.375, 0.60), right_hip    =(0.625, 0.60),
        left_knee    =(0.375, 0.80), right_knee   =(0.625, 0.80),
        left_ankle   =(0.375, 1.00), right_ankle  =(0.625, 1.00),
    )


# ---------------------------------------------------------------------------
# Tests: private angle helpers
# ---------------------------------------------------------------------------

class TestAngleAt(unittest.TestCase):

    def test_right_angle(self):
        a = np.array([0.0, 0.0])
        v = np.array([1.0, 0.0])
        c = np.array([1.0, 1.0])
        self.assertAlmostEqual(_angle_at(a, v, c), 90.0, places=5)

    def test_straight_angle(self):
        a = np.array([0.0, 0.0])
        v = np.array([1.0, 0.0])
        c = np.array([2.0, 0.0])
        self.assertAlmostEqual(_angle_at(a, v, c), 180.0, places=5)

    def test_60_degree_equilateral(self):
        # Equilateral triangle has all 60° angles
        a = np.array([0.0, 0.0])
        v = np.array([1.0, 0.0])
        c = np.array([0.5, math.sqrt(3) / 2])
        self.assertAlmostEqual(_angle_at(a, v, c), 60.0, places=4)

    def test_degenerate_same_point_returns_zero(self):
        p = np.array([0.5, 0.5])
        self.assertAlmostEqual(_angle_at(p, p, p), 0.0, places=5)

    def test_result_range(self):
        for _ in range(50):
            pts = np.random.rand(3, 2)
            angle = _angle_at(pts[0], pts[1], pts[2])
            self.assertGreaterEqual(angle, 0.0)
            self.assertLessEqual(angle, 180.0)


class TestAngleFromVertical(unittest.TestCase):

    def test_perfectly_upright(self):
        # In image coords y increases downward, so base below tip = upright
        base = np.array([0.5, 0.7])
        tip  = np.array([0.5, 0.3])
        self.assertAlmostEqual(_angle_from_vertical(base, tip), 0.0, places=5)

    def test_horizontal_right(self):
        base = np.array([0.0, 0.5])
        tip  = np.array([1.0, 0.5])
        self.assertAlmostEqual(_angle_from_vertical(base, tip), 90.0, places=5)

    def test_horizontal_left(self):
        base = np.array([1.0, 0.5])
        tip  = np.array([0.0, 0.5])
        self.assertAlmostEqual(_angle_from_vertical(base, tip), 90.0, places=5)

    def test_45_degrees(self):
        base = np.array([0.0, 1.0])
        tip  = np.array([1.0, 0.0])  # equal horizontal and vertical displacement
        self.assertAlmostEqual(_angle_from_vertical(base, tip), 45.0, places=4)

    def test_degenerate_same_point_returns_zero(self):
        p = np.array([0.3, 0.7])
        self.assertAlmostEqual(_angle_from_vertical(p, p), 0.0, places=5)


# ---------------------------------------------------------------------------
# Tests: vtaper_ratio
# ---------------------------------------------------------------------------

class TestVtaperRatio(unittest.TestCase):

    def test_known_ratio(self):
        # shoulder_w = 0.4, hip_w = 0.25 → ratio = 1.6
        lm = _make_landmarks(
            left_shoulder=(0.3, 0.3),  right_shoulder=(0.7, 0.3),
            left_elbow   =(0.1, 0.45), right_elbow   =(0.9, 0.45),
            left_wrist   =(0.0, 0.55), right_wrist   =(1.0, 0.55),
            left_hip     =(0.375, 0.6), right_hip    =(0.625, 0.6),
            left_knee    =(0.375, 0.8), right_knee   =(0.625, 0.8),
            left_ankle   =(0.375, 1.0), right_ankle  =(0.625, 1.0),
        )
        self.assertAlmostEqual(vtaper_ratio(lm), 1.6, places=2)

    def test_equal_shoulder_hip_ratio_one(self):
        # shoulder width == hip width → ratio = 1.0
        lm = _make_landmarks(
            left_shoulder=(0.25, 0.3),  right_shoulder=(0.75, 0.3),
            left_elbow   =(0.1,  0.45), right_elbow   =(0.9,  0.45),
            left_wrist   =(0.0,  0.55), right_wrist   =(1.0,  0.55),
            left_hip     =(0.25, 0.6),  right_hip     =(0.75, 0.6),
            left_knee    =(0.25, 0.8),  right_knee    =(0.75, 0.8),
            left_ankle   =(0.25, 1.0),  right_ankle   =(0.75, 1.0),
        )
        self.assertAlmostEqual(vtaper_ratio(lm), 1.0, places=5)

    def test_result_is_positive(self):
        lm = _symmetric_pose()
        self.assertGreater(vtaper_ratio(lm), 0.0)

    def test_result_is_rounded_to_3_decimals(self):
        lm = _symmetric_pose()
        ratio = vtaper_ratio(lm)
        self.assertEqual(ratio, round(ratio, 3))


# ---------------------------------------------------------------------------
# Tests: symmetry_score
# ---------------------------------------------------------------------------

class TestSymmetryScore(unittest.TestCase):

    def test_perfect_symmetry_all_100(self):
        scores = symmetry_score(_symmetric_pose())
        for key, val in scores.items():
            self.assertAlmostEqual(val, 100.0, places=1,
                                   msg=f"Expected 100 for '{key}', got {val}")

    def test_output_keys(self):
        scores = symmetry_score(_symmetric_pose())
        expected = {"shoulder_height", "hip_height", "arm_length",
                    "leg_length", "elbow_angle", "knee_angle", "overall"}
        self.assertEqual(set(scores.keys()), expected)

    def test_scores_bounded_0_to_100(self):
        scores = symmetry_score(_symmetric_pose())
        for key, val in scores.items():
            self.assertGreaterEqual(val, 0.0, msg=f"{key} < 0")
            self.assertLessEqual(val, 100.0,  msg=f"{key} > 100")

    def test_overall_is_mean_of_sub_scores(self):
        scores = symmetry_score(_symmetric_pose())
        sub_keys = ["shoulder_height", "hip_height", "arm_length",
                    "leg_length", "elbow_angle", "knee_angle"]
        expected_mean = round(sum(scores[k] for k in sub_keys) / len(sub_keys), 1)
        self.assertAlmostEqual(scores["overall"], expected_mean, places=1)

    def test_unequal_shoulder_height_drops_score(self):
        # left shoulder 0.1 lower than right
        lm = _make_landmarks(
            left_shoulder=(0.3, 0.40),  right_shoulder=(0.7, 0.30),
            left_elbow   =(0.1, 0.55),  right_elbow   =(0.9, 0.45),
            left_wrist   =(0.0, 0.65),  right_wrist   =(1.0, 0.55),
            left_hip     =(0.375, 0.70), right_hip    =(0.625, 0.70),
            left_knee    =(0.375, 0.85), right_knee   =(0.625, 0.85),
            left_ankle   =(0.375, 1.00), right_ankle  =(0.625, 1.00),
        )
        scores = symmetry_score(lm)
        self.assertLess(scores["shoulder_height"], 100.0)

    def test_unequal_arm_length_drops_score(self):
        # left arm longer
        lm = _make_landmarks(
            left_shoulder=(0.3, 0.30),  right_shoulder=(0.7, 0.30),
            left_elbow   =(0.05, 0.45), right_elbow   =(0.9, 0.45),  # left elbow further
            left_wrist   =(0.0,  0.55), right_wrist   =(1.0, 0.55),
            left_hip     =(0.375, 0.60), right_hip    =(0.625, 0.60),
            left_knee    =(0.375, 0.80), right_knee   =(0.625, 0.80),
            left_ankle   =(0.375, 1.00), right_ankle  =(0.625, 1.00),
        )
        scores = symmetry_score(lm)
        self.assertLess(scores["arm_length"], 100.0)


# ---------------------------------------------------------------------------
# Tests: joint_angles
# ---------------------------------------------------------------------------

class TestJointAngles(unittest.TestCase):

    def test_output_keys(self):
        angles = joint_angles(_symmetric_pose())
        expected = {
            "left_elbow", "right_elbow",
            "left_knee",  "right_knee",
            "left_shoulder_abduction", "right_shoulder_abduction",
            "left_hip_flexion",        "right_hip_flexion",
            "trunk_lean",
        }
        self.assertEqual(set(angles.keys()), expected)

    def test_all_angles_in_valid_range(self):
        angles = joint_angles(_symmetric_pose())
        for key, val in angles.items():
            self.assertGreaterEqual(val, 0.0,   msg=f"{key} < 0")
            self.assertLessEqual(val,   180.0,  msg=f"{key} > 180")

    def test_symmetric_pose_left_equals_right(self):
        angles = joint_angles(_symmetric_pose())
        pairs = [
            ("left_elbow",              "right_elbow"),
            ("left_knee",               "right_knee"),
            ("left_shoulder_abduction", "right_shoulder_abduction"),
            ("left_hip_flexion",        "right_hip_flexion"),
        ]
        for left_key, right_key in pairs:
            self.assertAlmostEqual(
                angles[left_key], angles[right_key], places=1,
                msg=f"{left_key} vs {right_key} mismatch on symmetric pose",
            )

    def test_upright_pose_trunk_lean_zero(self):
        # mid_shoulder and mid_hip share the same x → trunk_lean = 0°
        lm = _make_landmarks(
            left_shoulder=(0.4, 0.30),  right_shoulder=(0.6, 0.30),
            left_elbow   =(0.2, 0.45),  right_elbow   =(0.8, 0.45),
            left_wrist   =(0.1, 0.55),  right_wrist   =(0.9, 0.55),
            left_hip     =(0.4, 0.70),  right_hip     =(0.6, 0.70),
            left_knee    =(0.4, 0.85),  right_knee    =(0.6, 0.85),
            left_ankle   =(0.4, 1.00),  right_ankle   =(0.6, 1.00),
        )
        self.assertAlmostEqual(joint_angles(lm)["trunk_lean"], 0.0, places=1)

    def test_straight_arm_elbow_180(self):
        # shoulder → elbow → wrist all on same horizontal line → 180°
        lm = _make_landmarks(
            left_shoulder=(0.3, 0.40),  right_shoulder=(0.7, 0.40),
            left_elbow   =(0.2, 0.40),  right_elbow   =(0.8, 0.40),
            left_wrist   =(0.1, 0.40),  right_wrist   =(0.9, 0.40),
            left_hip     =(0.35, 0.70), right_hip     =(0.65, 0.70),
            left_knee    =(0.35, 0.85), right_knee    =(0.65, 0.85),
            left_ankle   =(0.35, 1.00), right_ankle   =(0.65, 1.00),
        )
        angles = joint_angles(lm)
        self.assertAlmostEqual(angles["left_elbow"],  180.0, places=1)
        self.assertAlmostEqual(angles["right_elbow"], 180.0, places=1)

    def test_right_angle_elbow_90(self):
        # shoulder directly above elbow, wrist directly right of elbow → 90°
        # left: ls=(0,0) le=(0,0.5) lw=(0.2,0.5)  va·vc = 0 → 90°
        lm = _make_landmarks(
            left_shoulder=(0.0, 0.0),  right_shoulder=(1.0, 0.0),
            left_elbow   =(0.0, 0.5),  right_elbow   =(1.0, 0.5),
            left_wrist   =(0.2, 0.5),  right_wrist   =(0.8, 0.5),
            left_hip     =(0.2, 0.8),  right_hip     =(0.8, 0.8),
            left_knee    =(0.2, 0.9),  right_knee    =(0.8, 0.9),
            left_ankle   =(0.2, 1.0),  right_ankle   =(0.8, 1.0),
        )
        self.assertAlmostEqual(joint_angles(lm)["left_elbow"], 90.0, places=1)

    def test_straight_legs_knee_180(self):
        # hip, knee, ankle all vertically aligned → knee = 180°
        lm = _make_landmarks(
            left_shoulder=(0.35, 0.30), right_shoulder=(0.65, 0.30),
            left_elbow   =(0.2,  0.45), right_elbow   =(0.8,  0.45),
            left_wrist   =(0.1,  0.55), right_wrist   =(0.9,  0.55),
            left_hip     =(0.35, 0.60), right_hip     =(0.65, 0.60),
            left_knee    =(0.35, 0.80), right_knee    =(0.65, 0.80),
            left_ankle   =(0.35, 1.00), right_ankle   =(0.65, 1.00),
        )
        angles = joint_angles(lm)
        self.assertAlmostEqual(angles["left_knee"],  180.0, places=1)
        self.assertAlmostEqual(angles["right_knee"], 180.0, places=1)


# ---------------------------------------------------------------------------
# Tests: compute_all_metrics
# ---------------------------------------------------------------------------

class TestComputeAllMetrics(unittest.TestCase):

    def test_output_has_all_keys(self):
        result = compute_all_metrics(_symmetric_pose())
        self.assertIn("vtaper_ratio", result)
        self.assertIn("symmetry",     result)
        self.assertIn("joint_angles", result)

    def test_vtaper_matches_standalone(self):
        lm = _symmetric_pose()
        self.assertEqual(compute_all_metrics(lm)["vtaper_ratio"], vtaper_ratio(lm))

    def test_symmetry_matches_standalone(self):
        lm = _symmetric_pose()
        self.assertEqual(compute_all_metrics(lm)["symmetry"], symmetry_score(lm))

    def test_joint_angles_matches_standalone(self):
        lm = _symmetric_pose()
        self.assertEqual(compute_all_metrics(lm)["joint_angles"], joint_angles(lm))

    def test_result_is_dict(self):
        result = compute_all_metrics(_symmetric_pose())
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["symmetry"],     dict)
        self.assertIsInstance(result["joint_angles"], dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
