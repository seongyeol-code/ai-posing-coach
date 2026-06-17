import numpy as np

# MediaPipe Pose landmark indices used in this module
_IDX = {
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
}


def _pt(landmarks, name: str) -> np.ndarray:
    lm = landmarks.landmark[_IDX[name]]
    return np.array([lm.x, lm.y])


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _angle_at(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at *vertex* formed by the triangle a–vertex–c (degrees)."""
    va = a - vertex
    vc = c - vertex
    denom = np.linalg.norm(va) * np.linalg.norm(vc)
    if denom < 1e-6:
        return 0.0
    cos_val = np.clip(np.dot(va, vc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def _angle_from_vertical(base: np.ndarray, tip: np.ndarray) -> float:
    """Angle between the base→tip vector and the upward vertical axis (degrees).
    0° = perfectly upright, 90° = horizontal."""
    v = tip - base
    # In normalized image coords y increases downward, so "up" is (0, -1)
    up = np.array([0.0, -1.0])
    denom = np.linalg.norm(v)
    if denom < 1e-6:
        return 0.0
    cos_val = np.clip(np.dot(v, up) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def vtaper_ratio(landmarks) -> float:
    """Shoulder-width ÷ hip-width ratio.

    Values above 1.0 indicate a V-taper; classic physique targets ≈ 1.4–1.6.
    Coordinates are normalized (0–1), so horizontal distances are comparable
    without knowing the image dimensions.
    """
    ls = _pt(landmarks, "left_shoulder")
    rs = _pt(landmarks, "right_shoulder")
    lh = _pt(landmarks, "left_hip")
    rh = _pt(landmarks, "right_hip")

    shoulder_w = _dist(ls, rs)
    hip_w = _dist(lh, rh)
    if hip_w < 1e-6:
        return 0.0
    return round(shoulder_w / hip_w, 3)


def symmetry_score(landmarks) -> dict[str, float]:
    """Per-metric and overall left-right symmetry scores (0–100, 100 = perfect).

    Height and length differences are normalized to shoulder width so the
    score is scale-invariant. Angle differences are normalized to 180°.
    """
    ls = _pt(landmarks, "left_shoulder")
    rs = _pt(landmarks, "right_shoulder")
    lh = _pt(landmarks, "left_hip")
    rh = _pt(landmarks, "right_hip")
    le = _pt(landmarks, "left_elbow")
    re = _pt(landmarks, "right_elbow")
    lw = _pt(landmarks, "left_wrist")
    rw = _pt(landmarks, "right_wrist")
    lk = _pt(landmarks, "left_knee")
    rk = _pt(landmarks, "right_knee")
    la = _pt(landmarks, "left_ankle")
    ra = _pt(landmarks, "right_ankle")

    scale = _dist(ls, rs) or 1.0  # shoulder width as the normalizing reference

    def _length_sym(left_val: float, right_val: float, sensitivity: float = 2.0) -> float:
        """Score drops linearly; sensitivity controls how fast."""
        return max(0.0, 100.0 - abs(left_val - right_val) / scale * sensitivity * 100.0)

    def _angle_sym(left_deg: float, right_deg: float, sensitivity: float = 3.0) -> float:
        return max(0.0, 100.0 - abs(left_deg - right_deg) / 180.0 * sensitivity * 100.0)

    scores = {
        # Height difference of matching joints (lower y = higher in frame)
        "shoulder_height": _length_sym(ls[1], rs[1]),
        "hip_height":      _length_sym(lh[1], rh[1]),
        # Total limb lengths
        "arm_length": _length_sym(
            _dist(ls, le) + _dist(le, lw),
            _dist(rs, re) + _dist(re, rw),
        ),
        "leg_length": _length_sym(
            _dist(lh, lk) + _dist(lk, la),
            _dist(rh, rk) + _dist(rk, ra),
        ),
        # Joint angle symmetry
        "elbow_angle": _angle_sym(
            _angle_at(ls, le, lw),
            _angle_at(rs, re, rw),
        ),
        "knee_angle": _angle_sym(
            _angle_at(lh, lk, la),
            _angle_at(rh, rk, ra),
        ),
    }
    scores["overall"] = round(sum(scores.values()) / len(scores), 1)
    scores = {k: round(v, 1) for k, v in scores.items()}
    return scores


def joint_angles(landmarks) -> dict[str, float]:
    """Key joint angles in degrees.

    - elbow / knee: interior flexion angle at the joint (180° = fully extended)
    - shoulder_abduction: angle between torso-side hip, shoulder, and elbow
      (approximates how far the arm is raised from the body)
    - hip_flexion: angle between shoulder, hip, and knee
    - trunk_lean: degrees from vertical of the mid-shoulder → mid-hip spine line
      (0° = perfectly upright)
    """
    ls = _pt(landmarks, "left_shoulder")
    rs = _pt(landmarks, "right_shoulder")
    le = _pt(landmarks, "left_elbow")
    re = _pt(landmarks, "right_elbow")
    lw = _pt(landmarks, "left_wrist")
    rw = _pt(landmarks, "right_wrist")
    lh = _pt(landmarks, "left_hip")
    rh = _pt(landmarks, "right_hip")
    lk = _pt(landmarks, "left_knee")
    rk = _pt(landmarks, "right_knee")
    la = _pt(landmarks, "left_ankle")
    ra = _pt(landmarks, "right_ankle")

    mid_shoulder = (ls + rs) / 2
    mid_hip = (lh + rh) / 2

    return {
        "left_elbow":              round(_angle_at(ls, le, lw), 1),
        "right_elbow":             round(_angle_at(rs, re, rw), 1),
        "left_knee":               round(_angle_at(lh, lk, la), 1),
        "right_knee":              round(_angle_at(rh, rk, ra), 1),
        "left_shoulder_abduction": round(_angle_at(lh, ls, le), 1),
        "right_shoulder_abduction":round(_angle_at(rh, rs, re), 1),
        "left_hip_flexion":        round(_angle_at(ls, lh, lk), 1),
        "right_hip_flexion":       round(_angle_at(rs, rh, rk), 1),
        # spine tilt: measured from mid-hip up to mid-shoulder vs. vertical
        "trunk_lean":              round(_angle_from_vertical(mid_hip, mid_shoulder), 1),
    }


def compute_all_metrics(landmarks) -> dict:
    """Aggregate all metrics into a single dict for passing to the feedback module."""
    return {
        "vtaper_ratio": vtaper_ratio(landmarks),
        "symmetry":     symmetry_score(landmarks),
        "joint_angles": joint_angles(landmarks),
    }
