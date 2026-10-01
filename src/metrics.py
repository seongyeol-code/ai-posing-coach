# src/metrics.py
# 관절 좌표로 포즈 지표(V-테이퍼, 좌우 대칭, 관절 각도)를 계산하는 모듈
#
# [중요] 사진 비율 보정
# MediaPipe 좌표는 가로·세로를 "각각" 0~1로 맞춘 값이에요.
# 예를 들어 1920x1080 사진에서 x 0.1은 192px인데 y 0.1은 108px이라,
# 이 숫자로 그대로 거리·각도를 재면 사진 모양에 따라 각도가 찌그러져요.
# 그래서 계산 전에 y에 (세로 ÷ 가로) 비율을 곱해 실제 사진 모양대로 되돌린 뒤 계산해요.
# 이렇게 하면 같은 자세는 사진을 어떻게 잘라도 같은 각도가 나와요.

import numpy as np

# 이 모듈에서 쓰는 MediaPipe 관절 번호
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


# ---------------------------------------------------------------------------
# 계산 도우미
# ---------------------------------------------------------------------------

def _points(landmarks, aspect: float = 1.0) -> dict[str, np.ndarray]:
    """필요한 관절 12개를 {이름: [x, y]} 형태로 꺼냄.
    aspect = 사진 세로 ÷ 가로. y에 곱해서 실제 사진 비율로 맞춰요."""
    # Tasks API는 관절 리스트를 바로 주고, 옛 방식은 .landmark 안에 리스트가 있음 → 둘 다 지원
    lms = getattr(landmarks, "landmark", landmarks)
    return {
        name: np.array([lms[i].x, lms[i].y * aspect])
        for name, i in _IDX.items()
    }


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    """두 점 사이의 직선 거리"""
    return float(np.linalg.norm(a - b))


def _angle_at(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float:
    """vertex에서 a와 c가 이루는 각도 (벡터 내적, 도 단위)"""
    va = a - vertex
    vc = c - vertex
    denom = np.linalg.norm(va) * np.linalg.norm(vc)
    if denom < 1e-6:
        return 0.0
    cos_val = np.clip(np.dot(va, vc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def _angle_from_vertical(base: np.ndarray, tip: np.ndarray) -> float:
    """base→tip 방향이 수직(위쪽)에서 몇 도 기울었는지. 0° = 완전히 수직, 90° = 수평"""
    v = tip - base
    # 이미지 좌표는 y가 아래로 커지므로 "위쪽"은 (0, -1)
    up = np.array([0.0, -1.0])
    denom = np.linalg.norm(v)
    if denom < 1e-6:
        return 0.0
    cos_val = np.clip(np.dot(v, up) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


# ---------------------------------------------------------------------------
# 지표 계산
# ---------------------------------------------------------------------------

def vtaper_ratio(landmarks, aspect: float = 1.0) -> float:
    """어깨너비 ÷ 골반너비 (1보다 크면 V자, 클래식 피지크 이상 범위 약 1.4~1.6).

    주의: MediaPipe의 11·12번은 어깨 관절, 23·24번은 고관절이라
    근육 윤곽이 아니라 뼈대 기준 비율이에요.
    """
    p = _points(landmarks, aspect)
    shoulder_w = _dist(p["left_shoulder"], p["right_shoulder"])
    hip_w = _dist(p["left_hip"], p["right_hip"])
    if hip_w < 1e-6:
        return 0.0
    return round(shoulder_w / hip_w, 3)


def symmetry_score(landmarks, aspect: float = 1.0) -> dict[str, float]:
    """항목별 좌우 대칭 점수와 전체 평균 (0~100, 100 = 완벽한 대칭).

    높이·길이 차이는 어깨너비로 나눠서 사진 크기와 상관없게 만들고,
    각도 차이는 180°로 나눠요.
    """
    p = _points(landmarks, aspect)
    ls, rs = p["left_shoulder"], p["right_shoulder"]
    le, re = p["left_elbow"], p["right_elbow"]
    lw, rw = p["left_wrist"], p["right_wrist"]
    lh, rh = p["left_hip"], p["right_hip"]
    lk, rk = p["left_knee"], p["right_knee"]
    la, ra = p["left_ankle"], p["right_ankle"]

    scale = _dist(ls, rs) or 1.0  # 기준 길이 = 어깨너비

    def _length_sym(left_val: float, right_val: float, sensitivity: float = 2.0) -> float:
        """차이가 클수록 점수가 직선으로 깎임 (sensitivity가 클수록 빨리 깎임)"""
        return max(0.0, 100.0 - abs(left_val - right_val) / scale * sensitivity * 100.0)

    def _angle_sym(left_deg: float, right_deg: float, sensitivity: float = 3.0) -> float:
        return max(0.0, 100.0 - abs(left_deg - right_deg) / 180.0 * sensitivity * 100.0)

    scores = {
        # 짝 관절의 높이 차이
        "shoulder_height": _length_sym(ls[1], rs[1]),
        "hip_height":      _length_sym(lh[1], rh[1]),
        # 팔·다리 전체 길이
        "arm_length": _length_sym(
            _dist(ls, le) + _dist(le, lw),
            _dist(rs, re) + _dist(re, rw),
        ),
        "leg_length": _length_sym(
            _dist(lh, lk) + _dist(lk, la),
            _dist(rh, rk) + _dist(rk, ra),
        ),
        # 관절 각도 대칭
        "elbow_angle": _angle_sym(_angle_at(ls, le, lw), _angle_at(rs, re, rw)),
        "knee_angle":  _angle_sym(_angle_at(lh, lk, la), _angle_at(rh, rk, ra)),
    }
    scores["overall"] = round(sum(scores.values()) / len(scores), 1)
    return {k: round(v, 1) for k, v in scores.items()}


def joint_angles(landmarks, aspect: float = 1.0) -> dict[str, float]:
    """주요 관절 각도 (도).

    - 팔꿈치·무릎: 관절이 굽은 안쪽 각도 (180° = 완전히 폄)
    - 어깨 외전: 골반-어깨-팔꿈치 각도 (팔을 몸에서 얼마나 들었나)
    - 고관절 굴곡: 어깨-골반-무릎 각도
    - 체간 기울기: 골반 중앙 → 어깨 중앙 선이 수직에서 기운 정도 (0° = 완전한 직립)
    """
    p = _points(landmarks, aspect)
    ls, rs = p["left_shoulder"], p["right_shoulder"]
    le, re = p["left_elbow"], p["right_elbow"]
    lw, rw = p["left_wrist"], p["right_wrist"]
    lh, rh = p["left_hip"], p["right_hip"]
    lk, rk = p["left_knee"], p["right_knee"]
    la, ra = p["left_ankle"], p["right_ankle"]

    mid_shoulder = (ls + rs) / 2
    mid_hip = (lh + rh) / 2

    return {
        "left_elbow":               round(_angle_at(ls, le, lw), 1),
        "right_elbow":              round(_angle_at(rs, re, rw), 1),
        "left_knee":                round(_angle_at(lh, lk, la), 1),
        "right_knee":               round(_angle_at(rh, rk, ra), 1),
        "left_shoulder_abduction":  round(_angle_at(lh, ls, le), 1),
        "right_shoulder_abduction": round(_angle_at(rh, rs, re), 1),
        "left_hip_flexion":         round(_angle_at(ls, lh, lk), 1),
        "right_hip_flexion":        round(_angle_at(rs, rh, rk), 1),
        "trunk_lean":               round(_angle_from_vertical(mid_hip, mid_shoulder), 1),
    }


def compute_all_metrics(landmarks, image_height: int = 1, image_width: int = 1) -> dict:
    """모든 지표를 한 번에 계산해서 dict로 돌려줌 (코치 노트 모듈에 그대로 넘김).

    image_height, image_width: 사진의 실제 픽셀 크기. 넘기지 않으면 정사각형 사진으로 봐요.
    """
    aspect = image_height / image_width
    return {
        "vtaper_ratio": vtaper_ratio(landmarks, aspect),
        "symmetry":     symmetry_score(landmarks, aspect),
        "joint_angles": joint_angles(landmarks, aspect),
    }
