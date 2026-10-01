# src/classify.py
# 사진 속 포즈가 프론트 더블 바이셉스인지 백 더블 바이셉스인지 판별하는 모듈
#
# 판별은 두 단계로 해요.
#   1단계: 더블 바이셉스 자세인가? (양 팔꿈치가 어깨 높이쯤, 손목이 팔꿈치보다 위, 팔이 굽어 있음)
#   2단계: 몸이 카메라를 보고 있나(정면), 등지고 있나(후면)?
# 둘 중 하나라도 확실하지 않으면 분석하지 않고 이유를 알려줘요.

import math
from dataclasses import dataclass, field

# MediaPipe 관절 번호
NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16

# 왼쪽-오른쪽 짝 (후면 사진에서 좌우 라벨을 바로잡을 때 사용)
LR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]

# ---------------------------------------------------------------------------
# 기준값 (실제 사진으로 숫자를 찍어보고 조정하는 값들)
# ---------------------------------------------------------------------------
# 1단계: 더블 바이셉스 자세 조건
ELBOW_HEIGHT_TOLERANCE = 0.35  # 팔꿈치가 어깨보다 (어깨너비 x 이 값)까지 낮아도 허용
MAX_ELBOW_ANGLE = 155          # 팔꿈치 각도가 이보다 크면 팔을 편 것으로 봄 (도)

# 2단계: 정면/후면 신호별 기준
# (front-1, back-1, ramon-test 사진으로 측정: 정면은 코가 귀보다 0.15~0.24, 어깨보다 0.23~0.35 가까웠고
#  후면은 반대로 귀보다 0.13, 어깨보다 0.18 멀었어요. 기준을 그 절반쯤으로 잡았어요.)
HEAD_DEPTH_SCALE = 0.08        # 코가 귀보다 카메라에 이만큼 가까우면 "확실히 정면"
NOSE_DEPTH_SCALE = 0.12        # 코가 어깨보다 카메라에 이만큼 가까우면 "확실히 정면"
MIN_CONFIDENCE = 0.35          # 종합 판정 확신도가 이보다 낮으면 분석 거부

FRONT = "front"
BACK = "back"
POSE_NAMES = {
    FRONT: "프론트 더블 바이셉스",
    BACK: "백 더블 바이셉스",
}
POSE_NAMES_EN = {
    FRONT: "Front Double Biceps",
    BACK: "Back Double Biceps",
}


@dataclass
class PoseVerdict:
    """판별 결과를 담는 상자"""
    ok: bool                      # 분석해도 되는가
    view: str | None = None       # "front" / "back"
    confidence: float = 0.0       # 0~1, 정면/후면 판정 확신도
    reason: str = ""              # 거부된 이유 (ok=False일 때)
    signals: dict = field(default_factory=dict)  # 각 신호의 점수 (+는 정면, -는 후면)

    @property
    def name(self) -> str:
        return POSE_NAMES.get(self.view, "")

    @property
    def name_en(self) -> str:
        return POSE_NAMES_EN.get(self.view, "")


# ---------------------------------------------------------------------------
# 계산 도우미
# ---------------------------------------------------------------------------

@dataclass
class _P:
    """계산용 점 하나 (x, y, z)"""
    x: float
    y: float
    z: float


def _to_points(lm, image_height, image_width):
    """MediaPipe 좌표는 가로·세로를 각각 0~1로 맞춘 값이라, 세로로 긴 사진에선 각도가 찌그러져요.
    그래서 y에 (세로 ÷ 가로) 비율을 곱해 실제 사진 모양대로 바꾼 점 리스트를 만들어요."""
    aspect = image_height / image_width
    return [_P(p.x, p.y * aspect, p.z) for p in lm]


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _angle_at(a, vertex, c):
    """vertex에서 a와 c가 이루는 각도 (벡터 내적 이용, 도 단위)"""
    v1 = (a.x - vertex.x, a.y - vertex.y)
    v2 = (c.x - vertex.x, c.y - vertex.y)
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 * n2 < 1e-9:
        return 180.0
    cos_val = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_val))))


def _squash(value, scale):
    """값을 -1~+1 사이로 부드럽게 눌러줌 (scale만큼 벗어나면 약 ±0.76)"""
    return math.tanh(value / scale)


# ---------------------------------------------------------------------------
# 1단계: 더블 바이셉스 자세인가?
# ---------------------------------------------------------------------------

def check_double_biceps(lm):
    """더블 바이셉스 자세면 (True, ""), 아니면 (False, 이유)"""
    shoulder_w = _dist(lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]) or 1e-6

    for side, (s, e, w) in {
        "왼팔": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
        "오른팔": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    }.items():
        sh, el, wr = lm[s], lm[e], lm[w]
        # 이미지 좌표는 y가 아래로 갈수록 커짐 → 팔꿈치 y가 어깨 y보다 많이 크면 팔이 내려간 것
        if el.y - sh.y > ELBOW_HEIGHT_TOLERANCE * shoulder_w:
            return False, f"{side} 팔꿈치가 어깨 높이보다 많이 내려가 있어요."
        if wr.y > el.y:
            return False, f"{side} 손목이 팔꿈치보다 아래에 있어요."
        if _angle_at(sh, el, wr) > MAX_ELBOW_ANGLE:
            return False, f"{side}이 펴져 있어요. 이두를 수축해 팔을 굽혀주세요."
    return True, ""


# ---------------------------------------------------------------------------
# 2단계: 정면인가, 후면인가?
# ---------------------------------------------------------------------------

def view_signals(lm):
    """정면/후면 신호 3개를 계산. 각 값은 -1(후면)~+1(정면)."""
    shoulder_w = _dist(lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]) or 1e-6
    mid_shoulder_z = (lm[LEFT_SHOULDER].z + lm[RIGHT_SHOULDER].z) / 2

    # 신호 1. 머리 방향: 정면이면 코가 두 귀보다 카메라에 가깝고(z가 더 작음), 후면이면 더 멀어요.
    #         (귀는 뒤에서도 보이기 때문에 "귀가 보이는가"가 아니라 "코가 귀보다 앞에 있나"를 봐요.)
    mid_ear_z = (lm[LEFT_EAR].z + lm[RIGHT_EAR].z) / 2
    head_depth = mid_ear_z - lm[NOSE].z
    s_head = _squash(head_depth, HEAD_DEPTH_SCALE)

    # 신호 2. 코의 깊이(z): 정면이면 코가 어깨보다 카메라에 가까워요(z가 더 작음).
    nose_depth = mid_shoulder_z - lm[NOSE].z
    s_depth = _squash(nose_depth, NOSE_DEPTH_SCALE)

    # 신호 3. 좌우 순서: 정면이면 선수의 왼어깨가 사진 오른쪽(x가 큼)에 와요.
    #         (MediaPipe가 후면을 정면으로 착각하면 틀릴 수 있어서 가중치를 조금 낮게 둠)
    side_order = (lm[LEFT_SHOULDER].x - lm[RIGHT_SHOULDER].x) / shoulder_w
    s_order = _squash(side_order, 0.5)

    return {
        "head_depth": round(head_depth, 3),
        "nose_depth": round(nose_depth, 3),
        "side_order": round(side_order, 3),
        "s_head": round(s_head, 3),
        "s_depth": round(s_depth, 3),
        "s_order": round(s_order, 3),
    }


# 신호별 가중치 (합이 1)
WEIGHTS = {"s_head": 0.35, "s_depth": 0.35, "s_order": 0.3}


def classify_pose(lm, image_height=1, image_width=1) -> PoseVerdict:
    """관절 33개와 사진 크기(픽셀)를 받아 포즈를 판별"""
    lm = _to_points(lm, image_height, image_width)
    is_db, reason = check_double_biceps(lm)
    if not is_db:
        return PoseVerdict(
            ok=False,
            reason="프론트·백 더블 바이셉스 자세가 아니에요. " + reason,
        )

    signals = view_signals(lm)
    score = sum(signals[k] * w for k, w in WEIGHTS.items())  # -1(후면) ~ +1(정면)
    signals["score"] = round(score, 3)
    confidence = abs(score)

    if confidence < MIN_CONFIDENCE:
        return PoseVerdict(
            ok=False,
            confidence=confidence,
            signals=signals,
            reason="정면인지 후면인지 확실하지 않아요. 카메라를 정면 또는 정후면에 두고 다시 찍어주세요.",
        )

    view = FRONT if score > 0 else BACK
    return PoseVerdict(ok=True, view=view, confidence=confidence, signals=signals)


# ---------------------------------------------------------------------------
# 후면 사진 좌우 보정
# ---------------------------------------------------------------------------

def normalize_sides(lm, view):
    """선수 기준 왼쪽/오른쪽이 맞도록 관절 라벨을 정리한 새 리스트를 돌려줌.

    정면 사진: 선수의 왼쪽은 사진 오른쪽(x가 큼)에 있어야 해요.
    후면 사진: 선수의 왼쪽은 사진 왼쪽(x가 작음)에 있어야 해요.
    MediaPipe가 후면을 정면으로 착각해 좌우를 뒤집어 붙였으면 짝끼리 바꿔줘요.
    """
    points = list(lm)
    left_is_on_image_right = points[LEFT_SHOULDER].x > points[RIGHT_SHOULDER].x
    should_be_on_image_right = view == FRONT
    if left_is_on_image_right != should_be_on_image_right:
        for a, b in LR_PAIRS:
            points[a], points[b] = points[b], points[a]
    return points
