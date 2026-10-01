"""src/classify.py 단위 테스트.

실제 사진 대신 관절 좌표를 직접 만들어서, 판별 규칙이 의도대로 동작하는지 확인해요.
"""

import os
import sys
import unittest
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.classify import (
    BACK,
    FRONT,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    classify_pose,
    normalize_sides,
)


@dataclass
class _LM:
    x: float
    y: float
    z: float = 0.0


def _double_biceps(view: str):
    """정사각형 사진 속 더블 바이셉스 자세 관절 33개를 만듦.
    view="front"면 카메라를 보는 정면, "back"이면 등을 보이는 후면."""
    lm = [_LM(0.5, 0.5) for _ in range(33)]
    # 정면이면 선수의 왼쪽이 사진 오른쪽(x가 큼), 후면이면 사진 왼쪽
    sign = 1 if view == FRONT else -1
    left = lambda x: 0.5 + sign * x
    right = lambda x: 0.5 - sign * x

    lm[11], lm[12] = _LM(left(0.12), 0.30), _LM(right(0.12), 0.30)   # 어깨
    lm[13], lm[14] = _LM(left(0.25), 0.28), _LM(right(0.25), 0.28)   # 팔꿈치 (어깨 높이)
    lm[15], lm[16] = _LM(left(0.22), 0.15), _LM(right(0.22), 0.15)   # 손목 (팔꿈치 위)
    lm[23], lm[24] = _LM(left(0.07), 0.55), _LM(right(0.07), 0.55)   # 골반
    lm[25], lm[26] = _LM(left(0.08), 0.72), _LM(right(0.08), 0.72)   # 무릎
    lm[27], lm[28] = _LM(left(0.08), 0.90), _LM(right(0.08), 0.90)   # 발목

    # 귀는 앞뒤 어디서든 보이고, 코의 깊이(z)만 달라짐
    lm[7], lm[8] = _LM(left(0.04), 0.19, 0.0), _LM(right(0.04), 0.19, 0.0)
    if view == FRONT:
        lm[0] = _LM(0.5, 0.20, -0.30)   # 코가 귀·어깨보다 카메라에 가까움
    else:
        lm[0] = _LM(0.5, 0.20, 0.20)    # 코가 귀·어깨보다 멀리
    return lm


class TestClassify(unittest.TestCase):
    def test_front(self):
        v = classify_pose(_double_biceps(FRONT), 1000, 1000)
        self.assertTrue(v.ok)
        self.assertEqual(v.view, FRONT)

    def test_back(self):
        v = classify_pose(_double_biceps(BACK), 1000, 1000)
        self.assertTrue(v.ok)
        self.assertEqual(v.view, BACK)

    def test_arms_down_is_rejected(self):
        lm = _double_biceps(FRONT)
        # 팔꿈치와 손목을 골반 높이로 내림
        lm[13].y = lm[14].y = 0.55
        lm[15].y = lm[16].y = 0.65
        v = classify_pose(lm, 1000, 1000)
        self.assertFalse(v.ok)
        self.assertIn("더블 바이셉스 자세가 아니에요", v.reason)

    def test_back_photo_labeled_as_front_gets_swapped(self):
        # MediaPipe가 후면 사진에 정면처럼 좌우 라벨을 붙인 경우
        lm = _double_biceps(FRONT)
        fixed = normalize_sides(lm, BACK)
        # 후면에선 선수 왼쪽이 사진 왼쪽(x가 작음)에 있어야 함
        self.assertLess(fixed[LEFT_SHOULDER].x, fixed[RIGHT_SHOULDER].x)

    def test_front_labels_unchanged(self):
        lm = _double_biceps(FRONT)
        fixed = normalize_sides(lm, FRONT)
        self.assertIs(fixed[LEFT_SHOULDER], lm[LEFT_SHOULDER])


if __name__ == "__main__":
    unittest.main()
