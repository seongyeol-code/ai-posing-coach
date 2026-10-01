# src/pose.py
# MediaPipe Tasks API(새 방식)로 사진에서 관절 33개를 찾는 모듈
#
# [바뀐 이유] MediaPipe 0.10.30 이후 버전에서 옛날 방식인 mp.solutions가 삭제됐어요.
# 그래서 Google이 권장하는 새 방식(mp.tasks.vision.PoseLandmarker)으로 바꿨어요.
# 새 방식은 모델 파일(.task)이 따로 필요해서, 처음 실행할 때 한 번 자동으로 내려받아요.

import os
import tempfile
import urllib.request

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# 포즈 모델 (full = 정확도와 속도의 중간. 더 정확한 heavy도 있지만 무겁고 느려요)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODEL_NAME = "pose_landmarker_full.task"

# 프로젝트 최상위 폴더 (이 파일 위치 src/ 의 한 단계 위)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 관절끼리 선으로 잇는 목록 (예: 11번 어깨 - 13번 팔꿈치)
# 얼굴 점(0~10번)은 포징 분석에 쓰지 않고, 후면 사진에선 뒤통수에 뭉쳐 보기 흉해서 몸(11번~)만 그려요.
BODY_CONNECTIONS = [
    (c.start, c.end)
    for c in vision.PoseLandmarksConnections.POSE_LANDMARKS
    if c.start >= 11 and c.end >= 11
]
# 손가락 끝 점(17~22번)은 너무 촘촘해서 선만 잇고 점은 생략
BODY_POINTS = [i for i in range(11, 33) if i not in range(17, 23)]
LEFT_SIDE = {11, 13, 15, 23, 25, 27, 29, 31}  # 선수 기준 왼쪽 관절

# 뼈대 색 (RGB) — 화면 디자인과 맞춘 무대 조명 골드 / 아이보리
LINE_COLOR = (214, 170, 72)
LEFT_COLOR = (240, 196, 92)
RIGHT_COLOR = (250, 244, 230)


def get_model_path():
    """모델 파일 경로를 돌려줌. 없으면 인터넷에서 한 번 내려받음."""
    # 1순위: 프로젝트 안 models/ 폴더 (내 PC에서 실행할 때)
    # 2순위: 임시 폴더 (Streamlit Cloud처럼 프로젝트 폴더에 쓰기가 막혔을 때)
    candidates = [
        os.path.join(PROJECT_ROOT, "models"),
        tempfile.gettempdir(),
    ]
    # 이미 받아둔 파일이 있으면 그걸 사용
    for folder in candidates:
        path = os.path.join(folder, MODEL_NAME)
        if os.path.exists(path):
            return path

    # 없으면 내려받기 (약 9MB, 처음 한 번만)
    for folder in candidates:
        try:
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, MODEL_NAME)
            print(f"포즈 모델 내려받는 중... ({path})")
            # 임시 이름으로 받은 뒤 다 받으면 이름을 바꿈
            # → 중간에 끊겨도 깨진 모델 파일이 남아 다음 실행을 망치지 않음
            partial = path + ".part"
            urllib.request.urlretrieve(MODEL_URL, partial)
            os.replace(partial, path)
            return path
        except OSError:
            continue  # 이 폴더에 못 쓰거나 다운로드가 실패하면 다음 후보로
    raise RuntimeError("포즈 모델 파일을 내려받지 못했어요. 인터넷 연결을 확인하세요.")


def create_landmarker():
    """포즈 감지기 만들기 (사진 1장씩 분석하는 IMAGE 모드, CPU 사용)"""
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=get_model_path(),
            delegate=BaseOptions.Delegate.CPU,  # GPU 끄기 (Streamlit Cloud엔 GPU 없음)
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,  # 사진 속 사람 1명만
    )
    return vision.PoseLandmarker.create_from_options(options)


def detect_landmarks(landmarker, rgb):
    """RGB 이미지(numpy 배열)에서 관절 33개 좌표 리스트를 찾아 돌려줌.
    사람을 못 찾으면 None."""
    mp_image = Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return None
    # pose_landmarks는 [사람1, 사람2, ...] 형태 → 첫 번째 사람의 관절 33개
    return result.pose_landmarks[0]


def draw_skeleton(rgb, landmarks):
    """원본 RGB 이미지 위에 몸 관절 점과 뼈대 선을 그린 새 이미지를 돌려줌.
    선수 기준 왼쪽 관절은 골드, 오른쪽 관절은 아이보리로 칠해요."""
    annotated = rgb.copy()
    h, w = annotated.shape[:2]
    # 사진 크기에 맞춰 선 굵기·점 크기 조절
    thick = max(2, round(min(h, w) / 260))
    radius = thick + 3

    def px(i):
        return int(landmarks[i].x * w), int(landmarks[i].y * h)

    for a, b in BODY_CONNECTIONS:
        cv2.line(annotated, px(a), px(b), LINE_COLOR, thick, cv2.LINE_AA)
    for i in BODY_POINTS:
        color = LEFT_COLOR if i in LEFT_SIDE else RIGHT_COLOR
        cv2.circle(annotated, px(i), radius + 2, (20, 16, 10), -1, cv2.LINE_AA)  # 어두운 테두리
        cv2.circle(annotated, px(i), radius, color, -1, cv2.LINE_AA)
    return annotated


def analyze_pose(image_path):
    """사진 파일 경로를 받아 관절을 찾고, 뼈대 그린 사진을 samples/result.jpg로 저장.
    관절 좌표 리스트(landmarks)를 돌려줌. (터미널에서 테스트할 때 사용)"""
    image = cv2.imread(image_path)
    if image is None:
        print("이미지를 찾을 수 없음.")
        return None

    # BGR → RGB 변환 (MediaPipe는 RGB를 씀)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with create_landmarker() as landmarker:
        landmarks = detect_landmarks(landmarker, rgb)

    if landmarks is None:
        print("사람을 감지하지 못했어요. 전신이 잘 나온 사진인지 확인하세요.")
        return None

    print(f"관절 감지 성공! 총 {len(landmarks)}개 랜드마크 검출")
    print(f"왼쪽 어깨: x={landmarks[11].x:.3f}, y={landmarks[11].y:.3f}")
    print(f"오른쪽 어깨: x={landmarks[12].x:.3f}, y={landmarks[12].y:.3f}")
    print(f"왼쪽 엉덩이: x={landmarks[23].x:.3f}, y={landmarks[23].y:.3f}")
    print(f"오른쪽 엉덩이: x={landmarks[24].x:.3f}, y={landmarks[24].y:.3f}")
    print(f"왼쪽 무릎  신뢰도: {landmarks[25].visibility:.2f}")
    print(f"오른쪽 무릎 신뢰도: {landmarks[26].visibility:.2f}")
    print(f"왼쪽 발목  신뢰도: {landmarks[27].visibility:.2f}")
    print(f"오른쪽 발목 신뢰도: {landmarks[28].visibility:.2f}")

    # 뼈대 그려서 저장 (cv2로 저장할 땐 다시 BGR로)
    annotated = draw_skeleton(rgb, landmarks)
    output_path = os.path.join(PROJECT_ROOT, "samples", "result.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print(f"결과 이미지 저장됨: {output_path}")

    return landmarks


# 직접 실행할 때 테스트
#   python src/pose.py                      → samples 폴더의 첫 번째 사진으로 테스트
#   python src/pose.py samples/back-1.jpg   → 지정한 사진으로 테스트
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        samples_dir = os.path.join(PROJECT_ROOT, "samples")
        photos = sorted(
            f for f in os.listdir(samples_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and f != "result.jpg"
        ) if os.path.isdir(samples_dir) else []
        if not photos:
            print("samples 폴더에 테스트할 사진(jpg, png)을 넣거나, 사진 경로를 함께 적어주세요.")
            print("예: python src/pose.py samples/front-1.jpg")
            sys.exit(1)
        target = os.path.join(samples_dir, photos[0])

    print(f"테스트 사진: {target}")
    analyze_pose(target)
