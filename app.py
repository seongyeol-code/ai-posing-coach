import urllib.request
from pathlib import Path

import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.framework.formats import landmark_pb2
from PIL import Image

from src.metrics import compute_all_metrics
from src.feedback import generate_feedback

# Tasks API — 0.10.x 공식 권장 경로 (내부 mediapipe.tasks.python 경로 사용 안 함)
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)
_MODEL_PATH = Path("pose_landmarker_heavy.task")


@st.cache_resource(show_spinner=False)
def _load_landmarker():
    if not _MODEL_PATH.exists():
        with st.spinner("포즈 모델을 다운로드하는 중... (최초 1회)"):
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_score=0.5,
    )
    return PoseLandmarker.create_from_options(options)


class _LandmarksAdapter:
    """Tasks API 결과를 metrics.py의 landmarks.landmark[idx] 인터페이스로 변환."""

    class _Seq:
        def __init__(self, lm):
            self._lm = lm

        def __getitem__(self, i):
            return self._lm[i]

    def __init__(self, landmarks):
        self.landmark = self._Seq(landmarks)


st.set_page_config(page_title="AI 포징 코치", page_icon="💪", layout="wide")
st.title("💪 AI 클래식 피지크 포징 코치")
st.caption("사진을 업로드하면 포즈를 분석하고 AI 코칭 피드백을 제공합니다.")

uploaded = st.file_uploader("포즈 사진 업로드", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    rgb = np.array(Image.open(uploaded).convert("RGB"))

    landmarker = _load_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        st.error("포즈 랜드마크를 감지하지 못했습니다. 전신이 잘 보이는 사진을 사용해 주세요.")
        st.stop()

    # Tasks API 결과(list[NormalizedLandmark])를 proto로 변환하여 drawing_utils에 전달
    annotated = rgb.copy()
    proto = landmark_pb2.NormalizedLandmarkList()
    proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
        for lm in result.pose_landmarks[0]
    ])
    mp.solutions.drawing_utils.draw_landmarks(
        annotated,
        proto,
        mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
    )

    col_img, col_metrics = st.columns([1, 1])

    with col_img:
        st.subheader("포즈 분석")
        st.image(annotated, use_container_width=True)

    metrics = compute_all_metrics(_LandmarksAdapter(result.pose_landmarks[0]))

    with col_metrics:
        st.subheader("측정 지표")

        vtaper = metrics["vtaper_ratio"]
        vtaper_color = "normal" if 1.4 <= vtaper <= 1.6 else "inverse"
        st.metric(
            label="V-테이퍼 비율 (어깨 / 골반)",
            value=f"{vtaper:.3f}",
            delta="이상적 범위 1.4–1.6" if 1.4 <= vtaper <= 1.6 else f"{'높음' if vtaper > 1.6 else '낮음'} (목표: 1.4–1.6)",
            delta_color=vtaper_color,
        )

        st.divider()
        st.markdown("**좌우 대칭성 점수** (100 = 완벽)")

        sym = metrics["symmetry"]
        sym_labels = {
            "shoulder_height": "어깨 높이",
            "hip_height": "골반 높이",
            "arm_length": "팔 길이",
            "leg_length": "다리 길이",
            "elbow_angle": "팔꿈치 각도",
            "knee_angle": "무릎 각도",
            "overall": "전체 대칭",
        }
        for key, label in sym_labels.items():
            score = sym[key]
            color = "green" if score >= 90 else "orange" if score >= 75 else "red"
            st.markdown(
                f"- {label}: <span style='color:{color}; font-weight:bold'>{score:.1f}</span>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("**관절 각도 (°)**")

        angles = metrics["joint_angles"]
        angle_labels = {
            "left_elbow": "왼쪽 팔꿈치",
            "right_elbow": "오른쪽 팔꿈치",
            "left_knee": "왼쪽 무릎",
            "right_knee": "오른쪽 무릎",
            "left_shoulder_abduction": "왼쪽 어깨 외전",
            "right_shoulder_abduction": "오른쪽 어깨 외전",
            "left_hip_flexion": "왼쪽 고관절 굴곡",
            "right_hip_flexion": "오른쪽 고관절 굴곡",
            "trunk_lean": "체간 기울기",
        }
        for key, label in angle_labels.items():
            st.markdown(f"- {label}: **{angles[key]:.1f}°**")

    st.divider()
    st.subheader("AI 코칭 피드백")

    with st.spinner("Claude AI가 피드백을 생성하는 중..."):
        try:
            feedback = generate_feedback(metrics)
            st.markdown(feedback)
        except Exception as e:
            st.error(f"피드백 생성 중 오류가 발생했습니다: {e}")
