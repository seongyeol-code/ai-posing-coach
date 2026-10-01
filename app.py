# app.py
# Streamlit 화면: 사진 업로드 → 관절 감지 → 포즈 판별(프론트/백) → 지표 계산 → 코치 노트

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# 관절 감지는 src/pose.py 한 곳에서만 처리 (MediaPipe Tasks API)
from src.pose import create_landmarker, detect_landmarks, draw_skeleton
from src.classify import classify_pose, normalize_sides, FRONT
from src.metrics import compute_all_metrics
from src.feedback import generate_feedback
from src import ui


# 포즈 감지기는 무거우니 앱이 켜질 때 한 번만 만들고 재사용
@st.cache_resource(show_spinner="포즈 모델 준비 중...")
def _load_landmarker():
    return create_landmarker()


# 같은 사진을 다시 볼 때 Claude API를 또 부르지 않도록 결과를 기억해둠
@st.cache_data(show_spinner=False)
def _cached_feedback(metrics, view):
    return generate_feedback(metrics, view)


st.set_page_config(page_title="AI Posing Coach", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)
st.markdown(ui.header_html(), unsafe_allow_html=True)

uploaded = st.file_uploader(
    "프론트 또는 백 더블 바이셉스 사진 (jpg, png)",
    type=["jpg", "jpeg", "png"],
)

if uploaded is None:
    st.markdown(ui.empty_html(), unsafe_allow_html=True)
    st.stop()

# 폰 사진은 회전 정보(EXIF)가 따로 저장돼 있어서, 실제 방향으로 돌려준 뒤 분석
image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
rgb = np.array(image)
height, width = rgb.shape[:2]

landmarks = detect_landmarks(_load_landmarker(), rgb)
if landmarks is None:
    st.markdown(
        ui.reject_html("사진에서 사람을 찾지 못했어요. 전신이 밝게 나온 사진인지 확인해주세요."),
        unsafe_allow_html=True,
    )
    st.stop()

verdict = classify_pose(landmarks, height, width)

if not verdict.ok:
    # 분석 거부: 뼈대 사진과 이유만 보여주고 끝
    col_photo, col_msg = st.columns([5, 7], gap="large")
    with col_photo:
        st.markdown(ui.photo_html(draw_skeleton(rgb, landmarks)), unsafe_allow_html=True)
    with col_msg:
        st.markdown(ui.reject_html(verdict.reason), unsafe_allow_html=True)
    st.stop()

# 후면 사진이면 선수 기준 왼쪽/오른쪽이 맞도록 라벨 정리
landmarks = normalize_sides(landmarks, verdict.view)
metrics = compute_all_metrics(landmarks, height, width)  # 사진 비율을 넘겨 각도 왜곡 방지
pose_number = 1 if verdict.view == FRONT else 3

# ── 1. 사진 + 판정 + 점수판 ─────────────────────────────
col_photo, col_score = st.columns([5, 7], gap="large")
with col_photo:
    st.markdown(ui.photo_html(draw_skeleton(rgb, landmarks)), unsafe_allow_html=True)
with col_score:
    st.markdown(
        ui.verdict_html(verdict, pose_number)
        + ui.board_html(metrics["symmetry"]["overall"], metrics["vtaper_ratio"]),
        unsafe_allow_html=True,
    )
    with st.expander("판정 근거 보기"):
        s = verdict.signals
        st.markdown(
            f"""
- 머리 방향 (코가 귀보다 카메라에 가까운 정도): **{s['head_depth']}** → 신호 {s['s_head']:+.2f}
- 몸 방향 (코가 어깨보다 카메라에 가까운 정도): **{s['nose_depth']}** → 신호 {s['s_depth']:+.2f}
- 좌우 순서 (왼어깨가 사진 오른쪽에 있는 정도): **{s['side_order']}** → 신호 {s['s_order']:+.2f}
- 종합 점수: **{s['score']:+.2f}** (+는 정면, -는 후면)
"""
        )

# ── 2. 좌우 대칭 ─────────────────────────────────────
st.markdown(ui.symmetry_html(metrics["symmetry"]), unsafe_allow_html=True)

# ── 3. 관절 각도 ─────────────────────────────────────
st.markdown(ui.angles_html(metrics["joint_angles"]), unsafe_allow_html=True)

# ── 4. 코치 노트 (AI 피드백) ───────────────────────────
st.markdown(ui.coach_title_html(), unsafe_allow_html=True)
with st.container(key="coach"):
    with st.spinner("코치 노트를 작성하는 중..."):
        try:
            st.markdown(_cached_feedback(metrics, verdict.view))
        except Exception as e:
            st.error(f"코치 노트를 만들지 못했어요: {e}")
