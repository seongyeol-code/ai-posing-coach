# src/feedback.py
# 계산된 지표를 Claude API에 보내 코치 노트(한국어 피드백)를 만드는 모듈
# API 키는 .env 파일의 ANTHROPIC_API_KEY에서 읽어요 (Streamlit Cloud에선 Secrets에 설정).

import anthropic
from dotenv import load_dotenv

load_dotenv()

_SYSTEM_PROMPT = """당신은 클래식 피지크 포징 전문 코치입니다.
선수의 포즈 분석 데이터를 바탕으로 구체적이고 실용적인 한국어 피드백을 제공합니다.

클래식 피지크 심사 기준:
- V-테이퍼 비율 (어깨 너비 / 엉덩이 너비): 이상적 범위 1.4–1.6
- 좌우 대칭성: 점수가 높을수록 좋음 (0–100, 100 = 완벽한 대칭)
- 관절 각도: 포즈에 따른 적절한 굴곡 및 신전
- 체간 자세: 수직에 가까울수록 좋음 (trunk_lean 0° = 완벽한 직립)

피드백 형식:
1. 전반적인 포즈 평가 (강점 위주)
2. V-테이퍼 분석
3. 대칭성 분석
4. 관절 각도 및 포징 디테일
5. 개선을 위한 구체적인 조언

간결하고 명확하게 작성하되, 선수에게 도움이 되는 실질적인 조언을 포함하세요.
이모지는 사용하지 마세요."""

# 포즈별로 코치가 특히 봐야 할 점
_POSE_FOCUS = {
    "front": "정면 포즈입니다. 팔꿈치 높이와 좌우 팔 각도의 균형, 허리 조임, 다리 자세를 중심으로 봐 주세요.",
    "back": (
        "후면 포즈입니다. 등을 보이고 선 자세이므로 팔 높이와 좌우 팔 각도의 균형, 등 넓이를 보여주는 어깨 라인, "
        "한쪽 발을 뒤로 빼 종아리를 보여주는 하체 자세를 중심으로 봐 주세요. "
        "좌우는 선수 본인 기준입니다."
    ),
}


def generate_feedback(metrics: dict, view: str | None = None) -> str:
    """지표를 Claude API에 보내 한국어 코치 노트를 받아옴.

    Args:
        metrics: compute_all_metrics()의 결과 (vtaper_ratio, symmetry, joint_angles)
        view: "front"(프론트 더블 바이셉스) 또는 "back"(백 더블 바이셉스)

    Returns:
        클래식 피지크 코칭 피드백 (마크다운 문자열)
    """
    client = anthropic.Anthropic()

    vtaper = metrics["vtaper_ratio"]
    symmetry = metrics["symmetry"]
    angles = metrics["joint_angles"]

    pose_line = _POSE_FOCUS.get(view, "")

    user_message = f"""다음은 포징 분석 결과입니다. 이를 바탕으로 클래식 피지크 코칭 피드백을 한국어로 작성해 주세요.
{pose_line}

**V-테이퍼 비율**: {vtaper:.3f}
(이상적 범위: 1.4–1.6 / 값이 높을수록 V자 체형이 뚜렷함)

**대칭성 점수** (0–100, 100=완벽):
- 어깨 높이 대칭: {symmetry['shoulder_height']:.1f}
- 골반 높이 대칭: {symmetry['hip_height']:.1f}
- 팔 길이 대칭: {symmetry['arm_length']:.1f}
- 다리 길이 대칭: {symmetry['leg_length']:.1f}
- 팔꿈치 각도 대칭: {symmetry['elbow_angle']:.1f}
- 무릎 각도 대칭: {symmetry['knee_angle']:.1f}
- **전체 대칭 점수**: {symmetry['overall']:.1f}

**관절 각도**:
- 왼쪽 팔꿈치: {angles['left_elbow']:.1f}°
- 오른쪽 팔꿈치: {angles['right_elbow']:.1f}°
- 왼쪽 무릎: {angles['left_knee']:.1f}°
- 오른쪽 무릎: {angles['right_knee']:.1f}°
- 왼쪽 어깨 외전: {angles['left_shoulder_abduction']:.1f}°
- 오른쪽 어깨 외전: {angles['right_shoulder_abduction']:.1f}°
- 왼쪽 고관절 굴곡: {angles['left_hip_flexion']:.1f}°
- 오른쪽 고관절 굴곡: {angles['right_hip_flexion']:.1f}°
- 체간 기울기: {angles['trunk_lean']:.1f}° (0°=완벽한 직립)"""

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text
