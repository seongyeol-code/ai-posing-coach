# src/ui.py
# 화면 디자인 모듈 — "대회 무대" 콘셉트
#
# 검정 무대 위에 조명이 떨어지는 느낌으로, 포인트 색은 트로피 골드 하나만 써요.
# 점수 색은 메달에서 따왔어요: 90점 이상 골드, 75점 이상 실버, 그 아래는 무대 커튼 레드(개선 필요).
# app.py는 "무엇을 보여줄지"만, 이 파일은 "어떻게 보여줄지"만 담당해요.

import base64
import html

import cv2

# ---------------------------------------------------------------------------
# 색 (디자인 토큰)
# ---------------------------------------------------------------------------
STAGE = "#000000"      # 무대 바닥
SPOT = "#2A1E10"       # 조명이 닿은 무대 (배경 그라데이션 중심)
PANEL = "#120E09"      # 패널 배경
GOLD = "#D6AA48"       # 트로피 골드 (유일한 강조색)
IVORY = "#F4ECDC"      # 본문 글자
BRONZE = "#9C8466"     # 보조 글자
SILVER = "#C9CCD1"     # 75~90점
RED = "#D0533F"        # 75점 미만 (개선 필요)
LINE = "#2E2418"       # 구분선

# ---------------------------------------------------------------------------
# 전체 스타일 (CSS)
# ---------------------------------------------------------------------------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@600;800;900&family=IBM+Plex+Sans+KR:wght@400;500;700&display=swap');

:root {{
  --stage: {STAGE}; --spot: {SPOT}; --panel: {PANEL}; --gold: {GOLD};
  --ivory: {IVORY}; --bronze: {BRONZE}; --silver: {SILVER}; --red: {RED}; --line: {LINE};
  --display: 'Big Shoulders Display', 'IBM Plex Sans KR', sans-serif;
  --body: 'IBM Plex Sans KR', sans-serif;
}}

/* 무대 배경: 위에서 조명이 떨어지는 느낌 */
.stApp {{
  background: radial-gradient(ellipse 70% 55% at 50% -5%, var(--spot) 0%, var(--stage) 70%) fixed, var(--stage);
  color: var(--ivory);
  font-family: var(--body);
  word-break: keep-all;  /* 한글 단어가 중간에서 끊기지 않게 */
}}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }}
.stApp p, .stApp li, .stApp label {{ font-family: var(--body); }}
/* Streamlit 아이콘은 전용 아이콘 글꼴을 써야 글자 대신 아이콘으로 보여요 */
[data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; }}

/* 머리글 */
.pc-head {{ display: flex; justify-content: space-between; align-items: flex-end;
  gap: 1rem; flex-wrap: wrap; border-bottom: 1px solid var(--line); padding-bottom: .9rem; margin-bottom: 1.4rem; }}
.pc-mark {{ font-family: var(--display); font-weight: 900; font-size: 2.1rem; line-height: 1;
  color: var(--gold); letter-spacing: .01em; }}
.pc-sub {{ color: var(--bronze); font-size: .9rem; }}

/* 업로드 박스 */
[data-testid="stFileUploader"] section {{ background: var(--panel); border: 1px dashed #4A3A24; border-radius: 6px; }}
[data-testid="stFileUploader"] section:hover {{ border-color: var(--gold); }}
[data-testid="stFileUploader"] button {{ border-color: var(--gold); color: var(--gold); }}

/* 빈 화면 안내 */
.pc-empty {{ text-align: center; padding: 3.5rem 1rem 1rem; }}
.pc-empty h2 {{ font-family: var(--display); font-weight: 800; font-size: clamp(2.6rem, 7vw, 4.6rem);
  line-height: .95; margin: 0 0 1rem; color: var(--ivory); }}
.pc-empty p {{ color: var(--bronze); margin: .2rem 0; }}
.pc-poses {{ display: flex; justify-content: center; gap: 2.5rem; margin-top: 2rem; flex-wrap: wrap; }}
.pc-poses div {{ text-align: left; border-left: 2px solid var(--gold); padding-left: .8rem; }}
.pc-poses b {{ display: block; font-family: var(--display); font-weight: 800; font-size: 1.5rem; color: var(--gold); }}
.pc-poses span {{ color: var(--bronze); font-size: .88rem; }}

/* 사진 */
.pc-photo {{ position: relative; border: 1px solid var(--line); border-radius: 4px; overflow: hidden;
  box-shadow: 0 30px 80px -30px rgba(214,170,72,.35); }}
.pc-photo img {{ display: block; width: 100%; height: auto; }}
.pc-legend {{ display: flex; gap: 1.1rem; margin-top: .6rem; color: var(--bronze); font-size: .85rem; }}
.pc-legend i {{ display: inline-block; width: .7rem; height: .7rem; border-radius: 50%;
  margin-right: .35rem; vertical-align: -1px; }}

/* 판정 (이 화면의 주인공) */
.pc-verdict-meta {{ color: var(--bronze); font-size: .92rem; margin-bottom: .2rem; }}
.pc-verdict-meta b {{ color: var(--ivory); font-weight: 500; }}
.pc-pose {{ font-family: var(--display); font-weight: 900; color: var(--gold);
  font-size: clamp(3.2rem, 7.4vw, 6.2rem); line-height: .86; margin: .2rem 0 .7rem;
  text-transform: uppercase; }}
.pc-pose-ko {{ color: var(--ivory); font-size: 1.05rem; margin-bottom: 1.6rem; }}

/* 점수판 */
.pc-board {{ display: grid; grid-template-columns: 1fr 1fr; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }}
.pc-cell {{ padding: 1.1rem 0; }}
.pc-cell + .pc-cell {{ border-left: 1px solid var(--line); padding-left: 1.4rem; }}
.pc-cell-label {{ color: var(--bronze); font-size: .9rem; }}
.pc-big {{ font-family: var(--display); font-weight: 800; font-size: 3.6rem; line-height: 1; margin-top: .25rem; }}
.pc-big small {{ font-size: 1.2rem; color: var(--bronze); font-weight: 600; margin-left: .25rem; }}
.pc-note {{ color: var(--bronze); font-size: .85rem; margin-top: .35rem; }}

/* V-테이퍼 게이지: 1.0~2.0 눈금 위에 이상 범위(1.4~1.6)를 띠로 표시 */
.pc-gauge {{ position: relative; height: 8px; background: #2A2117; border-radius: 4px; margin-top: .8rem; }}
.pc-gauge .band {{ position: absolute; top: 0; bottom: 0; background: rgba(214,170,72,.35); border-radius: 4px; }}
.pc-gauge .mark {{ position: absolute; top: -5px; width: 3px; height: 18px; background: var(--ivory); border-radius: 2px; }}
.pc-gauge-scale {{ display: flex; justify-content: space-between; color: var(--bronze); font-size: .75rem; margin-top: .35rem; }}

/* 섹션 제목 */
.pc-h {{ font-family: var(--display); font-weight: 800; font-size: 1.9rem; color: var(--ivory);
  margin: 2.6rem 0 .2rem; line-height: 1; }}
.pc-h-desc {{ color: var(--bronze); font-size: .9rem; margin-bottom: 1rem; }}

/* 대칭 막대 */
.pc-bars {{ display: grid; grid-template-columns: 1fr 1fr; column-gap: 2.4rem; row-gap: .9rem; }}
.pc-bar-top {{ display: flex; justify-content: space-between; font-size: .95rem; margin-bottom: .35rem; }}
.pc-bar-top b {{ font-family: var(--display); font-weight: 800; font-size: 1.35rem; line-height: 1; }}
.pc-track {{ height: 6px; background: #241C12; border-radius: 3px; overflow: hidden; }}
.pc-fill {{ height: 100%; border-radius: 3px; transform-origin: left; animation: pc-grow .9s cubic-bezier(.2,.7,.2,1) both; }}
@keyframes pc-grow {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
@media (prefers-reduced-motion: reduce) {{ .pc-fill {{ animation: none; }} }}
.pc-key {{ display: flex; gap: 1.2rem; color: var(--bronze); font-size: .82rem; margin-top: 1rem; flex-wrap: wrap; }}
.pc-key i {{ display: inline-block; width: 1.1rem; height: 4px; border-radius: 2px; margin-right: .35rem; vertical-align: middle; }}

/* 관절 각도 표: 왼쪽 | 부위 | 오른쪽 | 차이 */
.pc-table {{ width: 100%; border-collapse: collapse; font-size: .98rem; border: none; table-layout: fixed; }}
.pc-table th, .pc-table td {{ border-left: none !important; border-right: none !important; border-top: none; }}
.pc-table th:nth-child(4), .pc-table td.diff {{ width: 34%; }}
.pc-table th {{ color: var(--bronze); font-weight: 500; font-size: .85rem; text-align: center;
  padding: .5rem; border-bottom: 1px solid var(--line); }}
.pc-table td {{ padding: .7rem .5rem; border-bottom: 1px solid var(--line); text-align: center; }}
.pc-table td.part {{ color: var(--ivory); }}
.pc-table td.num {{ font-family: var(--display); font-weight: 700; font-size: 1.4rem; }}
.pc-table td.diff {{ color: var(--bronze); }}
.pc-table td.diff.warn {{ color: var(--red); }}

/* 코치 노트 (AI 피드백) */
.st-key-coach {{ background: var(--panel); border-left: 3px solid var(--gold); border-radius: 0 6px 6px 0;
  padding: 1.4rem 1.8rem; }}
.st-key-coach h1, .st-key-coach h2, .st-key-coach h3, .st-key-coach h4 {{
  font-family: var(--body); color: var(--gold); font-size: 1.08rem; font-weight: 700; margin-top: 1.1rem; }}
.st-key-coach p, .st-key-coach li {{ color: var(--ivory); line-height: 1.75; max-width: 72ch; }}
.st-key-coach strong {{ color: var(--ivory); }}

/* 분석 거부 */
.pc-reject {{ border: 1px solid #5A2A20; background: #1A0C08; border-radius: 6px; padding: 1.6rem 1.8rem; }}
.pc-reject h3 {{ font-family: var(--display); font-weight: 800; font-size: 2.2rem; color: var(--red); margin: 0 0 .6rem; line-height: 1; }}
.pc-reject p {{ color: var(--ivory); margin: 0 0 1rem; }}
.pc-reject ul {{ color: var(--bronze); margin: 0; padding-left: 1.1rem; }}

/* 판정 근거 펼치기 */
[data-testid="stExpander"] details {{ border-color: var(--line); background: transparent; }}
[data-testid="stExpander"] summary {{ color: var(--bronze); }}

@media (max-width: 720px) {{
  .pc-bars {{ grid-template-columns: 1fr; }}
  .pc-board {{ grid-template-columns: 1fr; }}
  .pc-cell + .pc-cell {{ border-left: none; border-top: 1px solid var(--line); padding-left: 0; }}
  .st-key-coach {{ padding: 1.1rem 1.1rem; }}
  .pc-table {{ font-size: .9rem; }}
}}
</style>
"""


# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------

def score_color(score):
    """메달 색: 90 이상 골드, 75 이상 실버, 그 아래 레드"""
    if score >= 90:
        return GOLD
    if score >= 75:
        return SILVER
    return RED


def image_to_html(rgb, max_side=1100):
    """RGB 이미지를 HTML <img>로 넣을 수 있게 base64 문자열로 바꿈"""
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    data = base64.b64encode(buf.tobytes()).decode()
    return f'<img src="data:image/jpeg;base64,{data}" alt="관절 뼈대가 그려진 포즈 사진">'


# ---------------------------------------------------------------------------
# 화면 조각들 (HTML 문자열을 돌려줌)
# ---------------------------------------------------------------------------

def header_html():
    return """
<div class="pc-head">
  <div class="pc-mark">AI Posing Coach</div>
  <div class="pc-sub">IFBB 클래식 피지크 규정 포즈 1번 프론트, 3번 백 더블 바이셉스 분석</div>
</div>"""


def empty_html():
    return """
<div class="pc-empty">
  <h2>Hit the pose.</h2>
  <p>더블 바이셉스 사진을 올리면 정면인지 후면인지 먼저 판정한 뒤 분석해요.</p>
  <p>머리부터 발끝까지 전신이 나오고, 카메라가 정면 또는 정후면에 있는 사진이 가장 정확해요.</p>
  <div class="pc-poses">
    <div><b>Front Double Biceps</b><span>규정 포즈 1번, 카메라를 보고 서기</span></div>
    <div><b>Back Double Biceps</b><span>규정 포즈 3번, 카메라에 등을 보이고 서기</span></div>
  </div>
</div>"""


def photo_html(rgb):
    return f"""
<div class="pc-photo">{image_to_html(rgb)}</div>
<div class="pc-legend"><span><i style="background:{GOLD}"></i>선수 왼쪽</span>
<span><i style="background:{IVORY}"></i>선수 오른쪽</span></div>"""


def verdict_html(verdict, number):
    """판정 결과 + 점수판 윗부분"""
    return f"""
<div class="pc-verdict-meta">판정 확신도 <b>{verdict.confidence * 100:.0f}%</b></div>
<div class="pc-pose">{html.escape(verdict.name_en)}</div>
<div class="pc-pose-ko">{html.escape(verdict.name)}, IFBB 클래식 피지크 규정 포즈 {number}번</div>"""


def board_html(overall, vtaper):
    """점수판: 전체 대칭 점수 + V-테이퍼 게이지"""
    lo, hi = 1.0, 2.0  # 게이지 눈금 범위
    pct = lambda v: max(0.0, min(100.0, (v - lo) / (hi - lo) * 100))
    in_range = 1.4 <= vtaper <= 1.6
    if in_range:
        v_note = "이상 범위(1.4~1.6) 안"
    elif vtaper > 1.6:
        v_note = "이상 범위(1.4~1.6)보다 높음"
    else:
        v_note = "이상 범위(1.4~1.6)보다 낮음"
    v_color = GOLD if in_range else IVORY
    return f"""
<div class="pc-board">
  <div class="pc-cell">
    <div class="pc-cell-label">전체 좌우 대칭</div>
    <div class="pc-big" style="color:{score_color(overall)}">{overall:.1f}<small>/ 100</small></div>
    <div class="pc-note">아래 6개 항목의 평균</div>
  </div>
  <div class="pc-cell">
    <div class="pc-cell-label">V-테이퍼 비율 (어깨 ÷ 골반)</div>
    <div class="pc-big" style="color:{v_color}">{vtaper:.3f}</div>
    <div class="pc-gauge">
      <div class="band" style="left:{pct(1.4)}%; width:{pct(1.6) - pct(1.4)}%"></div>
      <div class="mark" style="left:calc({pct(vtaper)}% - 1px)"></div>
    </div>
    <div class="pc-gauge-scale"><span>1.0</span><span>1.5</span><span>2.0</span></div>
    <div class="pc-note">{v_note}</div>
  </div>
</div>"""


SYMMETRY_LABELS = {
    "shoulder_height": "어깨 높이",
    "hip_height": "골반 높이",
    "arm_length": "팔 길이",
    "leg_length": "다리 길이",
    "elbow_angle": "팔꿈치 각도",
    "knee_angle": "무릎 각도",
}


def symmetry_html(sym):
    rows = []
    for i, (key, label) in enumerate(SYMMETRY_LABELS.items()):
        score = sym[key]
        color = score_color(score)
        rows.append(f"""
<div>
  <div class="pc-bar-top"><span>{label}</span><b style="color:{color}">{score:.1f}</b></div>
  <div class="pc-track"><div class="pc-fill" style="width:{score}%; background:{color}; animation-delay:{i * 60}ms"></div></div>
</div>""")
    return f"""
<div class="pc-h">좌우 대칭</div>
<div class="pc-h-desc">100점이면 왼쪽과 오른쪽이 완전히 같아요.</div>
<div class="pc-bars">{''.join(rows)}</div>
<div class="pc-key"><span><i style="background:{GOLD}"></i>90 이상</span>
<span><i style="background:{SILVER}"></i>75~90</span>
<span><i style="background:{RED}"></i>75 미만, 교정 필요</span></div>"""


ANGLE_PAIRS = [
    ("팔꿈치", "left_elbow", "right_elbow"),
    ("무릎", "left_knee", "right_knee"),
    ("어깨 외전", "left_shoulder_abduction", "right_shoulder_abduction"),
    ("고관절 굴곡", "left_hip_flexion", "right_hip_flexion"),
]
ANGLE_WARN_DIFF = 10  # 좌우 차이가 이 각도 이상이면 빨간색으로 표시


def angles_html(angles):
    rows = []
    for label, lk, rk in ANGLE_PAIRS:
        l, r = angles[lk], angles[rk]
        diff = abs(l - r)
        warn = " warn" if diff >= ANGLE_WARN_DIFF else ""
        rows.append(
            f'<tr><td class="num">{l:.1f}°</td><td class="part">{label}</td>'
            f'<td class="num">{r:.1f}°</td><td class="diff{warn}">{diff:.1f}°</td></tr>'
        )
    trunk = angles["trunk_lean"]
    rows.append(
        f'<tr><td class="num" colspan="3">{trunk:.1f}°</td>'
        f'<td class="diff">체간 기울기, 0°가 완전한 직립</td></tr>'
    )
    return f"""
<div class="pc-h">관절 각도</div>
<div class="pc-h-desc">선수 기준 왼쪽과 오른쪽을 나란히 놓았어요. 차이가 {ANGLE_WARN_DIFF}° 이상이면 빨간색이에요.</div>
<table class="pc-table">
  <thead><tr><th>왼쪽</th><th>부위</th><th>오른쪽</th><th>좌우 차이</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>"""


def reject_html(reason):
    return f"""
<div class="pc-reject">
  <h3>분석할 수 없는 사진이에요</h3>
  <p>{html.escape(reason)}</p>
  <ul>
    <li>지원 포즈는 프론트 더블 바이셉스와 백 더블 바이셉스예요.</li>
    <li>양 팔꿈치를 어깨 높이까지 올리고 주먹이 팔꿈치보다 위에 오게 해주세요.</li>
    <li>카메라는 정면 또는 정후면에 두고, 머리부터 발끝까지 나오게 찍어주세요.</li>
  </ul>
</div>"""


def coach_title_html():
    return """
<div class="pc-h">코치 노트</div>
<div class="pc-h-desc">위 수치를 바탕으로 Claude가 작성해요. 근육량과 컨디셔닝은 사진 속 관절만으로는 평가하지 않아요.</div>"""
