# 💪 AI 포징 코치 / AI Posing Coach

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit" />
  <img src="https://img.shields.io/badge/MediaPipe-0.10-00C853?logo=google" />
  <img src="https://img.shields.io/badge/Claude-claude--opus--4--8-8A2BE2?logo=anthropic" />
</p>

---

## 한국어

### 프로젝트 소개

클래식 피지크 선수를 위한 AI 포징 코치입니다.  
포즈 사진을 업로드하면 **MediaPipe**로 신체 랜드마크를 자동 감지하고, V-테이퍼 비율·좌우 대칭성·관절 각도 등 핵심 지표를 계산한 뒤 **Claude AI**가 한국어 코칭 피드백을 생성합니다.

### 동기

클래식 피지크 심사는 체형 비율과 사이즈, 대칭성, 심미성 등 많은 요소들이 심사 기준이 됩니다. 하지만 통상적으로 심사위원들의 주관적 판단과 성향에 따라 등수가 갈리기도 합니다. 이 프로젝트는 클래식 피지크 규정 포즈 1번(front double biceps)과 3번(back double biceps)에서 선수가 AI 포징 코치를 통해 자세를 개선하고 객관적인 평가를 받음으로써 더 나은 포징과 일관적인 자세를 유지할 수 있도록 돕기 위해 만들었습니다.

### 주요 기능

| 기능 | 설명 |
|---|---|
| **포즈 감지** | MediaPipe Pose로 33개 신체 랜드마크 추출 및 스켈레톤 시각화 |
| **V-테이퍼 비율** | 어깨 너비 / 골반 너비 비율 계산 (이상적 범위: 1.4–1.6) |
| **좌우 대칭성** | 어깨·골반 높이, 팔·다리 길이, 팔꿈치·무릎 각도 대칭 점수 (0–100) |
| **관절 각도** | 팔꿈치·무릎·어깨 외전·고관절 굴곡·체간 기울기 9개 항목 계산 |
| **AI 코칭 피드백** | Claude Opus 4.8이 지표를 종합해 한국어 코칭 텍스트 생성 |

### 기술 스택

| 영역 | 라이브러리 |
|---|---|
| 신체 포즈 추정 | `mediapipe` |
| 이미지 처리 | `Pillow`, `numpy` |
| 웹 UI | `streamlit` |
| AI 피드백 생성 | `anthropic` (Claude Opus 4.8) |
| 환경 변수 관리 | `python-dotenv` |

### 설치 및 실행

```bash
# 1. 저장소 클론
git clone https://github.com/<your-username>/ai-posing-coach.git
cd ai-posing-coach

# 2. 가상 환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 환경 변수 설정
# .env 파일을 생성하고 Anthropic API 키를 입력합니다.
echo ANTHROPIC_API_KEY=sk-ant-... > .env

# 5. 앱 실행
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 로 접속하세요.

### 프로젝트 구조

```
ai-posing-coach/
├── app.py              # Streamlit 앱 진입점 (UI, 파이프라인 연결)
├── src/
│   ├── metrics.py      # MediaPipe 랜드마크 → 지표 계산
│   └── feedback.py     # 지표 → Claude API 피드백 생성
├── tests/
│   └── test_metrics.py # metrics.py 단위 테스트 (32개)
├── requirements.txt
└── .env                # ANTHROPIC_API_KEY (git 미포함)
```

### 스크린샷

![결과 1](docs/images/AI-posing%20test%20result1.png)

![결과 2](docs/images/AI-posing%20test%20result2.png)

![결과 3](docs/images/AI-posing%20test%20result3.png)

---

## English

### Project Overview

An AI posing coach for Classic Physique athletes.  
Upload a pose photo and the app automatically detects 33 body landmarks via **MediaPipe**, computes key metrics (V-taper ratio, bilateral symmetry scores, joint angles), and generates actionable Korean-language coaching feedback using **Claude AI**.

### Motivation

Classic physique judging encompasses numerous criteria including physique proportions, muscle size, symmetry, and aesthetics. However, placements often vary depending on the subjective judgment and personal tendencies of individual judges.
This project was created to help competitors improve their posing and maintain consistent form by receiving AI-powered coaching feedback on the two mandatory poses in classic physique competition — Pose 1 (Front Double Biceps) and Pose 3 (Back Double Biceps) — enabling athletes to refine their technique and receive objective, data-driven evaluations.

### Key Features

| Feature | Description |
|---|---|
| **Pose Detection** | Extracts 33 body landmarks with MediaPipe Pose and overlays a skeleton |
| **V-Taper Ratio** | Shoulder width ÷ hip width (ideal range: 1.4–1.6) |
| **Bilateral Symmetry** | Per-metric scores (0–100) for shoulder/hip height, limb lengths, and joint angles |
| **Joint Angles** | 9 angles: elbows, knees, shoulder abduction, hip flexion, trunk lean |
| **AI Coaching Feedback** | Claude Opus 4.8 synthesises all metrics into structured Korean coaching text |

### Tech Stack

| Domain | Library |
|---|---|
| Body pose estimation | `mediapipe` |
| Image processing | `Pillow`, `numpy` |
| Web UI | `streamlit` |
| AI feedback generation | `anthropic` (Claude Opus 4.8) |
| Environment management | `python-dotenv` |

### Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ai-posing-coach.git
cd ai-posing-coach

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
# Create a .env file and add your Anthropic API key.
echo ANTHROPIC_API_KEY=sk-ant-... > .env

# 5. Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Project Structure

```
ai-posing-coach/
├── app.py              # Streamlit entry point (UI & pipeline wiring)
├── src/
│   ├── metrics.py      # MediaPipe landmarks → metric computation
│   └── feedback.py     # Metrics → Claude API feedback generation
├── tests/
│   └── test_metrics.py # Unit tests for metrics.py (32 tests)
├── requirements.txt
└── .env                # ANTHROPIC_API_KEY (not committed)
```

### Screenshots

![Result 1](docs/images/AI-posing%20test%20result1.png)

![Result 2](docs/images/AI-posing%20test%20result2.png)

![Result 3](docs/images/AI-posing%20test%20result3.png)

---

## Pipeline

```
Image Upload → MediaPipe Pose Detection → Metric Computation → Claude API → Feedback Display
```

## License

MIT
