# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI posing coach for IFBB Classic Physique. From one photo it detects body landmarks, classifies the pose as **Front Double Biceps (mandatory pose 1)** or **Back Double Biceps (pose 3)**, computes V-taper ratio, bilateral symmetry and joint angles, and asks Claude for Korean coaching notes. UI is a Streamlit web app deployed on Streamlit Community Cloud.

The owner is a beginner with Python and Git. Write code comments in **Korean**, keep changes small and explained, and prefer readable code over clever code.

## Setup (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
# .env 파일에 ANTHROPIC_API_KEY=sk-ant-... 입력
```

## Commands

```powershell
streamlit run app.py          # 앱 실행 (src/ 파일을 고친 뒤에는 Ctrl+C로 끄고 다시 실행)
python -m pytest              # 테스트
python src/pose.py [사진경로]  # 관절 감지만 단독 테스트 → samples/result.jpg
```

Always run from the project root.

## Architecture

Pipeline: upload → `pose.detect_landmarks` → `classify.classify_pose` → `classify.normalize_sides` → `metrics.compute_all_metrics` → `feedback.generate_feedback` → `ui` render.

- **`app.py`** — Streamlit entry point. Wires the pipeline; no styling logic.
- **`src/pose.py`** — MediaPipe **Tasks API** (`PoseLandmarker`). Downloads `pose_landmarker_full.task` into `models/` on first run (falls back to temp dir). `draw_skeleton` draws body joints only (11–32); face points are hidden.
- **`src/classify.py`** — Double-biceps check, then front/back from three weighted signals (nose vs ears depth, nose vs shoulders depth, left/right shoulder x order). Rejects ambiguous photos. Thresholds are constants at the top of the file. `normalize_sides` swaps L/R landmark pairs so labels match the athlete's own sides.
- **`src/metrics.py`** — V-taper, symmetry (0–100), joint angles. Pass the image size: `compute_all_metrics(landmarks, height, width)`; y is scaled by height/width so angles don't distort with photo aspect ratio.
- **`src/feedback.py`** — Claude API call; prompt adapts to `view` ("front"/"back").
- **`src/ui.py`** — "competition stage" design: black + single gold accent, Big Shoulders Display + IBM Plex Sans KR. Returns HTML strings; CSS lives here. No emojis in the UI.

## Gotchas

- `mp.solutions` no longer exists in current MediaPipe. Use `mediapipe.tasks.python.vision` only.
- MediaPipe landmarks are normalized 0–1 **per axis**. Any distance/angle math must correct for aspect ratio (see `metrics._points`, `classify._to_points`).
- Streamlit Cloud runs Linux and needs system libraries listed in `packages.txt` (`libgl1`, `libglib2.0-0`, `libegl1`, `libgles2`). If an import fails only on Cloud, check the `.so` dependencies before changing Python code.
- `samples/` and `models/` are git-ignored. Never commit personal or copyrighted photos.
- Screenshots in `docs/images/` must have faces and sensitive areas mosaicked.
