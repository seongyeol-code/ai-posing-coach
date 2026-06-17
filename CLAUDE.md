# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI posing coach that analyzes classic physique poses from a photo — symmetry, proportions, and joint angles — and delivers AI coaching feedback via a Streamlit web app.

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables (copy .env and fill in API key)
# ANTHROPIC_API_KEY=<your key>
```

## Running the App

```bash
streamlit run app.py
```

## Architecture

The pipeline flows: image upload → MediaPipe pose detection → metric computation → Claude API feedback.

- **`app.py`** — Streamlit entry point; handles UI, image upload, and wires the pipeline together.
- **`src/matrix.py`** — Pose analysis: extracts MediaPipe landmarks and computes metrics (joint angles, symmetry scores, proportions).
- **`src/feedback.py`** — Sends computed metrics to the Claude API (`anthropic` SDK) and returns coaching feedback text.

### Key dependencies

| Library | Role |
|---|---|
| `mediapipe` | 33-landmark body pose detection |
| `opencv-python` | Image loading and preprocessing |
| `numpy` | Landmark coordinate math |
| `streamlit` | Web UI |
| `anthropic` | Claude API client for feedback generation |
| `python-dotenv` | Loads `ANTHROPIC_API_KEY` from `.env` |

The `ANTHROPIC_API_KEY` must be present in `.env` (loaded via `python-dotenv`) for feedback generation to work.
