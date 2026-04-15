# Empathy Engine

A production-ready full-stack web application designed to analyze the emotional intent of text input and map it directly to voice synthesis parameters (rate, pitch, volume) using SSML.

## Features
- **Emotion Recognition**: Analyzes text using HuggingFace Transformers (`distilbert-base-uncased-emotion`), fallback to VADER sentiment rule-engine.
- **Dynamic Prosody Engine**: Maps detected emotion and scaled intensity directly into voice parameters.
- **SSML Generation**: Builds compliant SSML with word-level emphasis.
- **Fallbacks**: Switches from advanced TTS to local `pyttsx3` smoothly if credentials are missing or errors occur.
- **Minimal UI**: A strict black-and-white, highly professional frontend without flashy design elements.

## Architecture
- **Backend**: FastAPI
- **Emotion Adapter Layer**: `services.emotion`
- **TTS Adapter Layer**: `services.tts`
- **Frontend**: Plain HTML/Vanilla JS with purely functional styling.

## Local Setup

1. **Environment Config**
Copy `.env.example` to `.env` and fill in necessary fields.
```bash
cp .env.example .env
```

2. **Dependencies**
Install standard pip requirements:
```bash
pip install -r requirements.txt
```
*(Optionally setup your own venv)*

3. **Run Application**
Standard Uvicorn run loop:
```bash
python -m app.main
```
or 
```bash
uvicorn app.main:app --reload
```

## Testing
Run pytest for unit coverage:
```bash
pytest tests/
```
