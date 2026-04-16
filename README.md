<div align="center">
  
# 🎙️ Vaartalaap AI
**A Contextual, LLM-Driven Expressive Speech Synthesis Pipeline**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-2094f3.svg)
![Groq](https://img.shields.io/badge/Groq-Cloud-f54242.svg)
![Architecture](https://img.shields.io/badge/Architecture-Prosody_Aware-success.svg)

</div>

## 📖 Executive Summary
Vaartalaap AI is a sophisticated Text-to-Speech (TTS) orchestration engine designed to bridge the gap between robotic, monotonic TTS and human-like voice performance. Unlike standard TTS wrappers, Vaartalaap AI treats text generation as a **Speech Performance Pipeline**. 

It intelligently analyzes the emotional intent of the text, uses a Large Language Model (LLM) to extract prosodic intent (breath groups, pitch variation, speaking rate), automatically injects SSML (Speech Synthesis Markup Language), and handles complex code-mixed inputs (Hinglish/Hindi) without destructive translation.

This project was built to demonstrate advanced capabilities in **computational linguistics, conversational AI, and audio engineering pipeline design**—core competencies required for modern Voice AI architectures (comparable to systems at ElevenLabs or Sarvam AI).

---

## 🏗️ Architecture & Technology Stack

### 1. Core Framework & Concurrency
* **FastAPI (Python)**: The backbone of the application. Leverages asynchronous request handling (`asyncio`) to ensure non-blocking HTTP and WebSocket operations.
* **WebSocket Streaming**: Implements progressive, chunk-by-chunk audio generation and delivery (`app/api/routes/stream.py`). This drastically reduces Time To First Byte (TTFB), allowing the user to hear the opening sentence while the rest of the paragraph is still rendering in the background.

### 2. Speech Planning & Prosody Engine
* **Groq LLM Engine**: Acts as the "Speech Director." Instead of feeding raw text into a TTS engine, the text is first passed to a low-latency LLM running on Groq. The LLM chunks the text into natural "breath groups," determines emphasis words, and calculates dynamic percentage deltas for rate, pitch, and volume per segment.
* **SSML Builder** (`app/services/tts/ssml_builder.py`): Dynamically compiles standard text and LLM instructions into rich XML/SSML, applying `<prosody>` and `<break>` tags with micro-millisecond precision.

### 3. Emotion & Language Analysis
* **HuggingFace Transformers**: Utilizes `j-hartmann/emotion-english-distilroberta-base` for zero-shot text emotion classification, picking up on 7 core human emotions.
* **VADER Sentiment**: Acts as a robust, lightweight heuristic fallback when the neural model is unavailable or uncertain.
* **Custom Code-Mixed Language Detector** (`app/services/text/language_detector.py`): Contains a custom script and Romanized Hindi (Hinglish) heuristic classifier. It maps Devanagari, English, and Roman Hindi to enforce strict voice-localization (switching TTS underlying models from `en-US` to `hi-IN` or `en-IN` seamlessly).

### 4. Acoustic Synthesis Engines
* **Google Cloud TTS**: Utilizes WaveNet/Neural2 architectures.
* **Microsoft Edge Expressive TTS**: Monkey-patches Edge TTS to aggressively utilize `mstts:express-as` tags, mapping detected emotions (e.g., "grief" or "rage") directly to Azure's neural emotional styles.

### 5. Audio Post-Processing (DSP)
* **Pydub (FFmpeg)**: Applies post-synthesis Digital Signal Processing (DSP). Depending on the emotional intensity (derived from the NLP pipeline), the engine adjusts the EQ curve, normalizes volume, and splices audio files together.

### 6. Frontend Visualization
* **Native Web Audio API**: Features a custom-built, zero-dependency real-time spectrogram canvas. Parses byte-frequency data from the `<audio>` node to create a rolling, color-mapped visualization of the generated vocal frequencies.

---

## 🧠 Deep Dive: Key Engineering Logics

### 1. The Zero-Translation Guardrail (Language Preservation)
A critical challenge in modern LLMs modifying text for "human conversational cadence" is their tendency to translate code-mixed languages (like Hinglish) into standard English. 
**The Solution:**
* The `language_detector.py` scans inputs for Devanagari Unicode ranges and a dictionary of high-frequency Roman Hindi words.
* If a native language is detected, the LLM is injected with a dynamic, highly aggressive system prompt (`_SYSTEM_HINDI_CONSTRAINT`) that strictly prohibits translation, while still allowing the LLM to format the JSON prosody structure.
* The pipeline routes the output exclusively to localized TTS voices (e.g., `hi-IN-SwaraNeural`), setting the correct `xml:lang` tags.

### 2. Segment-Level Prosody Variation
Standard TTS sounds robotic because the speaking rate and pitch remain constant for an entire paragraph. 
**The Solution:**
* Vaartalaap AI splits a paragraph into logical clauses.
* The LLM generates a localized delivery arc (e.g., opening segment: standard rate; climax segment: +15% rate, +5st pitch).
* The backend parses these JSON deltas and converts them into cascading `<prosody>` SSML wrappers. For engines that don't support robust intra-sentence SSML, the application chunks the string, synthesizes them individually via concurrent `asyncio.gather`, and concatenates the resulting MP3 bytes via `pydub`.

### 3. Progressive WebSocket Streaming
To simulate human-like conversational latency:
* The `/api/v1/stream` endpoint accepts text over a persistent WebSocket.
* The backend analyzes the sentences, synthesizes them sequentially, and yields Base64-encoded chunked audio frames back to the client immediately.
* The frontend buffers and starts playback of chunk `n` while the server is concurrently rendering chunk `n+1`.

---

## 🔄 Detailed Pipeline Workflow & Edge Case Handling

The generation pipeline follows a stringent, stateful sequence. Below is the step-by-step logic, including how the system acts defensively against edge cases:

### Step 1: Input Reception & Language Triage
* **Logic**: Receives a raw string payload. Immediately invokes `detect_language`. Checks for Devanagari Unicode distributions and references a custom lexicon of 200+ Romanized Hindi/Hinglish terms (e.g., *yaar*, *matlab*, *achha*).
* **Edge Case Handling (Ambiguity)**: If the text is short or mathematically ambiguous, the system assigns a low-confidence Hinglish score. It defaults the NLP to English but safely routes the TTS output to an Indian-accented English voice (`en-IN`) rather than a pure Hindi voice (`hi-IN`), preventing heavy mispronunciation of English loan words.

### Step 2: Neural Emotion & Intensity Extraction
* **Logic**: The text passes through a HuggingFace Transformer (`j-hartmann/distilroberta-base`) mapping the context to 7 core emotions. High-probability confidence directly scales the `intensity` multiplier (e.g., 90% sadness yields a heavy slow-down in speaking rate).
* **Edge Case Handling (Model Timeout / Inference Failure)**: If the transformer API is unreachable, times out, or fails to initialize, the system seamlessly intercepts the error and routes the text to a deterministic `VaderSentimentFallback`. VADER processes the text via rule-based lexical heuristics to derive a fallback emotion, ensuring the API guarantees 100% uptime without 500 errors.

### Step 3: LLM Speech Prompting (The "Voice Actor" Strategy)
* **Logic**: The sanitized text and detected language state are forwarded to a Groq LLM (Llama / Mixtral) with JSON-mode enforced. The LLM acts as the "Speech Director," generating an array of `segments` (breath groups). For each segment, it calculates specific variance: `rate_delta_pct`, `pitch_delta_hz`, `pause_before_ms`, and an array of `emphasis_words`.
* **Edge Case Handling (Translation Hallucination)**: LLMs naturally want to translate code-mixed text to English. If `lang="hi-Latn"`, the system injects a fierce system constraint prohibiting translation. As an absolute fail-safe, the orchestrator explicitly discards the LLM's "humanized_text" rewrite to prevent English artifacts but *retains* the JSON prosodic metadata.
* **Edge Case Handling (LLM API Throttle)**: If Groq hits a rate limit, the system catches the exception, avoids crashing, and bypasses the LLM phase entirely—dropping back to deterministic, statically-mapped emotional rules.

### Step 4: SSML Engine & Contextual Variables
* **Logic**: The prosodic metadata and original text enter the `SSMLBuilder`. Targeted words are wrapped in `<emphasis>` tags. Segments are assigned their specific `<prosody>` configurations.
* **Edge Case Handling (Asynchronous Context Leakage)**: Because TTS generation happens concurrently inside `asyncio.gather()`, state collision is a high risk (e.g., an angry sentence adopting the "sad" voice of a parallel thread). To prevent this, the monkey-patched Edge TTS module uses Python's strict `contextvars` to securely isolate `mstts:express-as` tags per-task.
* **Edge Case Handling (Incompatible Voices)**: Native Hindi voice neural nodes (like `hi-IN-SwaraNeural`) occasionally reject `<mstts:express-as>` emotion tags, triggering a silent collapse in Azure. The pipeline acknowledges this constraint actively and strips style tags exclusively for the `hi-IN` routing, retaining only native prosody adjustments.

### Step 5: Backend Audio DSP & Splicing
* **Logic**: Raw MP3 byte-chunks arrive from the cloud engines. To smooth out harsh structural discrepancies between segment generations, `pydub` evaluates the intensity markers and applies localized volume curves/silence padding before dynamically splicing the segmented files into a singular waveform.

### Step 6: Progressive Streaming to Client
* **Logic**: Utilizing the WebSocket stream route, generated chunk `n+1` does not wait for chunk `n+n` to finish. The binary array is base64 encapsulated, shot to the user, and the native browser `<audio>` element plays it chronologically while the JS paints a rolling spectrogram derived from the browser's `AudioContext`.

---

## 🚀 How to Run Locally

1. **Clone the Repository**
2. **Environment Variables**: Provide your keys in a `.env` file:
   ```env
   GROQ_API_KEY=your_key
   GOOGLE_APPLICATION_CREDENTIALS=path/to/json
   HUGGINGFACE_EMOTION_MODEL=j-hartmann/emotion-english-distilroberta-base
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   # Ensure FFmpeg is installed and added to the system PATH
   ```
4. **Boot Server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
   ```

---

*This project highlights strong product thinking applied to Voice Generative AI—moving beyond direct API wrapping into orchestrating a seamless, emotionally intelligent audio generation pipeline.*
